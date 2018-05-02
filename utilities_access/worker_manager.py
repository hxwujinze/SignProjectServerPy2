# coding:utf-8
import json
import multiprocessing
import socket
import threading
import time
from multiprocessing import Queue

from . import armbands_manager
from . import models as dj_models
from .sign_recognize import RecognizeWorker
from .utilities_classes import Message

'''一个类要用多进程时，self储存的变量类型必须是python里原有的，不能是自定义的一个新类'''


# 单例模式管理
'''
    python 的模块本身就是一个单例模式
    当被import的时候被实例化一次 
    然后每次通过模块名调用都是相当于访问单例
'''

# band_id <-> socket

# 端口号下放至客户端进行连接 连接传给手环管理对象进行连接的工作
# 手环传入回调类或者消息队列 连接线程和手环控制线程互换消息队列
# 通过队列完成相互之间的信息传递


def set_up_connection(pair_armbands):
    connect_worker = MainWorkerThread()
    armbands_manager.occupy_armbands(connect_worker, pair_armbands)
    connect_worker.start()
    return connect_worker.get_available_port()

'''
主线程建立连接 然后将连接交由连接线程进行使用
建立连接后 启动子线程 将连接的地址端口以http响应的形式发给终端 Django的http阶段结束
终端收到响应后开始尝试连接 子线程等待终端的连接直至超时
终端也尝试连接直至超时 。
成功则以保持socket长连接 进行双向的数据交换 直到双方决定停止连接
两端都维护一个线程监听对方发来的内容 以及发送内容回去
客户端与服务端的通讯由这个套接字完成
'''

queue_lock = multiprocessing.Lock()

"""
    这个线程算是整个手语识别工作线程组中的主线程
    这个线程负责socket连接数据的发送以及响应其他线程加入消息队列中消息（响应请求）
    它下属有一个socket连接的监听线程 用于监听从客户端发来的消息
    自己维护一个消息队列 从中获得响应请求，根据请求的进行处理
    其内部有一个长循环 在没有工作时不断扫描消息队列，检查是否有新的消息
    若有则去执行 执行完毕后继续扫描消息队列
（查看 取出 添加消息的过程都是要对队列上锁的）
"""

class MainWorkerThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, name='MainWorkerThread')
        self.__find_available_port()
        self.message_q = Queue()
        # 消息队列
        self.client_connect_sock = None
        # socket连接对象
        self.curr_process_request_id = -1
        # 当前处理的手语请求id
        self.curr_recognize_res = None
        # 当前处理的手语识别请求的结果的对象
        self.recognize_process = None
        # 手语识别线程对象
        self.recognize_event = multiprocessing.Event()
        # 用于通知手语识别线程何时退出的event对象
        self.recognize_mode = multiprocessing.Event()
        # 设置手语识别模式 True is online False is offline

    """
    该方法用于获取一个可用的监听接口用于socket连接
    在启动socket对象的连接之前，获取去一个当前空闲的临时接口作为连接接口，
    避免与操作系统中其他程序的接口发生冲突
    """

    def __find_available_port(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ip = "127.0.0.1"
        try:
            sock.connect((ip, 1111))
            """
            python socket的connect方法会让os选择一个空闲的端口分给程序进行socket连接
            这个端口号会记录在socket对象中 用getsockname方法取出并保存
            利用这个机制 我们可以获取一个空闲的端口用于服务端的监听 同时也不影响其他程序
            """
        except:
            addr, self.port = sock.getsockname()
            sock.close()

    def get_available_port(self):
        return self.port

    """
    线程的实际工作代码
    """

    def run(self):
        # 建立套接字 等待连接
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.settimeout(10000)
        server_socket.bind(("0.0.0.0", self.port))
        print("start socket at %d " % self.port)
        server_socket.listen(1)

        # 连接成功后 根据连接获得的socket对象，启动监听线程
        self.client_connect_sock, addr = server_socket.accept()
        print("accept from %s connection" % str(addr))
        print("socket name: %s\nsocket peer: %s" %
              (self.client_connect_sock.getsockname(), self.client_connect_sock.getpeername()))

        listener_thread_event = multiprocessing.Event()
        listener_thread = ListenerThread(self.message_q,
                                         self.client_connect_sock,
                                         listener_thread_event)
        listener_thread.start()

        armbands = armbands_manager.get_occupied_armbands(self)
        # 初始化手语识别线程

        armbands_tags = []
        for each in armbands:
            time_tag = int(str(each.myo_obj.connect_time))
            armbands_tags.append(time_tag)

        self.recognize_process = RecognizeWorker(self.message_q,
                                                 armbands_tags,
                                                 self.recognize_event,
                                                 self.recognize_mode)
        self.recognize_process.start()
        # 进入工作线程的主循环 扫描消息队列
        self.standby_loop()
        # 循环退出说明主线程的工作结束了
        print("工作线程退出")
        # 通知其他线程停止工作 并释放资源
        listener_thread_event.set()
        print("connect release")
        self.recognize_event.set()
        print("stop recognize thread")
        armbands_manager.release_armbands(self)
        print("release armbands")

    """
    工作线程的主循环，以一定频率不断不断扫描消息队列
    从中获得消息并根据消息进入响应工作
    """

    def standby_loop(self):
        print("working thread start")
        global queue_lock
        while True:
            queue_lock.acquire(True)
            if not self.message_q.empty():
                msg = self.message_q.get(block=False)
                # print('process on msg: %s' % msg)
                queue_lock.release()
                if msg.control == 'end_connection':
                    return True
                self.dispatch(msg)
            else:
                queue_lock.release()
            time.sleep(0.04)

    '''
    实际收到的是个json数据包 ，先进行解包，转换为python对象（下面代码中的Message类）
    json数据包分为控制字段和数据字段 控制字段说明工作种类 数据字段保存所需的数据
    数据字段还是以json的形式存储 保证其结构化 解包的时候也能转换成python对象 
    各个方法自行处理数据字段中的内容
    通过字典对象的maping，根据控制字段的内容执行相应的响应方法，并将数据字段分发到指定的方法中去
    '''

    """
     通过字典对象的maping，根据控制字段的内容执行相应的响应方法
    """

    def dispatch(self, msg):
        self.task_maping[msg.control](self, msg.data)

    """
    向主线程消息队列中添加消息的方法 多线程数据访问需要加锁解锁
    该方法封装了这一过程
    """

    def put_message_into_queue(self, message):
        global queue_lock
        queue_lock.acquire(True)
        # print("put msg in queue : %s " % message)
        self.message_q.put(message)
        queue_lock.release()

    """
    下面是各个响应方法
    发生消息
    """

    def send_message(self, info):
        info_str = json.dumps(info, indent=2)
        info_str += "$"
        # 总是使用 $ 作为每次消息的结尾
        try:
            self.client_connect_sock.settimeout(3)
            self.client_connect_sock.sendall(info_str.encode())
        except socket.timeout:
            self.resend(info_str)
        # print('have send message %s' % info_str)

    def resend(self, info_str):
        print("sending timeout , resend ")
        try:
            self.client_connect_sock.settimeout(3)
            self.client_connect_sock.sendall(info_str.encode())
        except socket.timeout:
            self.resend(info_str)

    def sign_recognize(self, info):

        # only it's -1 ,means worker thread standby for recognize
        # only one recognize task can be ran by work at the same time
        if self.curr_process_request_id != -1:
            return

        print("sign_recognize start,\n recognize info : %s" % str(info))
        if info['sign_id'] == 0:
            curr_sign_id = dj_models.acquire_latest_sign_id()
        else:
            curr_sign_id = info['sign_id']
        recognize_info = dj_models.RecognizeInfo('',
                                                 "middle_symbol:",
                                                 "raw_capture_data:",
                                                 info['armband_id'],
                                                 curr_sign_id)
        recognize_info.create_recognize_history()
        self.curr_recognize_res = recognize_info
        # 在这里启动手语识别的工作
        self.recognize_process.start_recognize()
        # 手语识别结束使用消息队列方式发送消息 避免阻塞工作线程
        msg = Message(control='send_msg',
                      data=recognize_info.get_feedback_data())
        self.put_message_into_queue(msg)

    #  识别结果追加的工作在主线程做
    def append_recognize_result(self, data):
        # 在stop了之后的手语一概忽略
        if self.curr_recognize_res is None:
            print("recognize has stopped , append task abandon")
            return
        print('recognize_result: %s \n' % str(data['res_text']))
        self.curr_recognize_res.append_recognize_result(data['res_text'],
                                                        data['middle_symbol'],
                                                        data['raw_data'])
        # 向客户端发消息 更新识别结果
        msg = Message(control='send_msg',
                      data=self.curr_recognize_res.get_feedback_data())
        self.put_message_into_queue(msg)

    def stop_recognize(self, data):
        data = {
            # end_recognize 的消息是由dispatcher接受的 接收后退出工作loop
            'control': 'end_recognize',
            'sign_id': self.curr_recognize_res.sign_request_id,
            'type': 'by client' if data.get("type") is None else data['type']
        }
        msg = Message(control='send_msg', data=data)
        self.curr_recognize_res = None
        self.recognize_process.stop_recognize()
        self.curr_process_request_id = -1
        print("recognize stopped ")
        self.put_message_into_queue(msg)

    def switch_recognize_model(self, data):
        """
        切换识别模式 在线 or 离线
        :param data: data 的 mode 字段存有 online 或offline
        """
        if data['mode'] == 'online':
            self.recognize_mode.set()
        else:
            self.recognize_mode.clear()

    task_maping = {
        'send_msg': send_message,
        # 发送消息
        'sign_cognize_request': sign_recognize,
        # 手语识别请求
        'append_recognize_result': append_recognize_result,
        # 追加手语识别结果
        'stop_recognize': stop_recognize,
        # 停止手语识别
        'switch_recognize_mode': switch_recognize_model,
    }

    @staticmethod
    def recvall(sock):
        data = b''
        while True:
            more = sock.recv(128)
            if more is "":
                break
            else:
                data += more
        return data

"""
socket监听线程
"""

class ListenerThread(threading.Thread):
    """
    outer_queue : 收到消息后消息的发送目标
    listened_socket : 如其名
    event : 接受外界消息的对象 可用于终止监听
    """

    def __init__(self, message_q,
                 listened_socket,
                 event):
        threading.Thread.__init__(self, name='ListenerThread')
        self.message_q = message_q
        self.listened_socket = listened_socket
        self.outer_event = event

    def run(self):
        print("listener thread start")
        # event 初始化flag是false
        empty_cnt = 0
        while not self.outer_event.is_set():
            # 不阻塞 1s查看一次状态
            try:
                self.listened_socket.settimeout(0.2)
                get = self.listened_socket.recv(128)
                if get != b'':
                    print("receive text %s" % get)
                    empty_cnt = 0
                    # 收到非空的消息将其加入主线程的消息队列
                    self.put_message_into_queue(Message(get))
                else:
                    empty_cnt += 1
                    if empty_cnt > 20:
                        print("receive many empty strs, client close connect")
                        self.put_message_into_queue(Message(control="end_connection", data=None))
                        return
            except socket.timeout:
                pass
        print("listener thread ended ")

    def put_message_into_queue(self, message):
        global queue_lock
        queue_lock.acquire(True)
        # print("put msg in queue : %s " % message)
        self.message_q.put(message)
        queue_lock.release()
