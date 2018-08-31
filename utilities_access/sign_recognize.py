# coding:utf-8
import Queue
import json
import multiprocessing
import os
import pickle
import random
import threading
import time
from subprocess import Popen, PIPE

import numpy as np
from myo.lowlevel import VibrationType

from . import armbands_manager
from . import my_pickle
from . import process_data
from .utilities_classes import Message

# armbands_manager.update_connected_list()

GESTURE_SIZE = 160
CAPTURE_SIZE = 160

WINDOW_SIZE = 16
MAX_CAPTURE_TIME = 90

RNN_STATE = 566
SVM_STATE = 852

# 在这里进行更改识别算法
CURR_CLASSIFY_STATE = RNN_STATE

CURR_WORK_DIR = os.path.join(os.getcwd(), 'utilities_access')
CURR_DATA_DIR = os.path.join(CURR_WORK_DIR, 'models_param')

# 这里键入python3路径 for pytroch运行
# PYTORCH_INTP_PATH = 'C:\\Users\\Scarecrow\\AppData\\Local\\Programs\\Python\\Python36\\python.exe'
PYTORCH_INTP_PATH = 'D:\\Anaconda3\\python.exe'


# 这里将模型装载进来
# CLF = joblib.load(CURR_DATA_DIR + "\\train_model.m")


GESTURES_TABLE = [u'朋友 ', u'下午 ', u'天 ', u'早上 ', u'上午 ', u'中午 ', u'谢谢 ', u'对不起 ', u'没关系 ', u'昨天 ', u'今天 ',
                  u'明天 ', u'家 ', u'回 ', u'去 ', u'迟到 ', u'交流 ', u'联系 ', u'你 ', u'什么 ', u'想 ', u'我 ', u'机场 ', u'晚上 ',
                  u'卫生间 ', u'退 ', u'机票 ', u'着急 ', u'怎么 ', u'办 ', u'行李 ', u'可以 ', u'托运 ', u'起飞 ', u'时间 ', u'错过 ',
                  u'改签 ', u'航班 ', u'延期 ', u'请问 ', u'怎么走 ', u'在哪里 ', u'找 ', u'不到 ', u'没收 ', u'为什么 ', u'航站楼 ',
                  u'取票口 ', u'检票口 ', u'身份证 ', u'手表 ', u'钥匙 ', u'香烟 ', u'刀 ', u'打火机 ', u'沈阳 ', u'大家 ',
                  u'支持 ', u'我们 ', u'医生 ', u'帮助 ', u'聋哑人 ', u'', u'充电 ', u'寄存 ', u'中国 ', u'辽宁 ', u'北京 ',
                  u'世界 ']


queue_lock = multiprocessing.Lock()
data_scaler = process_data.DataScaler(CURR_DATA_DIR)


"""
手语识别工作线程
每次识别中 手环的连接时间是唯一的 几乎不会冲突
在没有其他可用的硬件唯一标识符时 连接时间可以代替使用
"""

class RecognizeWorker(multiprocessing.Process):

    def __init__(self, message_q,
                 armbands_timetag,
                 event,
                 recognize_mode):
        multiprocessing.Process.__init__(self, name='RecognizeWorker')
        print('RecognizeWorker created')
        # 接受识别结果的主线程
        self.message_q = message_q
        # 已经匹配 要作为识别数据源的手环对象
        print('armbands_list timestamp: ' + str(armbands_manager.connect_time_obj_map))

        # 手环的identity 使用手环的连接时间tag
        self.paired_armbands = armbands_timetag

        # event用于终止线程
        self.outer_event = event
        # status用于设置是否进行手语识别工作
        # set时进行识别 unset时不识别 初始是unset
        self.recognize_status = multiprocessing.Event()

        # 设置识别模式 在线online
        self.recognize_model = 'online'

        # 无法pickle序列化的类对象的实例化需要在进程的run方法里完成
        # 这里先用None 进行占位
        self.online_recognizer = None
        self.recg_proc = None

        # 手环数据采集周期
        self._t_s = 0.01
        self.start_time = None

        self.EMG_captured_data = np.array([0])
        self.ACC_captured_data = np.array([0])
        self.GYR_captured_data = np.array([0])
        self.each_capture_gap = []

        # 与识别进程通信使用的管道
        self.input_pipe = None
        self.output_pipe = None



    def is_timeout(self):
        """
        每次检查时 会作为一次开始
        根据start time 是否为None确定识别是否开始
        :return:
        """
        if self.start_time is None:
            self.start_time = time.clock()
        return time.clock() - self.start_time > MAX_CAPTURE_TIME


    #  setting start recognize flag
    def start_recognize(self):
        self.recognize_status.set()

    def stop_recognize(self):
        self.recognize_status.clear()



    def _stop_recognize(self, stop_type):
        self.recognize_status.clear()
        # 向主线程发送消息 通知已经停止手语识别了
        msg = Message(control='stop_recognize', data={"type": stop_type})
        self.put_message_into_queue(msg)

    """
    向主线程消息队列中添加消息的方法 多线程数据访问需要加锁解锁
    该方法封装了这一过程
    """

    def put_message_into_queue(self, message):
        global queue_lock
        queue_lock.acquire(True)
        self.message_q.put(message)
        queue_lock.release()

    """
    震动两下开始手语识别
    两下后会有个0.5s的gap 作为人的反应时间 
    震动一下结束采集
    """

    def run(self):
        armbands_timetag = self.paired_armbands
        self.paired_armbands = []
        for each in armbands_timetag:
            self.paired_armbands.append(armbands_manager.connect_time_obj_map[each])
        if self.recg_proc is None:
            self.input_pipe, self.output_pipe, err_pipe, self.recg_proc = \
                generate_recognize_subprocces()
            # 错误子进程监听线程 当出现错误时 print到主进程上
            err_printer = ErrorOutputListener(self.outer_event, err_pipe)
            err_printer.start()
            self.online_recognizer = OnlineRecognizer(self.message_q,
                                                      self.input_pipe,
                                                      self.output_pipe)

        while not self.outer_event.is_set():
            # 外层循环  判断是否启动识别
            time.sleep(0.01)
            if self.recognize_status.is_set():
                print('RecognizeWorker start working')
            while self.recognize_status.is_set():
                self.online_recognize()


        print("recognize thread stopped\n")
        self.input_pipe.write('end\n')
        self.recg_proc.terminate()
        self.recg_proc.wait()


    # 在线识别 一边采集一边识别
    def online_recognize(self):
        self.get_right_armband_obj().vibrate(VibrationType.short)
        time.sleep(0.15)
        print("online recognize start at: %s" % time.clock())
        cap_start_time = time.clock()
        # 当处在进行识别状态时才采集数据
        while self.recognize_status.is_set():
            if self.is_timeout():
                print("sign capture timeout , quiting")
                self._stop_recognize("timeout")
                self.start_time = None
                break
            current_time = time.clock()
            gap_time = current_time - cap_start_time
            if gap_time >= self._t_s:
                myo_obj = self.get_right_armband_obj().myo_obj
                emg_data = tuple(myo_obj.emg)
                acc_data = list(myo_obj.acceleration)
                gyr_data = list(myo_obj.gyroscope)
                if self.recognize_status.is_set():
                    self.online_recognizer.append_data(acc_data, gyr_data, emg_data)
                cap_start_time = time.clock()

        # debug 结束一次采集后  将历史数据保存起来
        # self.online_recognizer.data_processor.store_raw_history_data()
        self.online_recognizer.stop_recognize()
        print("online recognize end at: %s" % time.clock())
        # 识别结束 震动2下
        self.get_right_armband_obj().vibrate(VibrationType.short)
        time.sleep(0.1)
        self.get_right_armband_obj().vibrate(VibrationType.short)

    def get_right_armband_obj(self):
        return self.paired_armbands[0]



class OnlineRecognizer:
    SEG_SIZE = 128

    def __init__(self, message_q, pipe_input, pipe_output):
        self.outer_msg_queue = message_q
        self.timer_queue = Queue.Queue()
        self.pipe_input = pipe_input
        self.pipe_output = pipe_output
        # 停止工作标记 set之后退出线程
        self.stop_flag = threading.Event()
        # 在线识别工作启用标记 被set之后 在线识别被启用

        # 当前步进数据段的窗口指针
        self.step_win_start = 0
        self.step_win_end = random.randint(8, 20)

        # 数据缓冲区 用于接受采集的数据
        # 分别对应 acc gyr emg
        self.data_buffer = [[], [], []]

        # 数据处理线程 将采集的数据进行特征提取 scale 等工作
        self.data_processor = DataProcessor(pipe_input,
                                            pipe_output,
                                            self.stop_flag,
                                            self.timer_queue)
        self.data_processor.start()
        # 识别结果接受线程 当收到新的识别结果时
        # 将识别结果放入与工作线程通讯的消息队列 作为识别出的结果
        self.result_receiver = ResultReceiver(self.outer_msg_queue,
                                              self.pipe_output,
                                              self.stop_flag,
                                              self.timer_queue)
        self.result_receiver.start()



    def append_data(self, acc_data, gyr_data, emg_data):
        """
        每当采集一个数据时 追加到在线识别的数据buffer中
        移动窗口 判断当前是否是一个采集window
        :param acc_data: 一次cap的acc
        :param gyr_data: gyr
        :param emg_data: emg
        """
        new_data = (acc_data, gyr_data, emg_data)
        for each_cap_type in range(len(new_data)):
            # 将三种数据追加到各自数据种类的buffer中
            self.data_buffer[each_cap_type].append(new_data[each_cap_type])

        # 当步进数据缓冲区都有数据了 将数据传给数据处理对象
        if len(self.data_buffer[0]) >= self.step_win_end:
            new_data_seg = self.data_buffer
            self.data_processor.new_data_queue.put(new_data_seg)
            self.clean_buffer()  # 传递完成后将步进数据缓冲区重置

    def clean_buffer(self):
        self.step_win_end = random.randint(14, 24)
        # 随机值的窗口步进 避免数据阻塞 也能一定程度提高分辨率
        self.step_win_start = 0
        self.data_buffer = ([], [], [])

    def stop_recognize(self):
        self.stop_flag.clear()

class DataProcessor(threading.Thread):
    def __init__(self, input_pipe, output_pipe, stop_flag, timer_queue):
        threading.Thread.__init__(self,
                                  name='data_processor', )
        self.new_data_queue = Queue.Queue()
        self.timer_queue = timer_queue
        self.stop_flag = stop_flag
        self.input_pipe = input_pipe
        self.output_pipe = output_pipe

        self.normalized_data_buffer = {
            'acc': None,
            'gyr': None,
            #    不需要normalize emg数据 直接使用raw 即可
        }

        self.raw_data_buffer = {
            'acc': None,
            'gyr': None,
            'emg': None,
        }
        self.start_ptr = 0
        self.end_ptr = 0
        self.extract_ptr_start = 0
        self.extract_ptr_end = 128


    def run(self):
        while not self.stop_flag.isSet():
            time.sleep(0.08)
            while not self.new_data_queue.empty():
                self.append_raw_data()
                if self.end_ptr >= self.extract_ptr_end:
                    step_size = self.end_ptr - self.extract_ptr_end
                    self.extract_ptr_end = self.end_ptr
                    self.extract_ptr_start = self.extract_ptr_end - 128
                    # print("extract windows (%d, %d) step size %d" %
                    #       (self.extract_ptr_start, self.extract_ptr_end, step_size))
                    self.feat_extract_and_send()
        # 保存历史数据
        # self.store_raw_history_data()
        print("data processor stopped")
        self.input_pipe.write('end\n')
        self.input_pipe.flush()
        return

    def append_raw_data(self):
        new_seg_data = self.new_data_queue.get()
        type_list = ['acc', 'gyr', 'emg']
        for each_type_index in range(len(type_list)):
            each_type_name = type_list[each_type_index]
            if self.raw_data_buffer[each_type_name] is None:
                self.raw_data_buffer[each_type_name] = np.array(new_seg_data[each_type_index])
            else:
                self.raw_data_buffer[each_type_name] = \
                    np.vstack((self.raw_data_buffer[each_type_name], new_seg_data[each_type_index]))
        self.end_ptr += len(new_seg_data[0])  # 更新buffer长度


    def feat_extract_and_send(self):
        self.timer_queue.put(time.clock())
        acc_data = self.raw_data_buffer['acc'][self.extract_ptr_start:self.extract_ptr_end, :]
        gyr_data = self.raw_data_buffer['gyr'][self.extract_ptr_start:self.extract_ptr_end, :]
        emg_data = self.raw_data_buffer['emg'][self.extract_ptr_start:self.extract_ptr_end, :]
        data_mat = self.create_data_seg(acc_data, gyr_data, emg_data)
        self._send_to_recognize_process(data_mat)

    @staticmethod
    def create_data_seg(acc_data, gyr_data, emg_data):
        acc_data = np.array(acc_data)
        gyr_data = np.array(gyr_data)
        emg_data = np.array(emg_data)
        acc_data = process_data.feature_extract_single_polyfit(acc_data, 2)
        gyr_data = process_data.feature_extract_single_polyfit(gyr_data, 2)
        emg_data = process_data.wavelet_trans(emg_data)
        emg_data = process_data.expand_emg_data_single(emg_data)
        # 将三种采集类型进行拼接
        data_mat = process_data.append_single_data_feature(acc_data=acc_data,
                                                           gyr_data=gyr_data,
                                                           emg_data=emg_data)
        data_mat = data_scaler.normalize(data_mat, 'cnn')
        data_mat = np.where(data_mat > 0.00000000001, data_mat, 0)
        return data_mat

    def _send_to_recognize_process(self, data_mat):
        data_pickle_str = my_pickle.dumps(data_mat)
        # print data_pickle_str + '\n\n'
        self.input_pipe.write(data_pickle_str + '\n')
        self.input_pipe.flush()  # 用flush保证字节流及时被传入识别线程

    def store_raw_history_data(self):
        # 将buffer内的数据作为原始的历史采集数据进行保存
        # 便于之后的分析
        data_history = self.raw_data_buffer
        time_tag = time.strftime("%H-%M-%S", time.localtime(time.time()))
        file_ = open(os.path.join(CURR_DATA_DIR, 'raw_data_history_' + time_tag), 'w+b')
        pickle.dump(data_history, file_)
        file_.close()
        self.clean_buffer()

    def clean_buffer(self):

        self.raw_data_buffer = {
            'acc': None,
            'gyr': None,
            'emg': None,
        }
        self.start_ptr = 0
        self.end_ptr = 0
        self.extract_ptr_start = 0
        self.extract_ptr_end = 128





# 在线识别启动时 启用识别结果接受线程
class ResultReceiver(threading.Thread):
    def __init__(self, message_q, output_pipe, stop_flag, timer_queue):
        threading.Thread.__init__(self)
        self.message_q = message_q
        # 向主线程返回消息的消息队列
        self.timer_q = timer_queue
        # 计时器队列
        self.output_pipe = output_pipe
        # 来自识别算法线程的输出pipe
        self.stop_flag = stop_flag
        # 线程退出时的终止标记


    def run(self):
        while True:
            res = self.output_pipe.readline()
            try:
                res = json.loads(res)
            except ValueError:
                # 取出上次加入阻塞在pipe里的内容
                # print (res)
                if res == 'end':
                    print ("receiver stop working ")
                    return
                else:
                    continue
            time_tag = self.timer_q.get()
            end_time = time.clock()
            cost_time = end_time - time_tag
            print('**************************************')
            print('online recognize result:')
            print('diff: %s' % str(res['diff']))
            print('index: %d' % res['index'])
            print('verify_result: %s' % res['verify_result'])
            print('**************************************')
            sign_index = res['index']
            if res['verify_result'] == 'True':
                raw_capture_data = {
                    'acc': [],
                    'emg': [],
                    'gyr': [],
                }
                data = {
                    'res_text': GESTURES_TABLE[sign_index],
                    'middle_symbol': sign_index,
                    'raw_data': raw_capture_data
                }
                # 向主线程返回识别的结果
                msg = Message(control='append_recognize_result',
                              data=data)
                if not self.stop_flag.is_set():
                    self.message_q.put(msg)



class ErrorOutputListener(threading.Thread):
    def __init__(self, stop_flag, pipe):
        threading.Thread.__init__(self, name='Err listening thread')
        self.stop_flag = stop_flag
        self.pipe = pipe

    def run(self):
        while not self.stop_flag.is_set():
            err_info = self.pipe.readlines()
            time.sleep(0.001)
            if len(err_info) != 0:
                print('rnn subprocess error info:')
                for each_line in err_info:
                    if each_line != '':
                        print(each_line)


def vstack_data(target, step_data):
    step_data = np.array(step_data)
    if target.any() == 0:
        # 第一步 如果为空 直接赋值
        target = step_data
    else:
        # >1 步时则是使用np.vstack 将新的数据追加为矩阵新的维度
        target = np.vstack((target, step_data))
    return target

def generate_recognize_subprocces():
    # init recognize process
    target_python_dir = PYTORCH_INTP_PATH
    target_script_dir = CURR_WORK_DIR + '\\recognize_long_run.py'
    command = target_python_dir + ' ' + target_script_dir
    rnn_sub_process = Popen(args=command,
                            shell=True,
                            stdin=PIPE,
                            stderr=PIPE,
                            stdout=PIPE,
                            universal_newlines=True, )

    pipe_input = rnn_sub_process.stdin
    pipe_output = rnn_sub_process.stdout
    pipe_err = rnn_sub_process.stderr
    print('recognize subprocess started, pid %d' % rnn_sub_process.pid)
    return pipe_input, pipe_output, pipe_err, rnn_sub_process
