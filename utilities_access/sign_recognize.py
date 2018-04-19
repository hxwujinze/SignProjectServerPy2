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
from sklearn.externals import joblib

from . import armbands_manager
from . import process_data
from .utilities_classes import Message

armbands_manager.update_connected_list()

GESTURE_SIZE = 160
CAPTURE_SIZE = 160

WINDOW_SIZE = 16
MAX_CAPTURE_TIME = 60

RNN_STATE = 566
SVM_STATE = 852



# todo 在这里进行更改识别算法
CURR_CLASSIFY_STATE = RNN_STATE

CURR_WORK_DIR = os.path.join(os.getcwd(), 'utilities_access')
CURR_DATA_DIR = CURR_WORK_DIR + '\\models_data'

# todo 这里键入python3路径 for pytroch运行
PYTORCH_INTP_PATH = 'C:\\Users\\Scarecrow\\AppData\\Local\\Programs\\Python\\Python36\\python.exe'
# PYTORCH_INTP_PATH = 'D:\\Anaconda3\\python.exe'

SCALE_DATA = np.loadtxt(CURR_DATA_DIR + "\\scale.txt")

# 这里将模型装载进来
CLF = joblib.load(CURR_DATA_DIR + "\\train_model.m")

TYPE_LEN = {
    'acc': 3,
    'gyr': 3,
    'emg': 8
}
CAP_TYPE_LIST = ['acc', 'emg', 'gyr']

GESTURES_TABLE = ['肉 ', '鸡蛋 ', '喜欢 ', '您好 ', '你 ', '什么 ', '想 ', '我 ', '很 ', '吃 ',  # 0-9
                  '老师 ', '发烧 ', '谢谢 ', '', '大家 ', '支持 ', '我们 ', '创新 ', '医生 ', '交流 ',  # 10 - 19
                  '团队 ', '帮助 ', '聋哑人 ', '请 ']  # 20 - 23
queue_lock = multiprocessing.Lock()


"""
手语识别工作线程
每次识别中 手环的连接时间是唯一的 几乎不会冲突
在没有其他可用的硬件唯一标识符时 连接时间可以代替使用
"""

class RecognizeWorker(multiprocessing.Process):

    def __init__(self, message_q,
                 armbands_timetag,
                 event):
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

        # 无法pickle序列化的类对象的实例化需要在进程的run方法里完成
        # 这里先用None 进行占位
        self.online_recognizer = None
        self.rnn_recg_proc = None

        # 手环数据采集周期
        self._t_s = 0.01
        self.EMG_captured_data = np.array([0])
        self.ACC_captured_data = np.array([0])
        self.GYR_captured_data = np.array([0])
        self.each_capture_gap = []

        # 与识别进程通信使用的管道
        self.pipe_input = None
        self.pipe_output = None

        # todo 这里设置识别模式 在线online 离线offline
        self.RECOGNIZE_MODE = 'online'


    #  setting start recognize flag
    def start_recognize(self):
        self.recognize_status.set()

    def stop_recognize(self):
        self.recognize_status.clear()

    def set_online_recognize_mode(self, online):
        if online:
            self.RECOGNIZE_MODE = 'online'
        else:
            self.RECOGNIZE_MODE = 'offline'

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
        # print("put msg in queue : %s " % message)
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
        if self.rnn_recg_proc is None:
            self.pipe_input, self.pipe_output, err_pipe, self.rnn_recg_proc = \
                generate_recognize_subprocces()

            err_printer = ErrorOutputListener(self.outer_event, err_pipe)
            err_printer.start()

        if self.RECOGNIZE_MODE == 'online':
            self.online_recognizer = OnlineRecognizer(self.message_q,
                                                      self.pipe_input,
                                                      self.pipe_output)

        self.pipe_input.write(self.RECOGNIZE_MODE + '\n')

        while not self.outer_event.is_set():
            # 外层循环  判断是否启动识别
            time.sleep(0.05)
            if self.recognize_status.is_set():
                print('RecognizeWorker start working')
                start_time = time.time()
                curr_time = start_time

                while self.recognize_status.is_set():
                    # 判断是否超时 每次手语采集具有时间限制 超时后停止手语采集
                    if curr_time - start_time > MAX_CAPTURE_TIME:
                        # 超时后停止识别工作
                        print("sign capture timeout , quiting")
                        self._stop_recognize("timeout")
                        # 返回外层循环进行等待
                        break
                    curr_time = time.time()

                    if self.RECOGNIZE_MODE == 'offline':
                        self.offline_recognize()
                    else:
                        self.online_recognize()

        print("recognize thread stopped\n")
        self.pipe_input.write('end\n')
        self.rnn_recg_proc.terminate()

    # 离线识别 先采集足够的数据 在进行识别
    def offline_recognize(self):

        # 开始采集手语
        self.capture_sign()
        # 采集完成后根据采集的数据进行识别 采集方法返回手语结果的序号
        if self.recognize_status.is_set():
            print("sign captured , starting recognizing ")
            sign_index = self.recognize_sign()
            print("recognizing complete\n")
            # 保存识别时采集的手环数据
            raw_capture_data = {
                'acc': self.ACC_captured_data,
                'emg': self.EMG_captured_data,
                'gyr': self.GYR_captured_data,
            }
            data = {
                'res_text': GESTURES_TABLE[sign_index],
                'middle_symbol': sign_index,
                'raw_data': raw_capture_data
            }
            # 向主线程返回识别的结果
            msg = Message(control='append_recognize_result',
                          data=data)
            self.put_message_into_queue(msg)
            time.sleep(0.8)

    # 在线识别 一边采集一边识别
    def online_recognize(self):
        self.get_left_armband_obj().vibrate(VibrationType.short)
        time.sleep(0.31)
        print("capture start at: %s" % time.clock())
        cap_start_time = time.clock()
        # 当处在进行识别状态时才采集数据
        while self.recognize_status.is_set():
            current_time = time.clock()
            gap_time = current_time - cap_start_time
            if gap_time >= self._t_s:
                myo_obj = self.get_left_armband_obj().myo_obj
                emg_data = tuple(myo_obj.emg)
                acc_data = list(myo_obj.acceleration)
                gyr_data = list(myo_obj.gyroscope)
                if self.recognize_status.is_set():
                    self.online_recognizer.append_data(acc_data, gyr_data, emg_data)
                cap_start_time = time.clock()

        # 结束一次采集后  将历史数据保存起来
        self.online_recognizer.store_raw_history_data()

        # 识别结束 震动2下
        self.get_left_armband_obj().vibrate(VibrationType.short)
        time.sleep(0.1)
        self.get_left_armband_obj().vibrate(VibrationType.short)

    """
    采集足够长度的手语数据进行识别
    """
    # 如果采集过程中手环失联 会导致线程阻塞
    def capture_sign(self):
        self.init_data()
        self.get_left_armband_obj().vibrate(VibrationType.short)
        time.sleep(0.31)
        print("capture start at: %s" % time.clock())
        cap_start_time = time.clock()
        sign_start_time = time.clock()
        # 当处在进行识别状态时才采集数据
        while self.recognize_status.is_set():
            current_time = time.clock()
            # 只有当采集到达末尾时才开始检查数据长度是否满足
            # 减少采集时的无关操作
            if current_time - sign_start_time > 1.5:
                if self.is_data_length_satisfied():
                    # 满足长度后震动手环两下 跳出采集循环
                    self.get_left_armband_obj().vibrate(VibrationType.short)
                    time.sleep(0.1)
                    self.get_left_armband_obj().vibrate(VibrationType.short)
                    print("capture ended at : %s" % time.clock())
                    break
            gap_time = current_time - cap_start_time
            if gap_time >= self._t_s:
                cap_start_time = time.clock()
                if len(self.paired_armbands) == 1:
                    myo_obj = self.get_left_armband_obj().myo_obj
                    emg_data = myo_obj.emg
                    acc_data = list(myo_obj.acceleration)
                    gyr_data = list(myo_obj.gyroscope)
                else:
                    myo_left_hand = self.get_left_armband_obj().myo_obj
                    myo_right_hand = self.get_right_armband_obj().myo_obj
                    # 根据myoProxy获取数据 双手直接通过列表拼接处理即可

                    emg_data = myo_left_hand.emg + myo_right_hand.emg
                    acc_data = list(myo_left_hand.acceleration) \
                               + list(myo_right_hand.acceleration)
                    gyr_data = list(myo_left_hand.gyroscope) \
                               + list(myo_right_hand.gyroscope)

                self.GYR_captured_data = vstack_data(self.GYR_captured_data, gyr_data)
                self.ACC_captured_data = vstack_data(self.ACC_captured_data, acc_data)
                self.EMG_captured_data = vstack_data(self.EMG_captured_data, emg_data)

    # 每次开始采集数据时用于初始化数据集
    def init_data(self):
        self.EMG_captured_data = np.array([0])
        self.ACC_captured_data = np.array([0])
        self.GYR_captured_data = np.array([0])
        self.each_capture_gap = []

    def is_data_length_satisfied(self):
        return len(self.EMG_captured_data) == CAPTURE_SIZE and \
               len(self.ACC_captured_data) == CAPTURE_SIZE and \
               len(self.GYR_captured_data) == CAPTURE_SIZE

    def is_armbands_sync(self):
        if len(self.paired_armbands) == 1:
            return self.get_left_armband_obj().is_sync
        else:
            return self.get_left_armband_obj().is_sync and \
                   self.get_right_armband_obj().is_sync

    # 识别手语 rnn svm 使用相同的数据处理方法
    def recognize_sign(self):
        # 如果已经设置为结束识别了 就不进行识别操作
        if not self.recognize_status.is_set():
            return
        acc_data = process_data.feature_extract_single(self.ACC_captured_data, 'acc')
        gyr_data = process_data.feature_extract_single(self.GYR_captured_data, 'gyr')
        emg_data = process_data.wavelet_trans(self.EMG_captured_data)
        # 选取三种特性拼接后的结果/
        acc_data_appended = acc_data[4]
        gyr_data_appended = gyr_data[4]
        emg_data_appended = emg_data
        # 再将三种采集类型进行拼接
        data_mat = process_data.append_single_data_feature(acc_data=acc_data_appended,
                                                           gyr_data=gyr_data_appended,
                                                           emg_data=emg_data_appended)
        if CURR_CLASSIFY_STATE == RNN_STATE:
            data_file_name = generate_data_seg_file(data_mat)
            # 通过pipe向之前启动的py3识别进程发送识别数据id
            self.pipe_input.write(data_file_name + '\n')
            res = self.pipe_output.readline()
            res = json.loads(res)
            print('**************************************')
            print('recognize result:')
            each_prob = 'each prob: \n' + res['each_prob']
            print(each_prob)
            print('max_prob: %s' % res['max_prob'])
            print('index: %d' % res['index'])
            print('raw_index: %d' % res['raw_index'])
            print('**************************************')

            return res['index']

        elif CURR_CLASSIFY_STATE == SVM_STATE:
            # 直接把每个windows的数据展开成一个长的向量 44 * 10
            res = int(CLF.predict(data_mat.ravel()))
            return res


    def get_left_armband_obj(self):
        return self.paired_armbands[0]

    def get_right_armband_obj(self):
        return self.paired_armbands[1]

class OnlineRecognizer:
    SEG_SIZE = 128

    def __init__(self, message_q, pipe_input, pipe_output):
        self.outer_msg_queue = message_q
        self.pipe_input = pipe_input
        self.pipe_output = pipe_output
        self.stop_flag = threading.Event()

        # 当前识别数据段的窗口指针
        self.window_start = 0
        self.window_end = 128

        # 数据缓冲区 用于接受采集的数据
        # 分别对应 acc gyr emg
        self.data_buffer = ([], [], [])

        # 数据处理线程 将采集的数据进行特征提取 scale 等工作
        self.data_processor = DataProcessor(pipe_input,
                                            pipe_output,
                                            self.stop_flag)
        self.data_processor.start()

        # 识别结果接受线程 当收到新的识别结果时
        # 将识别结果放入与工作线程通讯的消息队列 作为识别出的结果
        self.result_receiver = ResultReceiver(self.outer_msg_queue,
                                              self.pipe_output,
                                              self.stop_flag)
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

        # 当window所选的数据缓冲区都有数据了 将窗口内数据传给数据处理对象
        if len(self.data_buffer[0]) >= self.window_end:
            new_data_seg = [each_cap_type_buffer[self.window_start:self.window_end]
                            for each_cap_type_buffer in self.data_buffer]
            self.data_processor.new_data_queue.put(new_data_seg)
            self.window_end += 8
            self.window_start += 8

    def store_raw_history_data(self):
        # 将buffer内的数据作为原始的历史采集数据进行保存
        # 便于之后的分析
        data_history = {
            'acc': self.data_buffer[0],
            'gyr': self.data_buffer[1],
            'emg': self.data_buffer[2]
        }
        time_tag = time.strftime("%H-%M-%S", time.localtime(time.time()))
        file_ = open(CURR_DATA_DIR + '\\raw_data_history_' + time_tag, 'w+b')
        pickle.dump(data_history, file_)
        file_.close()
        self.data_buffer = ([], [], [])

    def stop_recognize(self):
        self.stop_flag.set()

class DataProcessor(threading.Thread):
    def __init__(self, input_pipe, output_pipe, stop_flag):
        threading.Thread.__init__(self,
                                  name='data_processor', )
        self.data_list = []
        self.new_data_queue = Queue.Queue()
        self.stop_flag = stop_flag

        self.input_pipe = input_pipe
        self.output_pipe = output_pipe

        self.processed_data_history = []
        self.processed_data_tags = []

    def run(self):
        data_mat_cnt = 0
        while not self.stop_flag.isSet():
            time.sleep(0.08)
            while not self.new_data_queue.empty():
                new_seg_data = self.new_data_queue.get()
                data_mat = self.create_data_seg(acc_data=new_seg_data[0],
                                                gyr_data=new_seg_data[1],
                                                emg_data=new_seg_data[2])
                data_file_name = generate_data_seg_file(data_mat)
                self.input_pipe.write(data_file_name + '\n')

                data_mat_cnt += 1
        self.input_pipe.write('end\n')

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
        return data_mat



class ResultReceiver(threading.Thread):
    def __init__(self, message_q, output_pipe, stop_flag):
        threading.Thread.__init__(self)
        self.message_q = message_q
        self.output_pipe = output_pipe
        self.stop_flag = stop_flag

    def run(self):
        while not self.stop_flag.is_set():
            res = self.output_pipe.readline()
            if res == '':
                continue
            res = json.loads(res)

            # 是空手语时直接跳过
            if res['index'] == 13:
                continue
            print('**************************************')
            print("online mode ")
            print('recognize result:')
            print('diff: %s' % res['diff'])
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
                            universal_newlines=True)
    pipe_input = rnn_sub_process.stdin
    pipe_output = rnn_sub_process.stdout
    pipe_err = rnn_sub_process.stderr
    print('pid %d' % rnn_sub_process.pid)
    return pipe_input, pipe_output, pipe_err, rnn_sub_process

def generate_data_seg_file(data_mat):
    data_id = random.randint(0, 9999999)
    data_file_name = str(data_id) + '.data'
    data_path = CURR_WORK_DIR + '\\' + data_file_name
    file_ = open(data_path, 'w+b')
    pickle.dump(data_mat, file_)
    file_.close()
    return data_file_name


# #################### rnn  sector ##############
#     import from process_data package
