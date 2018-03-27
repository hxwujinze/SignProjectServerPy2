# coding:utf-8
import json
import multiprocessing
import os
import pickle
import random
import time
from subprocess import Popen, PIPE

import numpy as np
import sklearn
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

CURR_CLASSIFY_STATE = RNN_STATE

CURR_WORK_DIR = os.path.join(os.getcwd(), 'utilities_access')
CURR_DATA_DIR = CURR_WORK_DIR + '\\models_data'

# todo 这里键入python3路径 for pytroch运行
PYTORCH_INTP_PATH = 'C:\\Users\\Scarecrow\\AppData\\Local\\Programs\\Python\\Python36\\python.exe'

SCALE_DATA = np.loadtxt(CURR_DATA_DIR + "\\scale.txt")

# 这里将模型装载进来
CLF = joblib.load(CURR_DATA_DIR + "\\train_model.m")

TYPE_LEN = {
    'acc': 3,
    'gyr': 3,
    'emg': 8
}
CAP_TYPE_LIST = ['acc', 'emg', 'gyr']

GESTURES_TABLE = ['肉 ', '鸡蛋 ', '喜欢 ', '您好 ', '你 ', '什么 ', '想 ', '我 ', '很 ', '吃 ',
                  '老师 ', '发烧 ', '谢谢 ', '']

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

        # 手环的identity
        self.paired_armbands = armbands_timetag

        # event用于终止线程
        self.outer_event = event
        # status用于设置是否进行手语识别工作
        # set时进行识别 unset时不识别 初始是unset
        self.recognize_status = multiprocessing.Event()

        self.rnn_recg_proc = ''

        # 手环数据采集周期
        self._t_s = 0.01
        self.EMG_captured_data = np.array([0])
        self.ACC_captured_data = np.array([0])
        self.GYR_captured_data = np.array([0])
        self.each_capture_gap = []

        self.pipe_input = ''
        self.pipe_output = ''

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
        # for armband in self.target_armband:
        #     armband.lock()

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
        if self.rnn_recg_proc == '':
            self.pipe_input, self.pipe_output, self.rnn_recg_proc = \
                generate_recognize_subprocces()
        while not self.outer_event.is_set():
            # 外层循环  判断是否启动识别
            if self.recognize_status.is_set():
                print('RecognizeWorker start recognize')
                start_time = time.time()
                curr_time = start_time

                # for armband in self.target_armband:
                #     armband.unlock()

                while self.recognize_status.is_set():
                    # 判断是否超时 每次手语采集具有时间限制 超时后停止手语采集
                    if curr_time - start_time > MAX_CAPTURE_TIME:
                        # 超时后停止识别工作
                        print("sign capture timeout , quiting")
                        self._stop_recognize("timeout")
                        # 返回外层循环进行等待
                        break
                    curr_time = time.time()
                    # 开始采集手语
                    self.capture_sign()
                    print("sign captured , starting recognizing ")
                    # 采集完成后根据采集的数据进行识别 采集方法返回手语结果的序号
                    sign_index = self.recognize_sign()
                    if self.recognize_status.is_set():
                        print("recognizing complete")
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
                        print("a sign capturing and recognizing complete")
                        time.sleep(0.8)
        print("recognize thread stopped")
        # pipe_input.write('end')

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
        while True:
            current_time = time.clock()
            # 只有当采集到达末尾时才开始检查数据长度是否满足
            # 减少采集时的无关操作
            if current_time - sign_start_time > 1.5:
                if self.is_data_length_satisfied():
                    # 满足长度后跳出采集循环
                    # print('each capture gap: ' + str(self.each_capture_gap))
                    self.get_left_armband_obj().vibrate(VibrationType.short)
                    time.sleep(0.1)
                    self.get_left_armband_obj().vibrate(VibrationType.short)
                    print("capture ended at : %s" % time.clock())
                    break
            gap_time = current_time - cap_start_time
            if gap_time >= self._t_s:
                # self.each_capture_gap.append(gap_time)
                cap_start_time = time.clock()
                # if not self.is_armbands_sync():
                #     print("armband didn't sync")
                #     self._stop_recognize("unsync")
                #     return
                if len(self.paired_armbands) == 1:
                    myo_obj = self.get_left_armband_obj().myo_obj
                    Emg = myo_obj.emg
                    Acc = [it for it in myo_obj.acceleration]
                    Gyr = [it for it in myo_obj.gyroscope]
                else:
                    myo_left_hand = self.get_left_armband_obj().myo_obj
                    myo_right_hand = self.get_right_armband_obj().myo_obj
                    # 根据myoProxy获取数据 双手直接通过列表拼接处理即可
                    Emg = myo_left_hand.emg + myo_right_hand.emg
                    Acc = [it for it in myo_left_hand.acceleration] \
                          + [it for it in myo_right_hand.acceleration]
                    Gyr = [it for it in myo_left_hand.gyroscope] \
                          + [it for it in myo_right_hand.gyroscope]

                self.GYR_captured_data = vstack_data(self.GYR_captured_data, Gyr)
                self.ACC_captured_data = vstack_data(self.ACC_captured_data, Acc)
                self.EMG_captured_data = vstack_data(self.EMG_captured_data, Emg)

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

    # 识别手语
    def recognize_sign(self):
        # 如果已经设置为结束识别了 就不进行识别操作
        if not self.recognize_status.is_set():
            return
        if CURR_CLASSIFY_STATE == SVM_STATE:

            ACC_ges_fea = online_fea_extraction(self.ACC_captured_data[0:GESTURE_SIZE, 0:3],
                                                WINDOW_SIZE,
                                                GESTURE_SIZE, 3)
            GYR_ges_fea = online_fea_extraction(self.GYR_captured_data[0:GESTURE_SIZE, 0:3],
                                                WINDOW_SIZE,
                                                GESTURE_SIZE, 3)

            EMG_ges_fea = online_fea_extraction(self.EMG_captured_data[0:GESTURE_SIZE, 0:8],
                                                WINDOW_SIZE,
                                                GESTURE_SIZE, 8)

            # 进行预测并输出手势的序号

            Combined_fea_list = np.hstack((EMG_ges_fea, ACC_ges_fea, GYR_ges_fea))
            Combined_fea_list_tmp = Combined_fea_list
            Combined_fea_list = np.vstack((Combined_fea_list, Combined_fea_list_tmp))

            max_abs_scaler = sklearn.preprocessing.MaxAbsScaler()
            max_abs_scaler.scale_ = SCALE_DATA

            norm_online_feat = max_abs_scaler.transform(Combined_fea_list)
            res = int(CLF.predict([norm_online_feat[0, :]]))
            return res
        else:
            # rnn 识别模式
            acc_data = process_data.feature_extract_single(self.ACC_captured_data, 'acc')
            gyr_data = process_data.feature_extract_single(self.GYR_captured_data, 'gyr')
            emg_data = process_data.wavelet_trans(self.EMG_captured_data)
            # 选取三种特性拼接后的结果
            acc_data_appended = acc_data[3]
            gyr_data_appended = gyr_data[3]
            emg_data_appended = emg_data
            # 再将三种采集类型进行拼接
            data_mat = process_data.append_single_data_feature(acc_data=acc_data_appended,
                                                               gyr_data=gyr_data_appended,
                                                               emg_data=emg_data_appended)

            data_id = random.randint(222, 9999999)
            data_file_name = str(data_id) + '.data'
            data_path = CURR_WORK_DIR + '\\' + data_file_name
            file_ = open(data_path, 'w+b')
            pickle.dump(data_mat, file_)
            file_.close()

            # target_python_dir = PYTORCH_INTP_PATH
            # target_script_dir = CURR_WORK_DIR + '\\rnn_recognize.py'
            #
            # command = '%s %s %s' % (target_python_dir, target_script_dir, data_file_name)
            # res = os.system(command)

            self.pipe_input.write(data_file_name + '\n')
            res = self.pipe_output.readline()

            res = json.loads(res)
            print('each prob: %s' % res['each_prob'])
            print('max_prob: %s' % res['max_prob'])
            print('index: %d' % res['index'])
            print('raw_index: %d' % res['raw_index'])
            return res['index']

    def get_left_armband_obj(self):
        return self.paired_armbands[0]

    def get_right_armband_obj(self):
        return self.paired_armbands[1]

def vstack_data(target, step_data):
    step_data = np.array(step_data)
    if target.any() == 0:
        # 第一步 如果为空 直接赋值
        target = step_data
    else:
        # >1 步时则是使用np.vstack 将新的数据追加为矩阵新的维度
        target = np.vstack((target, step_data))
    return target

def online_fea_extraction(online_feat, window_size, gesture_size, width_data):
    # 这里意思大概是要将整个手语数据分成一个一个window的去跑匹配
    split_size = gesture_size / window_size
    window_samples = np.vsplit(online_feat, split_size)
    # 由于每次采集的数据是以矩阵叠加向量的方式存储的
    # vsplit是vstack的逆过程 他将矩阵"横向"进行拆分成一个个小矩阵
    RMS_feat_level = np.zeros((1, 3))
    ZC_feat_level = np.zeros((1, 3))
    ARC_feat_level = np.zeros((1, 4))
    window_index = 0
    for window_piece in window_samples:
        RMS_feat = np.mean(np.sqrt(np.square(window_piece)), axis=0)

        arg_1 = window_piece[1::, :]
        arg_2 = np.zeros((1, width_data))

        piece_move1 = np.vstack((arg_1, arg_2))
        ZC_feat = np.sum((-np.sign(window_piece) * np.sign(piece_move1) + 1) / 2, axis=0)
        ARC_feat = np.apply_along_axis(ARC3ord, 0, window_piece)
        # 在这里也是迭代
        if window_index == 0:
            RMS_feat_level = RMS_feat
            ZC_feat_level = ZC_feat
            ARC_feat_level = ARC_feat
        else:
            RMS_feat_level = np.vstack((RMS_feat_level, RMS_feat))
            ZC_feat_level = np.vstack((ZC_feat_level, ZC_feat))
            ARC_feat_level = np.vstack((ARC_feat_level, ARC_feat))
        window_index += 1
    temp_gest_sample_feat = np.vstack((RMS_feat_level, ZC_feat_level, ARC_feat_level))
    get_gest_feat = temp_gest_sample_feat.T.ravel()
    return get_gest_feat

def ARC3ord(Orin_Array):
    t_value = len(Orin_Array)
    AR_coeffs = np.polyfit(range(t_value), Orin_Array, 3)
    return AR_coeffs

def generate_recognize_subprocces():
    # init recognize process
    target_python_dir = PYTORCH_INTP_PATH
    target_script_dir = CURR_WORK_DIR + '\\rnn_recognize_long_run.py'
    command = target_python_dir + ' ' + target_script_dir
    rnn_sub_process = Popen(args=command,
                            shell=True,
                            stdin=PIPE,
                            stderr=PIPE,
                            stdout=PIPE,
                            universal_newlines=True)
    pipe_input = rnn_sub_process.stdin
    pipe_output = rnn_sub_process.stdout
    print('pid %d' % rnn_sub_process.pid)
    return pipe_input, pipe_output, rnn_sub_process

# #################### rnn  sector ##############
#     import from process_data package
