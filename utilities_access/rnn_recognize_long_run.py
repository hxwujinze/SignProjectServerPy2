# coding:utf-8
import json
import os
import pickle
import threading
import time

import numpy as np
import queue
import torch
from torch.autograd import Variable

from RNN_model import LSTM

CURR_WORK_DIR = os.path.dirname(__file__)
CURR_DATA_DIR = os.path.join(CURR_WORK_DIR, 'models_data')


# 取最大值 并且转换为int 用于处理rnn输出
def getMaxIndex(tensor):
    prob_each_sign = torch.squeeze(tensor).data.numpy()
    max_res = torch.max(tensor, dim=1)
    max_value = max_res[0].data.float()[0]
    raw_index = max_res[1].data.int()[0]
    if max_value < 0.90:
        index = 13
    else:
        index = raw_index

    return_info = {
        'each_prob': str(prob_each_sign[:]),
        'max_prob': max_value,
        'index': index,
        'raw_index': raw_index,
    }
    return return_info

class RecognizeQueue(threading.Thread):
    def __init__(self, stop_flag, rnn_model):
        threading.Thread.__init__(self, name='recognize_queue')
        self.data_queue = queue.Queue()
        self.stop_flag = stop_flag
        self.ignore_cnt = 0
        self.rnn_model = rnn_model
        self.recognize_data_history = []

        self.result_vote = {}
        self.is_active = False


    def run(self):
        time_tag = time.strftime("%H_%M_%S", time.localtime(time.time()))
        file_name = os.path.join(CURR_DATA_DIR, 'history_recognized_data_' + time_tag)
        while not self.stop_flag.is_set():
            time.sleep(0.01)
            # 保存每次处理的数据mas
            while not self.data_queue.empty():
                new_msg = self.data_queue.get()
                if self.ignore_cnt != 0:
                    self.ignore_cnt -= 1
                    continue
                data_mat = new_msg
                output = self.rnn_model(data_mat).cpu()
                res = getMaxIndex(output)
                res['info'] = 'ok'
                if res['max_prob'] < 0.90:
                    res['info'] = 'skip this'

                # 添加数据并保存
                # 识别出手语的id
                # 时间tag 以及识别时采集下来的数据
                # 时间tag可用于排序

                history = {
                    'index': res['raw_index'],
                    'time': time.clock(),
                    'data': data_mat.data.numpy()[0]
                    # 这里的pytorch 2 numpy转化会将data转换为一个三维数组 但里面只有一个二维面
                    # 使用一个index取出这个二维面

                }

                if not self.is_active:
                    if res['raw_index'] != 13:
                        self.recognize_data_history.append(history)

                        self.is_active = True
                        if self.result_vote.get(res['raw_index']) is None:
                            cnt = 1
                            self.result_vote[res['raw_index']] = [res, cnt]
                        else:
                            self.result_vote[res['raw_index']][1] += 1
                else:
                    if res['raw_index'] != 13:
                        self.recognize_data_history.append(history)

                        if self.result_vote.get(res['raw_index']) is None:
                            cnt = 1
                            self.result_vote[res['raw_index']] = [res, cnt]
                        else:
                            self.result_vote[res['raw_index']][1] += 1
                    else:

                        max_occur_time = -1
                        max_occur_data = None
                        for each in self.result_vote.values():
                            if each[1] > max_occur_time:
                                max_occur_time = each[1]
                                max_occur_data = each[0]

                        self.is_active = False
                        self.result_vote = {}
                        if max_occur_time != 1:
                            res_str = json.dumps(max_occur_data)
                            print(res_str)

                        file_ = open(file_name, 'wb')
                        pickle.dump(self.recognize_data_history, file_)
                        file_.close()


    def add_new_data(self, data):
        self.data_queue.put(data)


    def stop_thread(self):
        self.stop_flag.set()


def main():
    # load model

    read_ = input()
    mode = read_
    stop_event = threading.Event()
    online_recognizer = None
    rnn_model = LSTM()
    if mode == 'online':
        rnn_model.load_state_dict(torch.load(CURR_DATA_DIR + '\\online_model.pkl'))
        online_recognizer = RecognizeQueue(stop_event, rnn_model)
        online_recognizer.start()
    else:
        for root, dirs, files in os.walk(CURR_DATA_DIR):
            for file_ in files:
                file_name_split = os.path.splitext(file_)
                if file_name_split[1] == '.pkl' and file_name_split[0] != 'online_model':
                    file_ = CURR_DATA_DIR + '\\' + file_
                    rnn_model.load_state_dict(torch.load(file_))
                    rnn_model.eval()
                    break

    data_mat_cnt = 0


    while True:
        read_ = input()
        if read_ == 'end':
            if online_recognizer is not None:
                online_recognizer.stop_thread()
            print("")
            break

        data_file_name = read_
        data_path = os.path.join(CURR_WORK_DIR, data_file_name)
        data_file = open(data_path, 'r+b')

        data_mat = pickle.load(data_file, encoding='iso-8859-1')
        data_file.close()
        data_mat = torch.from_numpy(np.array([data_mat])).float()
        data_mat = Variable(data_mat)
        os.remove(data_path)

        if mode == 'offline':
            output = rnn_model(data_mat)
            res = getMaxIndex(output)
            res = json.dumps(res)
            print(res)
        else:
            online_recognizer.add_new_data(data_mat)
            # 用于对比数据传入是否同步
            data_history = {
                'data_file_name': data_file_name,
                'data_num': data_mat_cnt,
            }
            recognize_data_tag_history.append(data_history)
            data_mat_cnt += 1

recognize_data_tag_history = []

if __name__ == '__main__':
    # pass
    main()
