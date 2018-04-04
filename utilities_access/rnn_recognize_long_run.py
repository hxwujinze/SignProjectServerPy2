# coding:utf-8
import json
import os
import pickle
import threading

import numpy as np
import queue
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

CURR_WORK_DIR = os.path.dirname(__file__)
CURR_DATA_DIR = CURR_WORK_DIR + '\\models_data'

INPUT_SIZE = 44
NNet_SIZE = 30
NNet_LEVEL = 3
NNet_OUTPUT = 24
CLASS_CNT = 24
DROPOUT = 0.5

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,  # feature's number
            # 2*(3+3+3*4) + 8
            #    36       + 8
            hidden_size=NNet_SIZE,  # hidden size of rnn layers
            num_layers=NNet_LEVEL,  # the number of rnn layers
            batch_first=True,
            dropout=DROPOUT)
        self.out = nn.Linear(NNet_SIZE, NNet_OUTPUT)  # use soft max classifier.
        self.out2 = nn.Linear(NNet_OUTPUT, CLASS_CNT)

    def forward(self, x):
        lstm_out, (h_n, h_c) = self.lstm(x, None)
        out = F.relu(lstm_out)
        out = self.out(out[:, -1, :])
        out = F.relu(out)
        out2 = self.out2(out)
        out2 = F.softmax(out2, dim=1)
        return out2

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
        'max_prob': '%f' % max_value,
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

    def run(self):
        while not self.stop_flag.is_set():
            while not self.data_queue.empty():
                new_msg = self.data_queue.get()
                if self.ignore_cnt != 0:
                    self.ignore_cnt -= 1
                    continue
                data_mat = new_msg
                output = self.rnn_model(data_mat).cpu()
                res = getMaxIndex(output)
                if res['max_prob'] < 0.90:
                    res['info'] = 'skip this'
                elif res['raw_index'] != 13:
                    self.ignore_cnt = 7
                res = json.dumps(res)
                print(res)

    def add_new_data(self, data):
        self.data_queue.put(data)

    def stop_thread(self):
        self.stop_flag.set()

def main():
    # load model
    rnn_model = LSTM()
    for root, dirs, files in os.walk(CURR_DATA_DIR):
        for file_ in files:
            if os.path.splitext(file_)[1] == '.pkl':
                file_ = CURR_DATA_DIR + '\\' + file_
                rnn_model.load_state_dict(torch.load(file_))
                rnn_model.eval()
                break

    read_ = input()
    mode = read_
    stop_event = threading.Event()
    online_recognizer = ''
    if mode == 'online':
        online_recognizer = RecognizeQueue(stop_event, rnn_model)
        online_recognizer.start()
    while True:
        read_ = input()
        if read_ == 'end':
            if online_recognizer != '':
                online_recognizer.stop_thread()
            break

        data_file = read_
        data_path = CURR_WORK_DIR + '\\' + data_file
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
            online_recognizer.put_data(data_mat)

if __name__ == '__main__':
    main()
