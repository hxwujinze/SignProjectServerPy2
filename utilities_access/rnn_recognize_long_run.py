# coding:utf-8
import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

CURR_WORK_DIR = os.path.dirname(__file__)
CURR_DATA_DIR = CURR_WORK_DIR + '\\models_data'

INPUT_SIZE = 44
NNet_SIZE = 30
NNet_LEVEL = 3
CLASS_CNT = 24

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,  # feature's number
            # 2*(3+3+3*4) + 8
            #    36            8
            hidden_size=NNet_SIZE,  # hidden size of rnn layers
            num_layers=NNet_LEVEL,  # the number of rnn layers
            batch_first=True,
            dropout=0.4)
        # dropout :
        # 在训练时，每次随机（如 50% 概率）忽略隐层的某些节点；
        # 这样，我们相当于随机从 2^H 个模型中采样选择模型；同时，由于每个网络只见过一个训练数据

        self.out = nn.Linear(NNet_SIZE, CLASS_CNT)  # use soft max classifier.
        self.out2 = nn.Linear(CLASS_CNT, CLASS_CNT)

    def forward(self, x):
        lstm_out, (h_n, h_c) = self.lstm(x, None)
        out = self.out(lstm_out[:, -1, :])
        out = F.relu(out)
        # return out
        out2 = self.out2(out)
        out2 = F.softmax(out2, dim=1)
        return out2

# 取最大值 并且转换为int 用于处理rnn输出
def getMaxIndex(tensor):
    prob_each_sign = torch.squeeze(tensor).data.numpy()
    max_res = torch.max(tensor, dim=1)
    max_value = max_res[0].data.float()[0]
    raw_index = max_res[1].data.int()[0]
    if max_value < 0.992:
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

# if __name__ == '__main__':
rnn_model = LSTM()
for root, dirs, files in os.walk(CURR_DATA_DIR):
    for file_ in files:
        if os.path.splitext(file_)[1] == '.pkl':
            file_ = CURR_DATA_DIR + '\\' + file_
            rnn_model.load_state_dict(torch.load(file_))
            rnn_model.eval()
            break

# file_ = open('log','w')
# file_.write('process start ' + time.strftime('%H-%M-%S' ,time.localtime(time.time())) + '\n')
# file_.close()

while True:
    read_ = input()
    if read_ == 'end':
        break

    # file_ = open('log', 'a')
    # file_.write('data ' + read_ +' '+ time.strftime('%H-%M-%S', time.localtime(time.time())) + '\n')
    # file_.close()

    data_file = read_
    data_path = CURR_WORK_DIR + '\\' + data_file
    data_file = open(data_path, 'r+b')
    data_mat = pickle.load(data_file, encoding='iso-8859-1')
    data_file.close()
    os.remove(data_path)

    data_mat = torch.from_numpy(np.array([data_mat])).float()
    data_mat = Variable(data_mat)
    output = rnn_model(data_mat)
    res = getMaxIndex(output)
    res = json.dumps(res)
    # file_ = open('log', 'a')
    # file_.write( res + ' '+ time.strftime('%H-%M-%S', time.localtime(time.time())) + '\n')
    # file_.close()
    print(res)

# file_ = open('log','w')
# file_.write('process exit ' + time.strftime('%H-%M-%S',time.localtime(time.time())))
# file_.close()
