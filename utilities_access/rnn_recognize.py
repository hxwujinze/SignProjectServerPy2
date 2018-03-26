# coding:utf-8
import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

CURR_WORK_DIR = os.path.dirname(__file__)
CURR_DATA_DIR = CURR_WORK_DIR + '\\models_data'

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=44,  # feature's number
            # 2*(3+3+3*4) + 8
            #    36            8
            hidden_size=25,  # hidden size of rnn layers
            num_layers=3,  # the number of rnn layers
            batch_first=True,
            dropout=0.4)
        # dropout :
        # 在训练时，每次随机（如 50% 概率）忽略隐层的某些节点；
        # 这样，我们相当于随机从 2^H 个模型中采样选择模型；同时，由于每个网络只见过一个训练数据

        self.out = nn.Linear(25, 14)  # use soft max classifier.
        self.out2 = nn.Linear(14, 14)

    def forward(self, x):
        lstm_out, (h_n, h_c) = self.lstm(x, None)
        out = F.relu(lstm_out)
        out = self.out(lstm_out[:, -1, :])
        out = F.relu(out)
        # return out
        out2 = self.out2(out)
        out2 = F.softmax(out2, dim=1)
        return out2

# 取最大值 并且转换为int 用于处理rnn输出
def getMaxIndex(tensor):
    prob_each_sign = tensor.data.numpy()
    print('prob_each_sign:\n' + str(prob_each_sign))
    tensor = torch.max(tensor, dim=1)[1]
    return torch.squeeze(tensor).data.int()

if __name__ == '__main__':

    rnn_model = LSTM()
    for root, dirs, files in os.walk(CURR_DATA_DIR):
        for file in files:
            if os.path.splitext(file)[1] == '.pkl':
                file = CURR_DATA_DIR + '\\' + file
                rnn_model.load_state_dict(torch.load(file))

    parser = argparse.ArgumentParser(description='rnn recognize process')
    parser.add_argument('source', type=str)
    args = parser.parse_args()

    data_path = CURR_WORK_DIR + '\\' + args.source
    data_file = open(data_path, 'r+b')
    data_mat = pickle.load(data_file, encoding='iso-8859-1')
    data_file.close()
    os.remove(data_path)

    data_mat = torch.from_numpy(np.array([data_mat])).float()
    data_mat = Variable(data_mat)
    output = rnn_model(data_mat)
    res = getMaxIndex(output)[0]

    # print(str(res))

    sys.exit(res)
