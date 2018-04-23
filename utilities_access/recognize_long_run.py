# coding:utf-8
import json
import os
import pickle
import threading
import time

import numpy as np
import queue
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import my_pickle
from CNN_model import CNN, get_max_index
from RNN_model import LSTM
from verify_model import SiameseNetwork

CURR_WORK_DIR = os.path.dirname(__file__)
CURR_DATA_DIR = os.path.join(CURR_WORK_DIR, 'models_data')


class RecognizeQueue(threading.Thread):
    def __init__(self, stop_flag):
        threading.Thread.__init__(self, name='recognize_queue')
        self.data_queue = queue.Queue()
        # 存放已经处理好的numpy对象 供模型识别
        self.stop_flag = stop_flag

        self.cnn_model = CNN()  # classify model
        load_model_param(self.cnn_model, 'cnn_model')
        self.cnn_model.double()
        self.cnn_model.eval()
        self.cnn_model.cpu()

        self.verify_model = SiameseNetwork(train=False)  # verify model
        load_model_param(self.verify_model, 'verify_model')
        self.verify_model.double()
        self.verify_model.eval()
        self.verify_model.cpu()

        vector_file_path = os.path.join(CURR_DATA_DIR, 'reference_verify_vector')
        file_ = open(vector_file_path, 'rb')
        self.reference_vectors = pickle.load(file_)  # reference vector
        file_.close()

        self.recognize_data_history = []
        self.is_redundant = False

    def run(self):
        while not self.stop_flag.is_set():
            time.sleep(0.01)
            while not self.data_queue.empty():
                new_msg = self.data_queue.get()
                data_mat = np.array([new_msg.T])
                data_mat = torch.from_numpy(data_mat).double()
                data_mat = Variable(data_mat)
                classify_output = self.cnn_model(data_mat)
                predict_index = get_max_index(classify_output)[0]
                verify_result, diff = self.verify_correctness(data_mat, predict_index)

                if verify_result:
                    if predict_index == 13:
                        continue
                    if not self.is_redundant:
                        self.is_redundant = True
                        return_info = {
                            'index': predict_index,
                            'diff': diff,
                            'verify_result': str(verify_result)
                        }
                        print(json.dumps(return_info))
                else:
                    self.is_redundant = False

                # return_info['data'] = new_msg
                # self.recognize_data_history.append(return_info)



    def verify_correctness(self, data, predict_index):
        """
        :return: 验证的正误以及 差异程度
        """
        data_vector = self.verify_model(data)
        reference_vector = np.array([self.reference_vectors[predict_index + 1]])
        reference_vector = Variable(torch.from_numpy(reference_vector).double())
        diff = F.pairwise_distance(data_vector, reference_vector)
        diff = torch.squeeze(diff).data[0]
        if diff > 0.12:
            return False, diff
        else:
            return True, diff

    def add_new_data(self, data):
        self.data_queue.put(data)

    def clean_data_queue(self):
        self.data_queue = queue.Queue()


    def stop_thread(self):
        # 保存历史数据
        time_tag = time.strftime("%H_%M_%S", time.localtime(time.time()))
        file_name = os.path.join(CURR_DATA_DIR, 'history_recognized_data_' + time_tag)
        file_ = open(file_name, 'wb')
        pickle.dump(self.recognize_data_history, file_)
        file_.close()

        self.recognize_data_history = []
        self.stop_flag.set()

def load_model_param(model, model_type_name):
    for root, dirs, files in os.walk(CURR_DATA_DIR):
        for file_ in files:
            file_name_split = os.path.splitext(file_)
            if file_name_split[1] == '.pkl' and file_name_split[0].startswith(model_type_name):
                file_ = CURR_DATA_DIR + '\\' + file_
                model.load_state_dict(torch.load(file_))
                model.eval()
                return model

# 取最大值 并且转换为int 用于处理rnn输出
def generate_offline_recognize_result(tensor):
    prob_each_sign = torch.squeeze(tensor).data.numpy()
    max_res = torch.max(tensor, dim=1)
    max_value = max_res[0].data.float()[0]
    raw_index = max_res[1].data.int()[0]
    if max_value < 0.20:
        index = 13
    else:
        index = raw_index

    return_info = {
        'type': 'offline',
        'each_prob': str(prob_each_sign),
        'max_prob': max_value,
        'index': index,
        'raw_index': raw_index,
    }
    return return_info


def main():
    # load model

    read_ = input()
    mode = read_
    stop_event = threading.Event()
    offline_rnn_model = LSTM()
    online_recognizer = RecognizeQueue(stop_event)
    online_recognizer.start()
    offline_rnn_model = load_model_param(offline_rnn_model, 'rnn_model')


    while True:
        read_ = input()
        if read_ == 'end':
            if online_recognizer is not None:
                online_recognizer.stop_thread()
            print("")
            break
        if read_.endswith('line'):
            mode = read_
            if mode == 'offline':
                online_recognizer.clean_data_queue()
                print('clean stab')  # 用于清空上次在线识别阻塞住的readline
            continue

        data_mat = my_pickle.loads(read_)

        if mode == 'offline':
            data_mat = torch.from_numpy(np.array([data_mat])).float()
            data_mat = Variable(data_mat)
            output = offline_rnn_model(data_mat)
            res = generate_offline_recognize_result(output)
            res = json.dumps(res)
            print(res)
        else:
            online_recognizer.add_new_data(data_mat)


if __name__ == '__main__':
    main()
