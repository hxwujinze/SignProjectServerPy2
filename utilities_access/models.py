# coding:utf-8
import os
import pickle
import threading

from django.db import models

GESTURES_TABLE = ['肉 ', '鸡蛋 ', '喜欢 ', '您好 ', '你 ', '什么 ', '想 ', '我 ', '很 ', '吃 ',
                  '老师 ', '发烧 ', '谢谢 ', '']

access_lock = threading.RLock()
CURR_DATA_DIR = os.path.dirname(__file__) + '\\models_data\\'

# Create your models here.
class SignRecognizeHistory(models.Model):
    result_text = models.CharField(max_length=256, null=False, default="")
    middle_symbol = models.CharField(max_length=1024, null=False, default="")
    capture_armband = models.CharField(max_length=64, null=False, default="")
    sign_request_id = models.IntegerField(null=False, unique=False)
    capture_id = models.AutoField(primary_key=True)
    capture_date = models.DateTimeField(auto_now=True)
    correctness = models.BooleanField(null=False, default=False)

class SingleSignCapture(models.Model):
    sign_text = models.CharField(max_length=32, null=False, default='')
    capture_id = models.AutoField(primary_key=True)
    sign_sentence_belong_id = models.IntegerField(null=False, default=-1)
    correctness = models.BooleanField(null=False, default=False)
    middle_symbol = models.CharField(max_length=32, null=False, default='')
    raw_capture_data_acc = models.BinaryField(null=False, default='')
    raw_capture_data_emg = models.BinaryField(null=False, default='')
    raw_capture_data_gyr = models.BinaryField(null=False, default='')

class RecognizeInfo:
    def __init__(self, result_text, middle_symbol, raw_capture_data, capture_armband, sign_request_id):
        self.result_text = result_text
        self.middle_symbol = middle_symbol
        self.raw_capture_data = raw_capture_data
        self.capture_armband = capture_armband
        self.sign_request_id = sign_request_id
        self.capture_id = 0

    def create_recognize_history(self):
        db_obj = SignRecognizeHistory.objects.create(
            result_text=self.result_text,
            middle_symbol=self.middle_symbol,
            capture_armband=self.capture_armband,
            sign_request_id=self.sign_request_id,
        )
        self.capture_id = db_obj.capture_id

    """
    识别结果data定义 
    text 识别出的明文结果
    is_end 是否进行追加
        一次手语句子的识别是一次识别请求，由于手语是一个手势一个手势被识别出来的，
        一次手势的识别是一次识别过程。一个句子是由多个手语构成，所以一次识别请求由多次识别过程构成。
        每个识别过程的结果需要在过程完成后追加到识别的句子之后，
    sign_id 
        手语句子的id 每个手语句子具有一个id 可对一个句子进行多次识别 
        这个id是服务端与客户端共享的 客户端根据这个id对固定句子进行识别结果的更新
    capture_id 
        识别请求的id 每次识别请求都会分配一个id 这个id供服务端对识别历史的管理 
        这个id用于客户端向服务端反馈 这次手语识别的正误性
    """

    def get_feedback_data(self):
        data = {
            'control': 'update_recognize_res',
            'text': self.result_text,
            'sign_id': self.sign_request_id,
            'capture_id': self.capture_id,
        }
        return data

    def append_recognize_result(self, text, middle_symbol, raw_capture_data):
        self.result_text += str(text)
        self.middle_symbol += "\n" + str(middle_symbol)

        SingleSignCapture.objects.create(
            raw_capture_data_acc=pickle.dumps(raw_capture_data['acc']),
            raw_capture_data_emg=pickle.dumps(raw_capture_data['emg']),
            raw_capture_data_gyr=pickle.dumps(raw_capture_data['gyr']),
            middle_symbol=middle_symbol,
            sign_text=text,
            sign_sentence_belong_id=self.capture_id,
        )

        db_obj = SignRecognizeHistory.objects.filter(capture_id=self.capture_id)
        db_obj.update(result_text=self.result_text)
        db_obj.update(middle_symbol=self.middle_symbol)

def get_latest_sign_id():
    # 查询数据库获取当前序号
    access_lock.acquire()
    file_ = open(CURR_DATA_DIR + 'latest_sign_id', 'r+b')
    latest_sign_id = pickle.load(file_)
    file_.close()
    access_lock.release()
    return latest_sign_id

def acquire_latest_sign_id():
    access_lock.acquire()
    latest_sign_id = int(get_latest_sign_id())
    latest_sign_id = latest_sign_id + 1
    file_ = open(CURR_DATA_DIR + 'latest_sign_id', 'w+b')
    pickle.dump(latest_sign_id, file_)
    file_.close()
    access_lock.release()
    return latest_sign_id

def store_capture_feedback(capture_id, correctness):
    print("get recognize result feedback:\ncapture id: %s,correctness: %s" % (capture_id, str(correctness)))
    target = SignRecognizeHistory.objects.filter(capture_id=int(capture_id))
    target.update(correctness=correctness)
    single_signs = SingleSignCapture.objects.filter(sign_sentence_belong_id=int(capture_id))
    single_signs.update(correctness=True)



def pickle_cap_data2file():
    # 根据手势获取数据
    # 实际开发环境
    # data_list = []
    # for each_sign in GESTURES_TABLE:
    #     sign_history_items = SingleSignCapture.objects.filter(sign_text=each_sign)
    #     sign_id = GESTURES_TABLE.index(each_sign)
    #     data_set = parse_data(sign_history_items, sign_id=sign_id)
    #     data_list.extend(data_set)

    # 获取全部数据 不含手势内容
    # 适用于测试环境
    specified_sign_id = 5
    sign_history_items = SingleSignCapture.objects.all()
    data_list = [parse_data(sign_history_items, sign_id=specified_sign_id)]
    sign_history_items.delete()
    # 最后获得的列表是
    # [ (sign_id , 数据的字典 ) , .... ]

    file = open(CURR_DATA_DIR + 'feedback_data_', 'w+b')
    pickle.dump(data_list, file)
    file.close()

def parse_data(sign_history_items, sign_id=None):
    data_set = {
        'acc': [],
        'gyr': [],
        'emg': [],
        'text': ''
    }
    for each in sign_history_items:
        acc_data = pickle.loads(each.raw_capture_data_acc)
        gyr_data = pickle.loads(each.raw_capture_data_gyr)
        emg_data = pickle.loads(each.raw_capture_data_emg)
        data_set['acc'].append(acc_data)
        data_set['gyr'].append(gyr_data)
        data_set['emg'].append(emg_data)
    return sign_id, data_set
