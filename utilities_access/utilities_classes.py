# coding:utf-8
import json
import os

"""
message 示例
msg = {
    'control': 'control_type',
    'data': 还是 json , 解包时会递归的都转换成python对象
}
"""

"""
这个Message类是用于在服务端各个线程进行沟通时的消息对象
control属性为消息携带的控制信息 通过此活的处理消息的方法
data 为消息携带的在方法执行时需要的数据 
"""

class Message:
    def __init__(self, json_str=None, control=None, data=None):
        if json_str is not None:
            info_obj = json.loads(json_str)
            self.control = info_obj['control']
            self.data = info_obj['data']
            if self.data == '':
                self.data = {}
        else:
            self.control = control
            self.data = data

    def __str__(self):
        obj = {
            'control': self.control,
            'data': self.data,
        }
        return json.dumps(obj, indent=2)

CURR_WORK_DIR = os.path.dirname(__file__) + '\\models_data'
