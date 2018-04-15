# coding:utf-8
import os
import threading

import myo
from myo.lowlevel.enums import VibrationType, StreamEmg

lock = threading.RLock()

# 手环id与手环对象的映射 通过该映射可以用手环id获取手环对象实例
armband_id_obj_map = {}

# 由于没有统一硬件标识符
# 在一次手语识别实例运行时 连接时间不会发生变化
# 当做统一标识符凑合用了
connect_time_obj_map = {}

# 工作线程和手环配对:
# worker_thread -> myo_obj
# 工作线程可通过自身的对象 从manager获取已经匹配成功的手环
# 在识别过程中从手环中获取数据
thread_armbands_pair_book = {}

# armbands_obj
# 所有已连接的手环的对象实例 （Class Armband）
armbands_connected = []

"""
封装myo手环对象的类，便于整个程序进行管理
"""

class MyoFeedback(myo.Feed):
    def __init__(self):
        myo.Feed.__init__(self)
        myo.isSync = False

    def on_arm_sync(self, myo, timestamp, arm, x_direction, rotation,
                    warmup_state):
        print("armband sync 0x{0:x}".format(myo.value))
        update_connected_list()
        armband_id = armband_id_assign_book[myo.value]
        armband_id_obj_map[armband_id].is_sync = True

    def on_arm_unsync(self, myo, timestamp):
        print("armband unsync 0x{0:x}".format(myo.value))
        update_connected_list()
        armband_id = armband_id_assign_book[myo.value]
        armband_id_obj_map[armband_id].is_sync = False

class Armband:
    def __init__(self, armband_myo_obj):
        self.armband_id = assign_armband_id(armband_myo_obj)
        armband_id_obj_map[self.armband_id] = self
        self.myo_obj = armband_myo_obj
        self.myo_obj.set_stream_emg(StreamEmg.enabled)
        self.is_occupy = False
        self.is_sync = False

    def vibrate(self, vibrate_type):
        self.myo_obj.vibrate(vibrate_type)

    def unlock(self):
        self.myo_obj.unlock()

    def lock(self):
        self.myo_obj.lock()

    def is_occupied(self):
        return self.is_occupy

    def __str__(self):
        return u'armband_id: %s\nmyo_obj: %s\nis_occupied: %s\nconnect_time: %s' % \
               (self.armband_id, self.myo_obj, str(self.is_occupied()), str(self.myo_obj.connect_time))

    def __cmp__(self, other):
        return self.myo_obj.__hash__() == other.__hash__()

    def __hash__(self):
        return self.myo_obj.__hash__()

def get_armbands_list():
    update_connected_list()
    return armbands_connected

"""
更新已连接手环实例 根据手环对象的hash value进行判重
同时会更新手环id与手环对象的dict
"""
def armbands_list_find(target_list, elem):
    for each in target_list:
        if each.__cmp__(elem):
            return each
    return None

def update_connected_list():
    global armbands_connected
    lock.acquire(True)
    device_list = myo_feed.get_connected_devices()
    new_armband_objs = []
    # 将myo对象转化为内部的手环对象
    for each_raw_device in device_list:
        new_armband_objs.append(Armband(each_raw_device))
    # 对比之前的手环列表 去旧填新

    for each in armbands_connected:
        curr_armband = armbands_list_find(new_armband_objs, each)
        if curr_armband is None and each.is_occupy == False:
            armbands_connected.remove(each)
        elif not (curr_armband is None):
            new_armband_objs.remove(curr_armband)

    armbands_connected += new_armband_objs

    for each_armband in armbands_connected:
        time_tag = int(str(each_armband.myo_obj.connect_time))
        connect_time_obj_map[time_tag] = each_armband

    lock.release()
    print("curr armbands list :")
    for each in armbands_connected:
        print(str(each) + '\n')

"""
占用手环
将手环标记为已占用 并通过dict记录占用手环的识别线程与手环对象的映射关系
"""

def occupy_armbands(working_thread, armbands_id):
    lock.acquire(True)
    print("occupying armbands %s" % str(armbands_id))
    thread_armbands_pair_book[working_thread] \
        = [armband_id_obj_map[each] for each in armbands_id]
    for each in armbands_id:
        armband_id_obj_map[each].is_occupy = True
    lock.release()

"""
释放手环
"""
def release_armbands(working_thread):
    lock.acquire(True)
    print("release armbands %s" %
          thread_armbands_pair_book[working_thread].__str__())
    for each in thread_armbands_pair_book[working_thread]:
        each.is_occupy = False
    thread_armbands_pair_book[working_thread] = []
    lock.release()

"""
获取当前线程占用的手环对象
"""
def get_occupied_armbands(working_thread):
    return thread_armbands_pair_book[working_thread]

# 这个dict用于具有具有可读性的手环id分配情况
# 由各个手环的hash value进行映射
# 在实例持续运行的过程中,待机重连后,每个手环的hash value保存不变
# 可以通过这样一个数据结构，用这个hash value为每个手环分配一个可读的id
# myo_obj.__hash__() -> armband_id:str
armband_id_assign_book = {}
armband_cnt = 0

def assign_armband_id(myo_obj):
    global armband_cnt
    try:
        res = armband_id_assign_book[myo_obj.__hash__()]
    except KeyError:
        res = armband_id_assign_book[myo_obj.__hash__()] = "armband_%d" % armband_cnt
        armband_cnt += 1
    return res

def vibrate_armband(armband_id):
    armband_id_obj_map[armband_id].vibrate(VibrationType.medium)

myo.init(os.path.dirname(__file__))
myo_hub = myo.Hub()
myo_feed = MyoFeedback()
myo_hub.run(1000, myo_feed)
update_connected_list()
