# coding:utf-8
import json

from django.http import HttpResponse

from . import armbands_manager
from . import models
from . import worker_manager

# Create your views here.
"""
整个服务端的各种应用入口就在这个文件里，服务器框架响应来着客户端的请求，
通过解析url，执行指定的方法，并将post请求取出作为参数传给相应方法
"""

"""
获取手环的列表 在准备进行手环匹配时，通过该方法向客户端返回
已经连接到服务端的各个手环的id，以及手环是否被占用
"""

def get_armbands_list(request):
    armbands_list = armbands_manager.get_armbands_list()
    armbands_jsonlist = []
    for each in armbands_list:
        armbands_jsonlist.append({
            "armband_id": each.armband_id,
            "armband_status": -2 if each.is_occupied() else -1
        })
    return HttpResponse(json.dumps(armbands_jsonlist, indent=3))

"""
这个方法用于启动服务端与客户端之间的socket连接 以及占用被客户端选定进行手语识别的手环
该方法将启动一组工作线程（用于socket数据接收监听，客户端事务请求响应，手语识别等工作），
创建成功后返回创建的socket连接的监听端口供客户端进行连接
返回端口号后 Django响应工作就结束了 接下来客户端与服务端的通讯都由建立起的socket连接传递，
并由这个工作线程进行响应
"""

def request_socket_connection(request):
    print(request.POST)
    armband_id = request.POST['?armband_id']
    print("connecting armbands: %s" % str(armband_id))
    port = worker_manager.set_up_connection(json.loads(armband_id)['armbands_list'])
    return HttpResponse(str(port))

"""
该方法用于响应客户端对于手语识别结果正误反馈的响应，它的工作是解析post中的内容
并通过Django models修改数据库中对应手语的数据
"""

def capture_feedback(request):
    capture_id = request.POST['?capture_id']
    correctness = request.POST['correctness']
    correctness = True if correctness == 'True' else False
    models.store_capture_feedback(capture_id, correctness)
    return HttpResponse("success")

"""
该方法用于ping某一个已连接的手环 使其震动一下
"""

def ping_armband(request):
    armband_id = request.POST['?armband_id']
    print("ping armband: %s" % armband_id)
    armbands_manager.vibrate_armband(armband_id)
    return HttpResponse('success')

def generate_feedback_data(requset):
    models.pickle_cap_data2file()
    return HttpResponse('success')
