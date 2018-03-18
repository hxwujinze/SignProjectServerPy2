# coding:utf-8
from django.conf.urls import url

from utilities_access import views

urlpatterns = [
    url(r'get_armbands_list/$', views.get_armbands_list),
    url(r'request_socket_connection/$', views.request_socket_connection),
    url(r'ping_armband/$', views.ping_armband),
    url(r'capture_feedback/$', views.capture_feedback),
    url(r'generate_feedback_data/$', views.generate_feedback_data),
]
