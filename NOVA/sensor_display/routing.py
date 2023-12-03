from django.urls import re_path
from .consumers import PressureSensorConsumer

websocket_urlpatterns = [
    re_path(r'ws/pressure/', PressureSensorConsumer.as_asgi()),
]