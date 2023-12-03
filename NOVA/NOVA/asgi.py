import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import sensor_display.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'NOVA.settings')

application = ProtocolTypeRouter({
  "http": get_asgi_application(),
  "websocket": AuthMiddlewareStack(
        URLRouter(
            sensor_display.routing.websocket_urlpatterns
        )
    ),
})