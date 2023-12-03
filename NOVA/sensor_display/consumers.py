from channels.generic.websocket import AsyncWebsocketConsumer
import json
from .models import PressureSensor

class PressureSensorConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.group_name = 'pressure_updates'

        # Join room group
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )

    async def receive_json(self, content):
        # Send message to WebSocket
        await self.send(text_data=json.dumps(content))

    async def send_pressure_update(self):
        for sensor in PressureSensor.objects.all():
            await self.send(text_data=json.dumps({
                'sensor_id': sensor.id,
                'pressure': sensor.pressure
            }))