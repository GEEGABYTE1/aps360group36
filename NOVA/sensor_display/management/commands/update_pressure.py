from django.core.management.base import BaseCommand
from sensor_display.models import PressureSensor
import time
import random
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer


class Command(BaseCommand):
    help = 'Updates pressure sensor values'

    def handle(self, *args, **kwargs):
        channel_layer = get_channel_layer()
        print (channel_layer)
        while True:
            for sensor in PressureSensor.objects.all():
                new_pressure = random.gauss(mu=1.0, sigma=0.1) 
                sensor.pressure = new_pressure
                sensor.save()
                print (sensor.pressure)
                async_to_sync(channel_layer.group_send)(
                    "pressure_updates",
                    {
                        "type": "receive_json",
                        "sensor_id": sensor.id,
                        "pressure": new_pressure
                    }
                )

            time.sleep(0.5)  