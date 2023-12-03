from django.db import models

# Create your models here.
class GenericSensor(models.Model):
    display_name = models.CharField(max_length=100)
    description = models.TextField()


class PressureSensor(GenericSensor):
    pressure = models.FloatField()

    def __str__(self):
        return f"Sensor {self.id} - Pressure: {self.pressure}"