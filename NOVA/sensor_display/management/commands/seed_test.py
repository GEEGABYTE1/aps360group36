from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from sensor_display.models import PressureSensor
from django.core.exceptions import ObjectDoesNotExist

class Command(BaseCommand):
    help = 'Seeds the database with a superuser and a pressure sensor'

    def handle(self, *args, **kwargs):
        # Create superuser
        username = 'root'
        password = '12345'
        email = 'example@example.com'
        try:
            if not User.objects.filter(username=username).exists():
                User.objects.create_superuser(username, email, password)
                self.stdout.write(self.style.SUCCESS('Successfully created superuser'))
            else:
                self.stdout.write(self.style.WARNING('Superuser already exists'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error creating superuser: {e}'))

        try:
            if not PressureSensor.objects.exists():
                PressureSensor.objects.create(pressure=1.0) 
                self.stdout.write(self.style.SUCCESS('Successfully created a PressureSensor instance'))
            else:
                self.stdout.write(self.style.WARNING('PressureSensor instance already exists'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error creating PressureSensor instance: {e}'))