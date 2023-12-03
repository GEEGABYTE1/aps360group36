# Register your models here.
from django.contrib import admin
from .models import PressureSensor

@admin.register(PressureSensor)
class PressureSensorAdmin(admin.ModelAdmin):
    list_display = ('id', 'pressure')
    search_fields = ('id',)