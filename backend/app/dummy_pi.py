import random
from datetime import datetime
from data_file import data_store

async def get_timestamp():
    # Getting the current date and time
    dt = datetime.now()
    str_date_time = dt.strftime("%H:%M:%S,%d-%m-%Y")
    data_store["timestamp"]=str_date_time
async def generate_sensor_data ():
    # Simulate random sensor data update every second
    for sensor in data_store["sensors"]:
        sensor["value"] = round(random.uniform(0.0, 100.0), 2)


async def handle_comand (command):
    # Handle actuator toggling
    if command.get("action") == "toggle":
        for actuator in data_store["actuators"]:
            if actuator["name"] == command["name"]:
                actuator["status"] = "on" if actuator["status"] == "off" else "off"
                break