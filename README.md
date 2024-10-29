
# NovaOps Backend
## Overview
TBD

## Directory Structure
```bash
/novaOps
│
├── backend/
│   └── app/
│       ├── __init__.py            # Marks this directory as a Python package
│       ├── main.py                # Main application entry point for FastAPI
│       ├── auth.py                # Handles authentication using OAuth2 and JWT tokens
│       ├── data_file.py           # Stores mock sensor and actuator data used in the system
│       ├── html_generator.py      # Temporary generator for the HTML for the sensor and actuator dashboard
├── docker-compose.yml              # Defines services and configurations for running Docker containers
├── Dockerfile                      # Docker instructions for building the FastAPI app image
└── README.md                       # This README file
```


## Requirements

To run this project, you will need Docker and Docker Compose installed on your machine. Installation guides for Docker can be found [here](https://docs.docker.com/get-docker/) and for Docker Compose [here](https://docs.docker.com/compose/install/).

## Running the Application

1. **Clone the Repository**
   ```bash
   git clone https://github.com/UTATRocketry/novaOps-back.git
   cd /path/to/novaOps-back
   ```

2. **Build and Run the Docker Containers**
   ```bash
   docker-compose up --build
   ```

3. **Viewing the Application**
   Once the container is running, you can access the application in your browser at: `http://localhost:8000`

4. **Stopping the Application**
   To stop the application, run:
    ```bash
   sudo docker-compose down
   ```

## To setup a Raspberry Pi with the Config Scripts
1. **Clone the Repository**
   ```bash
   git clone https://github.com/UTATRocketry/novaOps-back.git
   cd novaOps-back
   ```
2. **Update Log File Path**
Update line 4 `exec > /path/to/nova/session.log 2>&1` of the `initial_config.sh` file, so that the file path is accurate

3. **Make the scripts executable**
```bash
chmod +x /path/to/nova/initial_config.sh
chmod +x /path/to/nova/post_reboot_config.sh
```
4. **Run the scripts**
Run `initial_config.sh` wait for it to reboot the Pi and then run `post_reboot_config.sh`. This will start the server on `192.68`. Use `sudo docker-compose down` to stop and `sudo docker-compose up` to run again.

## Important Notes

1. The `html_generator.py` file is currently used as a small simple ui for backend testing but it will be replaced soon

2. The server is not currently connected to any ground hardware so the `dummy_pi.py` file randomly generates new data and changes the actuator status in the data file. This will eventually be replaced by the MQTT hardware interface. 