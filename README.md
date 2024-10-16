
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
Description of Key Files
```


## Requirements

To run this project, you will need Docker and Docker Compose installed on your machine. Installation guides for Docker can be found [here](https://docs.docker.com/get-docker/) and for Docker Compose [here](https://docs.docker.com/compose/install/).

## Running the Application

1. **Clone the Repository**
   ```bash
   git clone https://github.com/UTATRocketry/novaOps-back.git
   cd novaOps-back
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
   docker-compose down
   ```

## Important Notes

1. The `html_generator.py` file is currently used as a small simple ui for backend testing but it will be replaced soon

2. The server is not currently connected to any ground hardware so the `dummy_pi.py` file randomly generates new data and changes the actuator status in the data file. This will eventually be replaced by the MQTT hardware interface. 