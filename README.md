# Nova Mission Operations Back-End

The back-end for NOVA mission control/operations. This project uses the Django framework.

## Installation

1. **Clone the Repository**

Add an ssh key to Github ([How to add SSH Key to Github ssh-agent](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)).
```
git clone git@github.com:UTATRocketry/nova-ops-back.git
cd nova-ops-back
```

2. **Install dependencies**
First, it is recommended to set up a virtual environment, good options are conda ([Getting started with Conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)) or venv.

```
pip install -r requirements.txt
```

Additionally, you should make sure that Redis is installed and running on your computer, as this is crucial for handling Websocket connections. Installation guide is largely simple: [Redis: Installation](https://redis.io/docs/install/install-redis/)

3. **Run Migrations**

This will initialize a database, and commit model schemas, etc.
```
python manage.py migrate
```

4. **Recommended: Seed the database**

Running this command will add a superuser (username: root, password: 12345) and an initial PressureSensor instance to the database.
```
python manage.py seed_test
```

## Usage

Start the Django development server by running: 

```
python manage.py runserver
```

Separately, run the background task for updating the example pressure sensor data:

```
python manage.py update_pressure
```

You should have an active WebSocket connection at `ws://localhost:8000/ws/pressure/` that serves the pressure and id of the PressureSensor instance.
