#!/bin/bash

# Ask if the user wants to enable the container to run at boot
read -p "Would you like Docker to run at boot? (y/n): " run_at_boot

if [ "$run_at_boot" == "y" ]; then
    echo "Setting up Docker to run at boot..."
    sudo systemctl enable docker
    sudo systemctl start docker
    echo "The container will now start automatically at boot."
else
    echo "Skipping auto-start configuration."
fi

# Build and run Docker container using docker-compose
echo "Starting FastAPI container..."
sudo docker-compose up --build

