#!/bin/bash

# Save a log of all commands and their output
exec > /path/to/directory/session.log 2>&1

# Update and upgrade the system
echo "Updating system..."
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install ufw
sudo apt install -y dnsmasq docker-compose

#firewall set up
sudo ufw allow 8000/tcp
sudo ufw allow 8000/udp
sudo ufw enable
# sudo ufw status

# Set a static IP address for Ethernet (eth0)
echo "Configuring static IP address for for the Raspberry Pi Ethernet..."
sudo bash -c 'cat << EOF >> /etc/dhcpcd.conf
interface eth0
static ip_address=192.168.0.1
static routers=192.168.0.1
static domain_name_servers=192.168.0.1 8.8.8.8
EOF'

# Configure dnsmasq for DHCP
echo "Setting up dnsmasq for DHCP..."
sudo bash -c 'cat << EOF > /etc/dnsmasq.conf
interface=eth0
listen-address=192.168.0.1
bind-dynamic
domain-needed
bogus-priv
dhcp-range=192.168.0.2,192.168.0.10,255.255.255.0,12h
EOF'

# Restart the necessary services
echo "Restarting services to apply changes..."
sudo systemctl restart dhcpcd        # Restart DHCP client daemon
sudo service dnsmasq restart     # Restart dnsmasq to apply changes

# Check if Docker is installed
if command -v docker &> /dev/null
then
    echo "Docker is already installed."
else
    echo "Docker is not installed. Installing Docker..."
    # Install Docker
    curl -sSL https://get.docker.com | sh
    # Add the current user to the Docker group
    sudo usermod -aG docker {$USER}
    echo "Docker installation complete. Please log out and log back in to apply group changes."
fi

#reboot
echo "Rebooting Raspberry Pi..."
sudo reboot