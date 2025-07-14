## Install carla-simulator
### 1. Install Required System Dependency
Before downloading CARLA, install the necessary system dependency:
```
sudo apt-get -y install libomp5
```
### 2. Download the CARLA 0.9.15 Release
Download the CARLA_0.9.15.tar.gz file (approximately 16GB) from the official release:
```
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
```
### 3. Unpack CARLA to the Desired Directory
Unpack the downloaded file to /opt/carla-simulator/:
```
sudo mkdir /opt/carla-simulator
sudo tar -xzvf CARLA_0.9.15.tar.gz -C /opt/carla-simulator/
```
### 4. Install the CARLA Python Module
Finally, install the CARLA Python module and necessary dependencies:
```
python -m pip install carla==0.9.15
```
#### Ref: https://github.com/carla-simulator/carla/issues/7017#issuecomment-1908462106