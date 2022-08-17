# first_response comp1

## To run the code ensure that you have a raspberry pi 4 with a camera

1. Open the terminal
2. Ensure that your system is up-to-date
  - sudo apt-get update && sudo apt-get upgrade
3. Once that is done Run sudo raspi-config
  - Navigate to Interface Options
  - Select Legacy camera to enable it
4. Reboot your Raspberry Pi
  - sudo reboot

## Installing OpenCV

### Get Pip
 Python 2
  - sudo apt-get install python-pip
 Python 3
  - sudo apt-get install python3-pip

### Option 1: Install OpenCV for the Whole System:
   - sudo pip install opencv-contrib-python

### Option 2: Install OpenCV in a Python Virtual Environment:
   - sudo pip install virtualenv virtualenvwrapper
   - vim ~/.bashrc
   - add the following:
      - export WORKON_HOME=$HOME/.virtualenvs
      - export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
      - source /usr/local/bin/virtualenvwrapper.sh
   - save the file and then run:
      - source ~/.bashrc
   - make your virtualenv:
      - mkvirtualenv [name_of_venv] -p python3
   - you know you are in the venv if it looks like this:
      - ([name_of_venv])pi@raspberrypi:
   - install OpenCV
      - pip install opencv-contrib-python

### Test OpenCV installation
  - python3
    -  >> import cv2
    - >> cv2.__version__
    - will output the version ('4.5.3') or your installed version
    
### Run code

Once everything is confirmed go into the counter directory and run `python3 people_counter.py`


