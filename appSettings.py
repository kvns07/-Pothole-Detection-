classNames = ['Pothole']
myColor = (0, 0, 255)
my_map = {3: 1}
useCuda = True
useConf = 0.4
MONGODB_URI = "mongodb://172.22.129.175:12345/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.0.2"
MODEL_NAME = "potholev1.pt"
FLASK_SERVER = "localhost:5000"

import time
import datetime

def getTimestring():
    return datetime.datetime.now().strftime("%A, %d %B %Y %I:%M:%S%p")

def getTimestamp():
    return str(time.time()).replace('.', '', 1)