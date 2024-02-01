# Import Settings
from appSettings import *

def handle_error(e):
    print(e)
    return "Error!"

# GeoJSON and GeoPY Imports
from geojson import Point
from geopy.distance import geodesic as GD

# MongoDB Imports
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson import json_util
import json

# MongoDB Settings - URL
uri = MONGODB_URI
client = MongoClient(uri, server_api=ServerApi('1'))

db = 0

# Check Ping
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
    
    # Choose the database
    db = client.codeutsava

except Exception as e:
    handle_error(e)


# Model Imports
import pandas as pd
import cv2
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import argparse
import numpy as np
# For making requests to backend server
import asyncio
import requests

# Loading Model Name from .env & Initializing Model
print('Using Model:', MODEL_NAME)

model = YOLO(MODEL_NAME)
if useCuda:
    model.to('cuda')


# Flask Imports

import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, session
from flask_cors import CORS
from flask_session import Session
from werkzeug.utils import secure_filename


# Image Upload Config
UPLOAD_FOLDER = 'static_detect'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Starting flask

app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

sess = Session()
sess.init_app(app)

app.debug = True

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Flask Routes

@app.route("/")
def index():
    return "Server is live!"


# Get and Post for pothole data
def parse_json(data):
    return json.loads(json_util.dumps(data))

@app.route("/potholes", methods=["GET"])
def get_pothole_data():
    try:
        client.admin.command('ping')
        readCursor = db.potholes.find({})
        return parse_json(readCursor)
    except Exception as e:
        handle_error(e)

@app.route("/potholes", methods=["POST"])
def insert_pothole_data():
    try:
        # Lat Lon Mandatory
        lat = request.form['lat']
        lon = request.form['lon']

        # Use Image Name if present
        imgName = getTimestamp() + '.jpg'
        if imgName in request.form:
            imgName = request.form['imgName']
        # Use Area if present
        area = -1
    # if area in request.form:
    #     area = request.form['area']
        
        writeCursor = db.potholes.insert_one({
            "lat": lat,
            "lon": lon,
            "imgName": imgName,
            "area": area,
            "repaired": False,
            "insertTS": getTimestamp(),
            "repairTS": -1
        })

        print("DB Insert:", str(writeCursor.inserted_id))
        return "Success"
    except Exception as e:
        handle_error(e)

# This route will save pothole images coming from live stream
@app.route("/potholes_save", methods=["POST"])
def save_pothole_from_live():
    # Extract longitude latitute and other data
    try:
        # Lat Lon Mandatory
        lat = request.form['lat']
        lon = request.form['lon']
        print("Latitude:", lat)
        print("Longitude:", lon)

        # Use Image Name if present
        imgName = getTimestamp()
        if imgName in request.form:
            imgName = request.form['imgName']
        # Use Area if present
        area = -1
        if area in request.form:
            area = request.form['area']
        
        writeCursor = db.potholes.insert_one({
            "lat": lat,
            "lon": lon,
            "imgName": imgName + '.jpg',
            "area": area,
            "repaired": False,
            "insertTS": getTimestamp(),
            "repairTS": -1
        })
        print("Live Stream DB Insert:", str(writeCursor.inserted_id))
    except Exception as e:
        handle_error(e)

    # Check for the file, and upload. 
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = imgName
        uploads = app.root_path + '/' + app.config['UPLOAD_FOLDER']
        try:
            file.save(uploads + '/live' + filename + '.jpg')
            print("Live Stream Upload:", '/live' + filename + '.jpg')
            return "Success!"
        except Exception as e:
            return handle_error(e)

# Handle File Upload and Download from Android App with Inference

@app.route('/uploads/<path:filename>')
def download_file(filename):
    uploads = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
    # print(os.path.join(uploads, filename))
    return send_from_directory(uploads, filename)

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = getTimestamp()
            # uploads = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
            # This contains the data sent from frontend
            try:
                lat = request.form['lat']
                lon = request.form['lon']
                print("Latitude:", lat)
                print("Longitude:", lon)
            except Exception as e:
                return handle_error(e)
            
            # Run Inference and save output
            try:
                # file.save(os.path.join(uploads, filename))
                filestr = request.files['file'].read()
                # file_bytes = np.fromstring(filestr, np.uint8)
                nparr = np.fromstring(filestr, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
                results = model(img, stream=True)
                boxCount = 0
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Bounding Box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                        w, h = x2 - x1, y2 - y1
                        # cvzone.cornerRect(img, (x1, y1, w, h))

                        # Confidence
                        conf = math.ceil((box.conf[0] * 100)) / 100

                        # Class Name
                        cls = int(box.cls[0])
                        currentClass = classNames[cls]
                        # print(currentClass)

                        if conf > useConf:
                            boxCount += 1
                            # cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                            #   (max(0, x1), max(35, y1-10)), scale=2, thickness=2,colorB=myColor,
                                            #   colorT=(255,255,255),colorR=myColor, offset=5)
                            cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 2)
                            # currentArray = np.array([x1, y1, x2, y2, conf])
                            # detections = np.vstack((detections, currentArray))
                cv2.imwrite(app.root_path + "/static_detect/" + filename + '.jpg', img)
                print('Write Success:', app.root_path + "/static_detect/" + filename + '.jpg')


            except Exception as e:
                return handle_error(e)
            
            if boxCount > 0:
                # Call the backend endpoint to save data into backend
                writeData(lat, lon, filename + '.jpg', -1)
                return {"pothole":"true"}
            else:
                return {"pothole":"false"}
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

# Pothole elimination from database using Haversine formula
@app.route("/eliminate", methods=["POST"])
def eliminate_potholes():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        uploads = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
        # This contains the data sent from frontend
        try:
            lat = request.form['lat']
            lon = request.form['lon']
            print("Latitude:", lat)
            print("Longitude:", lon)
        except Exception as e:
            return handle_error(e)
        
        # Run Inference and save output
        try:
            # file.save(os.path.join(uploads, filename))
            filestr = request.files['file'].read()
            # file_bytes = np.fromstring(filestr, np.uint8)
            nparr = np.fromstring(filestr, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
            results = model(img, stream=True)
            boxCount = 0
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                    w, h = x2 - x1, y2 - y1
                    # cvzone.cornerRect(img, (x1, y1, w, h))

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Class Name
                    cls = int(box.cls[0])
                    currentClass = classNames[cls]
                    # print(currentClass)

                    if conf > useConf:
                        boxCount += 1
                        # cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                        #   (max(0, x1), max(35, y1-10)), scale=2, thickness=2,colorB=myColor,
                                        #   colorT=(255,255,255),colorR=myColor, offset=5)
                        cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 2)
                        # currentArray = np.array([x1, y1, x2, y2, conf])
                        # detections = np.vstack((detections, currentArray))
            if boxCount == 0:
                # Fetching latitudes and longitudes from DB, 
                # Removing using Haversine Formula
                my = (lat, lon)
                client.admin.command('ping')
                readCursor = db.potholes.find({})
                objListRem = []
                for doc in readCursor:
                    curr = (doc['lat'], doc['lon'])
                    if GD(my,curr).km < 0.01:
                        objListRem.append(doc['_id'])
                print("Potholes to set to repaired: ", objListRem)
                db.potholes.update_many(
                    {
                        "_id": {"$in": objListRem}
                    },
                    {
                        "$set": {
                            "repaired": True
                        }
                    }
                )
                return {"status":"true"}
            else:
                currTS = getTimestamp()
                print("Pothole Detected")
                cv2.imwrite(app.root_path + "/static_detect/wrong" + currTS + ".jpg", img)
                print("File Saved:", app.root_path + "/static_detect/wrong" + currTS + ".jpg")
                return {"status":"false"}
        except Exception as e:
            return handle_error(e)

# Function call to make network requests
def writeData(lat, lon, imgName, area):
    req = requests.post('http://localhost:5000/potholes', data={
        "lat": lat,
        "lon": lon,
        "imgName": imgName,
        "area": area
    })
    print("req", req)


# Image Filters
def denoise_image(image):
    # Apply Non-Local Means Denoising
    denoised_image = cv2.fastNlMeansDenoising(image, None, h=10, searchWindowSize=21, templateWindowSize=7)
    return denoised_image

def convert_to_grayscale(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image
