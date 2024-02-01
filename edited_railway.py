
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
import winsound
from sort import *
import asyncio
import websockets
import json
import base64

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import time

# Initialize Firebase Admin SDK
cred = credentials.Certificate('alert-interface-firebase-adminsdk-rnedn-3c2dfa4f32.json')
firebase_admin.initialize_app(cred, {'projectId': 'alert-interface'})  # Replace with your Firebase project ID

# Initialize Firestore
db = firestore.client()
alerts_ref = db.collection('alerts1')





def denoise_image(image):
    # Apply Non-Local Means Denoising
    denoised_image = cv2.fastNlMeansDenoising(image, None, h=10, searchWindowSize=21, templateWindowSize=7)
    return denoised_image


def convert_to_grayscale(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


# For Video
cap = cv2.VideoCapture("../Videos/final.mp4")


#loading the trained model
model = YOLO("best(1).pt")

#printing details of the model
#print(model.summary())

#initializing the SORT tracker
tracker = Sort(max_age=20, min_hits=0, iou_threshold=0.3)


#type of defects
classNames = [
    'Dent',
    'Dirt',
    'Crush',
    'Scratch',
    'Slant',
    'Damage',
    'Unknown',
    'Gap',
    'Fastener Defect'
]


myColor = (0, 0, 255)


my_map = {
            3: 1,
        }

while True:
    success, img = cap.read()
    detections = np.empty((0, 5))
    if success:
        # Convert the image to grayscale
        gray_img = convert_to_grayscale(img)

        # Denoise the grayscale image
        denoised_img = denoise_image(gray_img)

        results = model(img, stream=True)


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
                print(currentClass)

                if conf>0.5:

                    #cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                    #                   (max(0, x1), max(35, y1-10)), scale=2, thickness=2,colorB=myColor,
                    #                   colorT=(255,255,255),colorR=myColor, offset=5)
                    #cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        #updating the tracker list
        resultsTracker = tracker.update(detections)

        for result in resultsTracker:
            loop_count = 1
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(result)
            w, h = x2 - x1, y2 - y1
            #cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            #cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
            #                   scale=2, thickness=3, offset=10)
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                              (max(0, x1), max(35, y1-10)), scale=2, thickness=2,colorB=myColor,
                              colorT=(255,255,255),colorR=myColor, offset=5)
            cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
            if(my_map.get(id) is None):
                alert = {
                    'Message': classNames[cls],
                    'Location': 'XXYYZZ',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'Confidence': conf
                }

                # Add the alert to Firestore
                alerts_ref.add(alert)

                print('Alert sent:', alert)
                my_map[id] = 1

                # Call the send_alert function with the alert_data when a defect is detected
                frequency = 2500  # Hz
                duration = 100  # milliseconds
                while loop_count != 0:
                    #play(alert_sound)
                    loop_count -= 1
                    winsound.Beep(frequency, duration)


    #display the annotated image
    cv2.imshow("Image", img)
    cv2.waitKey(1)



