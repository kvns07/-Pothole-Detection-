# Import Settings
import requests
from appSettings import *

# Global latitude and longitude
globalLat = -1
globalLon = -1

# UDP running on a thread
import threading
import socket

# Creating the socket

data = 'Nothing received'
UDP_ADDR = '0.0.0.0'
UDP_PORT = 31337
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_ADDR, UDP_PORT))

def rec_UDP():
    while True:
        data, addr = sock.recvfrom(4096)
        globalLat, globalLon = data.decode('utf-8').split(',')
        print("Echoing " + data.decode('utf-8') + " back to " + str(addr))

# The thread that ables the listen for UDP packets is loaded
listen_UDP = threading.Thread(target=rec_UDP)
listen_UDP.start()

# Model and OpenCV starts

import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import argparse
import numpy as np
import time

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    parser.add_argument(
        "--model",
        default=["potholev1.pt"],
        nargs=1
    )
    parser.add_argument(
        "--src",
        default=[0],
        nargs=1
    )
    args = parser.parse_args()
    return args

# Image Filters
def denoise_image(image):
    # Apply Non-Local Means Denoising
    denoised_image = cv2.fastNlMeansDenoising(image, None, h=10, searchWindowSize=21, templateWindowSize=7)
    return denoised_image


def convert_to_grayscale(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    rtmpSource = args.src[0]
    modelName = args.model[0]

    print("RTMP Arg", rtmpSource)

    cap = cv2.VideoCapture(rtmpSource)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    model = YOLO(modelName)
    if useCuda:
        model.to('cuda')

    tracker = Sort(max_age=20, min_hits=0, iou_threshold=0.3)

    # classNames = [
    #     'Pothole'
    # ]

    # myColor = (255, 0, 0)

    # my_map = {3: 1,}

    lastBoxCount = 0
    while True:
        success, img = cap.read()
        if success:
            detections = np.empty((0, 5))

            results = model(img, stream=True)
            # result = model(frame, agnostic_nms=True)[0]
            # print(result)

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
                    print(currentClass)

                    if conf>0.25:
                        #cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                        #                   (max(0, x1), max(35, y1-10)), scale=2, thickness=2,colorB=myColor,
                        #                   colorT=(255,255,255),colorR=myColor, offset=5)
                        #cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
                        currentArray = np.array([x1, y1, x2, y2, conf])
                        detections = np.vstack((detections, currentArray))
                
            #updating the tracker list
            resultsTracker = tracker.update(detections)

            for result in resultsTracker:
                boxCount += 1
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
                    # alert = {
                    #     'Message': classNames[cls],
                    #     'Location': 'XXYYZZ',
                    #     'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    #     'Confidence': conf
                    # }

                    # # Add the alert to Firestore
                    # alerts_ref.add(alert)

                    # print('Alert sent:', alert)
                    # my_map[id] = 1

                    # # Call the send_alert function with the alert_data when a defect is detected
                    # frequency = 2500  # Hz
                    # duration = 100  # milliseconds
                    # while loop_count != 0:
                    #     #play(alert_sound)
                    #     loop_count -= 1
                    #     winsound.Beep(frequency, duration)

                    # This block will call the backend api
                    pass
            if boxCount > lastBoxCount:
                # send_data_backend(img)
                sendDataThread = threading.Thread(target=send_data_backend, args=[img])
                sendDataThread.start()
                sendDataThread.join()

            cv2.imshow("Real Time Pothole Detection on Stream", img)
            lastBoxCount = boxCount

            if (cv2.waitKey(30) == 27):
                break


def send_data_backend(frame):
    try:
        print('Pothole Detected, Backend API Call')
        imencoded = cv2.imencode(".jpg", frame)[1]
        file = {'file': ('image.jpg', imencoded.tostring(), 'image/jpeg', {'Expires': '0'})}
        data = {"lat": globalLat, "lon": globalLon}
        response = requests.post("http://" + FLASK_SERVER + "/potholes_save", files=file, data=data, timeout=10)
        if response.status_code == 200:
            print('Data Sent Successfully!')
        else:
            print('Could not send data to server')
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()