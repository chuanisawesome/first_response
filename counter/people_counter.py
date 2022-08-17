from picamera.array import PiRGBArray  # Generates a 3D RGB array
from picamera import PiCamera  # Provides a Python interface for the RPi Camera Module
import time  # Provides time-related functions
import numpy as np
import cv2
import os
import imutils

import datetime
import argparse
from threading import Thread

NMS_THRESHOLD = 0.3
MIN_CONFIDENCE = 0.2

def human_detection(image, model, layer_name, personidz=0):
    (H, W) = image.shape[:2]
    results = []

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    layerOutputs = model.forward(layer_name)

    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personidz and confidence > MIN_CONFIDENCE:

                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
    # ensure at least one detection exists
    if len(idzs) > 0:
        # loop over the indexes we are keeping
        for i in idzs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # update our results list to consist of the person
            # prediction probability, bounding box coordinates,
            # and the centroid
            res = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(res)
    # return the list of results
    return results

# set up camera
resX = 320
resY = 240
# Initialize the camera
camera = PiCamera()
# Generates a 3D RGB array and stores it in rawCapture
raw_capture = PiRGBArray(camera, size=(resX, resY))
# Set the camera resolution
camera.resolution = (resX, resY)
# Set the number of frames per second
camera.framerate = 10
# Wait a certain number of seconds to allow the camera time to warmup
time.sleep(0.1)
print("test")

# initialize textIn and textOut values
textIn = 0
textOut = 0

def testIntersectionIn(x, y):
    res = -450 * x + 400 * y + 157500
    if((res >= -550) and  (res < 550)):
        print (str(res))
        return True
    return False

def testIntersectionOut(x, y):
    res = -450 * x + 400 * y + 180000
    if ((res >= -550) and (res <= 550)):
        print (str(res))
        return True
    return False

# load the model
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"

model = cv2.dnn.readNetFromDarknet(config_path, weights_path)

"""s
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
"""

layer_name = model.getLayerNames()

layer_name = [
    layer_name[i[0] - 1] for i in model.getUnconnectedOutLayers()
]   # [layer_name[len(layer_name) - 1]]
cap = cv2.VideoCapture(0)
writer = None

# capture frames from the camera
for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):

    image = frame.array

    # image = imutils.resize(image, width=640)
    
    results = human_detection(
        image, model, layer_name, personidz=LABELS.index("person")
    )

    print("human_detection reached")

    for res in results:
        # (x, y, w, h) = cv2.boundingRect(res)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(
            image, (res[1][0], res[1][1]), (res[1][2], res[1][3]), (0, 255, 0), 2
        )
        b=len(results)
        cv2.putText(image,"peoplecount: {}".format(str(b)),(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),2)
        
        print("person detected")
        
        # if(testIntersectionIn((x + x + w) / 2, (y + y + h) / 2)):
        #     textIn += 1

        # if(testIntersectionOut((x + x + w) / 2, (y + y + h) / 2)):
        #     textOut += 1

    cv2.line(image, (320 // 2, 0), (320, 400), (250, 0, 1), 2) #blue line
    cv2.line(image, (320 // 2 - 30, 0), (320 - 30, 400), (0, 0, 255), 2)#red line
    
    cv2.putText(image, "In: {}".format(str(textIn)), (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(image, "Out: {}".format(str(textOut)), (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(image, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
            (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.imwrite("FLY.png", image)
        # cv2.imshow("FLY", image)
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    raw_capture.truncate(0)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    # key = cv2.waitKey(1)
    # if cv2.waitKey(25) & 0xFF == ord("q"):
    #     break
    # raw_capture.truncate(0)

# When everything done, release the capture
cap.release()
# finally, close the window
cv2.destroyAllWindows()