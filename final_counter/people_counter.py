from turtle import right
from objecttracker.centroidtracker import CentroidTracker
from objecttracker.trackableobject import TrackableObject
from picamera.array import PiRGBArray  # Generates a 3D RGB array
from picamera import PiCamera  # Provides a Python interface for the RPi Camera Module
import io
import time  # Provides time-related functions
import numpy as np
import cv2
import os
import numpy as np
import imutils
import dlib
import cv2
import datetime
from datetime import timedelta

from typing import Tuple

# Variables that can be set according to programmer need
# Some thresholding values to be set here as well
interval = timedelta(seconds=60)
startTime = datetime.datetime.now()
endTime = startTime + interval

countOfEntered = 0
countOfExited = 0
skip_frames = 10
confidence_value = 0.4
video_file_path = "videos/5.mov"
# output_file_path = "output/newFour.avi"

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalPersonsEntered = 0
totalPersonsExited = 0

personidz=0
NMS_THRESHOLD = 0.3
MIN_CONFIDENCE = 0.2

font = cv2.FONT_HERSHEY_PLAIN

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a Trackable Object
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# set up camera
resX = 640
resY = 480
# Initialize the camera
camera = PiCamera()
# Set the camera resolution
camera.resolution = (resX, resY)
# Set the number of frames per second
camera.framerate = 32
# Generates a 3D RGB array and stores it in rawCapture
raw_capture = PiRGBArray(camera, size=(resX, resY))
# Wait a certain number of seconds to allow the camera time to warmup
time.sleep(0.1)
print("[INFO] warming up camera")

# load the model person detection
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

# load the model gun detection
labelsPath_g = "gun_dtect_model/obj.names"
LABELS_G = open(labelsPath_g).read().strip().split("\n")

weights_path_g = "gun_dtect_model/yolov4.weights"
config_path_g = "gun_dtect_model/yolov4-g.cfg"

model_g = cv2.dnn.readNetFromDarknet(config_path_g, weights_path_g)

layer_name_g = model_g.getLayerNames()
layer_name_g = [
    layer_name_g[i[0] - 1] for i in model_g.getUnconnectedOutLayers()
]

def gun_detection(image, model_g, layer_name_g, personidz=0):
    (H, W) = image.shape[:2]
    results = []

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model_g.setInput(blob)
    layerOutputs_g = model_g.forward(layer_name_g)

    boxes = []
    class_ids = []
    confidences = []

    for output in layerOutputs_g:
        for detection in output:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.98:
                # Calculating coordinates
                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                w = int(detection[2] * W)
                h = int(detection[3] * H)

				# Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
    #Draw boxes around detected objects
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(LABELS_G[class_ids[i]])
            confidence = confidences[i]
            color = (256, 0, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label + " {0:.1%}".format(confidence), (x, y - 20), font, 3, color, 3)

            # update our results list to consist of the person
            # prediction probability, bounding box coordinates,
            # and the centroid
            res = (confidence, (x, y, x + w, y + h), label)
            results.append(res)
    # return the list of results
    return results

# capture frames from the camera
for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):

    image = frame.array
    (H, W) = image.shape[:2]

    results = gun_detection(
        image, model_g, layer_name_g, personidz=LABELS_G.index("Handgun")
    )
    print("gun detection reached")

    # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []
    
    # running the detection model
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    layerOutputs = model.forward(layer_name)

    print("status" + status)

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % skip_frames == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []
        print("status" + status)
    
    boxes = []
    cent = []
    confidences = []

    for output in layerOutputs:
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personidz and confidence > MIN_CONFIDENCE:
                idx = int(scores[classID])
                
                if LABELS[idx] != "person":
                    continue

                print(confidence)
                print(LABELS[idx])

                box = detection[0:4] * np.array([W, H, W, H])
                (startX, startY, endY, endX) = box.astype("int")
                

                x = int(startX - (endY / 2))
                y = int(startY - (endX / 2))

                boxes.append([x, y, int(endY), int(endX)])
                cent.append((startX, startY))
                confidences.append(float(confidence))

                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endY, endX)
                # print(rect)
                tracker.start_track(image, rect)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                trackers.append(tracker)
                # apply non-maxima suppression to suppress weak, overlapping

    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(image)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'up' or 'down'
    cv2.line(image, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

    
    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    print("human_detection reached")

   # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        print("looping over objects")

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                if direction < 0 and centroid[1] < H // 2:
                    totalPersonsExited += 1
                    countOfExited += 1
                    to.counted = True

                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[1] > H // 2:
                    totalPersonsEntered += 1
                    countOfEntered += 1
                    to.counted = True

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Exited Persons Count", totalPersonsExited),
        ("Entered Persons Count", totalPersonsEntered),
        ("Status", status),
    ]
    print(info)
    print("total of people in room: " + str(totalPersonsEntered - totalPersonsExited))

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(image, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # check to see if we should write the frame to disk
    if writer is not None:
        writer.write(image)

    # show the output frame
    cv2.putText(image, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
            (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.imwrite("FLY.png", image)
    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    raw_capture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    totalFrames += 1

print("Total People Entered:", totalPersonsEntered)
print("Total People Exited:", totalPersonsExited)

# close any open windows
cv2.destroyAllWindows()