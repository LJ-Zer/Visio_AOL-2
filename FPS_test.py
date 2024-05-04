import parser
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import argparse
import glob
import importlib.util
import xml.etree.ElementTree as ET
from threading import Thread
import os

# Argument for --name 
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--name', help='Put the name of the person.', default="face")
args = arg_parser.parse_args()
person_name = args.name


Face_Folder = (person_name) 
if not os.path.exists(Face_Folder):
  os.makedirs(Face_Folder)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'face.caffemodel')

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# initialize FPS variables
start_time = time.time()
fps = 0

# loop over the frames from the video stream
while True:
  # grab the frame from the threaded video stream and resize it
  frame = vs.read()

  # Change this based sa input mo sa CNN
  # frame = imutils.resize(frame, height=300, width=300)
  frame = cv2.resize(frame, (2560, 1440))

  (h, w) = frame.shape[:2]
  blob = cv2.dnn.blobFromImage(cv2.resize(frame, (640, 360)), 1.0, (640, 360), (104.0, 177.0, 123.0))
  net.setInput(blob)
  detections = net.forward()

  # loop over the detections
  for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence < 0.8:
      continue
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    y = startY - 10 if startY - 10 > 10 else startY + 10

    # Extract the face ROI (region of interest)
    face_roi = frame[startY:endY, startX:endX]

    # Generate a unique filename based on timestamp
    current_time = time.strftime("%Y-%m-%d %H-%M-%S")
    image_name = f"{current_time}_{person_name}.jpg"
    image_path = os.path.join(Face_Folder, image_name)

    # Save the face ROI as an image
    cv2.imwrite(image_path, face_roi)

  # Update FPS variables
  elapsed_time = time.time() - start_time
  fps = 1 / elapsed_time
  start_time = time.time()  # Reset for next iteration

  # Display FPS on the frame
# Display FPS on the frame
  display_frame = cv2.resize(frame, (640, 360))  # Change display size here
  cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

  cv2.imshow("VisioAccelerAI Data Collector", display_frame)

  key = cv2.waitKey(1) & 0xFF

  if key == ord("q"):
    break

cv2.destroyAllWindows()
vs.stop()
