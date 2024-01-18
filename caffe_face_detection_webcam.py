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

# Arguement for --name 
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--name', help='Put the name of the person.', default="Visitor")
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

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
	frame = vs.read()

	# Change this based sa input mo sa CNN
	# frame = imutils.resize(frame, height=300, width=300)
	frame = cv2.resize(frame, (1080, 1080))

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence < 0.5:
			continue
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		y = startY - 10 if startY - 10 > 10 else startY + 10


################################################################################################
		current_time = time.strftime("%Y-%m-%d %H-%M-%S")
		#For Saving Annotation
		image_name = f"{current_time} {person_name}.jpg"
		image_path = os.path.join(Face_Folder, image_name)
	
		#For Annotation
		annotation = ET.Element('annotation')
		folder = ET.SubElement(annotation, 'folder')
		folder.text = 'Face-Detected'
		filename = ET.SubElement(annotation, 'filename')
		filename.text = image_name
		path = ET.SubElement(annotation, 'path')
		path.text = os.path.abspath(image_path)
		source = ET.SubElement(annotation, 'source')
		database = ET.SubElement(source, 'database')
		database.text = 'Unknown'
		size = ET.SubElement(annotation, 'size')
		width_elem = ET.SubElement(size, 'width')
		width_elem.text = str(320)
		height_elem = ET.SubElement(size, 'height')
		height_elem.text = str(320)
		depth = ET.SubElement(size, 'depth')
		depth.text = str(3)
		segmented = ET.SubElement(annotation, 'segmented')
		segmented.text = '0'
		obj = ET.SubElement(annotation, 'object')
		name = ET.SubElement(obj, 'name')
		name.text = person_name
		pose = ET.SubElement(obj, 'pose')
		pose.text = 'Unspecified'
		truncated = ET.SubElement(obj, 'truncated')
		truncated.text = '1'
		difficult = ET.SubElement(obj, 'difficult')
		difficult.text = '0'
		bndbox = ET.SubElement(obj, 'bndbox')
		xmin_elem = ET.SubElement(bndbox, 'xmin')
		xmin_elem.text = str(startX)
		ymin_elem = ET.SubElement(bndbox, 'ymin')
		ymin_elem.text = str(startY)
		xmax_elem = ET.SubElement(bndbox, 'xmax')
		xmax_elem.text = str(endX)
		ymax_elem = ET.SubElement(bndbox, 'ymax')
		ymax_elem.text = str(endY)
		#For Saving Annotation
		xml_filename = f"{current_time} {person_name}.xml"
		xml_path = os.path.join(Face_Folder, xml_filename)
		tree = ET.ElementTree(annotation)

	cv2.imshow("VisioAccelerAI Data Collector", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("c"):
		cv2.imwrite(image_path, frame) 
		tree.write(xml_path)
		print("Image captured and annotate and saved!")

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()