import os
import numpy as np
import cv2
import time
import argparse
import xml.etree.ElementTree as ET

# Argument for --name and directory of the images
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--name', help='Put the name of the person.', default="Visitor")
arg_parser.add_argument('--images', help='Put the directory of the images.', default=None)

args = arg_parser.parse_args()

person_name = args.name
image_folder = args.images

person_name_counter = 0                                          
Face_Folder = person_name
if not os.path.exists(Face_Folder):
    os.makedirs(Face_Folder)

# Load our serialized model from disk
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'face.caffemodel')

# Loop over all images in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # Construct the full path to the image
        image_path = os.path.join(image_folder, filename)

        # Load the input image and construct an input blob
        frame = cv2.imread(image_path)
        # I'm doing resizing here based on my CNN Algorithm
        frame = cv2.resize(frame, (300, 300))
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            y = startY - 10 if startY - 10 > 10 else startY + 10

            ################################################################################################
            current_time = time.strftime("%M-%S")
            # For Saving Annotation
            image_name = f"{person_name} {person_name_counter} {current_time}.jpg"
            image_path = os.path.join(Face_Folder, image_name)
            cv2.imwrite(image_path, frame)

            # For Annotation
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
            # For Saving Annotation
            xml_filename = f"{person_name} {person_name_counter} {current_time}.xml"
            xml_path = os.path.join(Face_Folder, xml_filename)
            tree = ET.ElementTree(annotation)
            tree.write(xml_path)
            person_name_counter += 1
            cv2.imshow("VisioAccelerAI-Automatic-Labeller", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

cv2.destroyAllWindows()
