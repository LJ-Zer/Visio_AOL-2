# Visio_Automation-of-Labelling_V2

## Preparation
1. Install git
2. Git clone this repository
3. Open gitbash on the folder

## Execution
1. Edit the "run" file, open it on notepad or VSCode.
  - change the "your_name_here" with your actual name.
2. Execute the run.sh file in the gitbash
  - source run.sh

## For Real-Time Data Gathering
Note: No need to run the source run.sh, just activate the virtual environment and execute this code by putting the right "name".

python caffe_face_detection_webcam_no_time.py --name "your_name_here"

## For images only 
Note: Don't execute this, this command is only used after the data augmentation process.

python caffe_face_detection_multiple-images.py --name "your_name_here" --images "path/of/your/images"