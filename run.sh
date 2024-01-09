conda create --name caffe python=3.6
conda activate caffe
pip install -r requirements.txt
# Change your_name_here with your actual name
python detect_faces_video.py --name "your_name_here"