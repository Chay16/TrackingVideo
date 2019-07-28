# TrackingVideo

Repository for tracking an object in a videogti

Need to install libraries with :
pip install -r requirements.txt

Need to download the weights of the YOLO model with :
wget https://pjreddie.com/media/files/yolov3.weights_

Put the weights file into the "darkpy" folder


Create coco.data in darkypy folder as follow :
classes= 80
names = YOUR PATH/TrackingVideo/TrackingVideoYOLO/darkpy/coco.names
eval=coco

In combine9k.data and yolo9000.cfg replace all "/Users/Chayan/Desktop/" with YOUR PATH

Our implementation of the GOTURN version is available here : https://colab.research.google.com/drive/1faWd1fuwwL0gBQkDAlj8aNvuX6_s4AmO
