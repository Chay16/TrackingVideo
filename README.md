# TrackingVideo

Repository for tracking an object in a videogti

Need to install libraries with :
pip install -t requirements.txt

Need to download the weights of the YOLO model with :
wget https://pjreddie.com/media/files/yolov3.weights_

Put the weights file into the "darkpy" folder


Create coco.data in darkypy folder as follow :
classes= 9000
names = YOUR PATH/TrackingVideo/TrackingVideoYOLO/darkpy/9k.names
eval=coco
