import cv2
import os
import numpy as np
from tqdm import tqdm as tq
from darkpy import Yolo
from skimage.measure import regionprops

####################### Global Variables #############################

#os.chdir('/Users/Chayan/Desktop/TrackingVideo/TrackingVideoYOLO')
original_path = os.getcwd()
path = "Sequences/"
os.chdir(path)

################### Basic uselful functions ##########################

def getFolderNames():
    """

    :return: list of the folder path containing the sequences
    """

    folders = []
    for file in os.listdir(os.getcwd()):
        if os.path.isdir(file):
            folders.append(file+"/")
    return folders


def getFrames(folder):
    """

    :param folder: name of the folder containing the video frames and masks
    :return: list of the paths only to the video frames
    """

    frames = []
    for file in os.listdir(folder):
        if file.split(".")[1] == "bmp":
            frames.append(folder+file)
    frames.sort()
    return frames


def getMasks(folder):
    """

    :param folder: name of the folder containing the video frames and masks
    :return: list of the paths only to the masks
    """

    masks = []
    for file in os.listdir(folder):
        if file.split(".")[1] == "png":
            masks.append(folder+file)
    masks.sort()
    return masks


def getBBox_Mask(mask):
    """

    :param mask: mask of the object we are interessed in
    :return: corners of the associated bounding box
    """

    ind = np.where(mask == mask.max())
    corner1 = (min(ind[1]), min(ind[0]))
    corner3 = (max(ind[1]), max(ind[0]))
    return corner1, corner3


def draw_BBox(frame,bbox):
    """

    :param frame: frame viewed as matrice
    :param bbox: bounding box XY coordinates, witdh, height
    :return: frame with the bouding box plot on it
    """

    x,y,w,h = bbox
    cv2.rectangle(frame, (x-w//2,y-h//2), (x+w//2,y+h//2), (0,255,0), 2)
    return frame

def centroid_assessment(groundtruth,estimated):
    a = regionprops(groundtruth)
    b = regionprops(estimated)
    return np.linalg.norm(np.array(a[0].centroid)-np.array(b[0].centroid))


def evaluate_centroid_dist(gt,predicted):
    dist = centroid_assessment(gt, predicted)
    return dist


def cornerDistance(previousBBox,currentBBox):
    xp, yp, wp, hp = previousBBox
    xc, yc, wc, hc = currentBBox
    current_corner = ((xc-wc//2,yc-hc//2), (xc+wc//2, yc+hc//2))
    previous_corner = ((xp-wp//2,yp-hp//2), (xp+wp//2, yp+hp//2))
    dist = np.sqrt((previous_corner[0][0]-current_corner[0][0])**2+(previous_corner[0][1]-current_corner[0][1])**2+
                   (previous_corner[1][0]-current_corner[1][0])**2+(previous_corner[1][1]-current_corner[1][1])**2)
    return dist


def BBox_Tuple2List(tple):
    liste = []

    liste.append([tple[0][0],tple[0][1],tple[1][0] - tple[0][0],tple[1][1] - tple[0][1]])
    liste.append('')
    liste.append(0)

    return liste


######################## Video functions #############################

def init_video(video_name, width, height):
    """

    :param video_name: name of the video file
    :param width: width of each frame of the video
    :param height: height of each frame of the video
    :return: initialisation of video writer
    """

    video_name = video_name + ".mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_name, fourcc, 24, (width, height))

    return video


def close_video_writer(video):
    """

    :return: Finish the writing into the video
    """

    video.release()
    cv2.destroyAllWindows()


########################## Main ##################################

folders = getFolderNames()

folder = "camel/"
detector = Yolo()

#for folder in folders:
#    print(folder)
frames = getFrames(folder)
masks  = getMasks(folder)
bbox_previous = getBBox_Mask(cv2.imread(masks[0]))
print(bbox_previous)
bbox_previous = BBox_Tuple2List(bbox_previous)
print("Previous BBox : ",bbox_previous)

height, width, channel = cv2.imread(frames[0]).shape
video_name_gt = folder.split("/")[0]+"_groundtruth"
video_name_predict = folder.split("/")[0]+"predict"

video_gt = init_video(video_name_gt,width,height)
video_predict = init_video(video_name_predict,width,height)

for i in tq(range(len(frames))):
#for i in [0,1,2,3,4,5]:
    frame_gt = cv2.imread(frames[i])
    frame_predict = cv2.imread(frames[i])

    # Ground Truth
    mask = cv2.imread(masks[i])
    corner1_gt,corner3_gt = getBBox_Mask(mask)
    cv2.rectangle(frame_gt,corner1_gt,corner3_gt,(0,255,0),2)
    video_gt.write(frame_gt)

    # YOLO Prediction
    detection = detector.detect(frame_predict,0.2)

    detection_kept = detection[0]
    for i in range(1,len(detection)):
        if cornerDistance(bbox_previous[0],detection[i][0])<cornerDistance(bbox_previous[0],detection_kept[0]):
            detection_kept = detection[i]

    print("New Bbox : " ,detection_kept)
    bbox_previous = detection_kept
    draw_BBox(frame_predict,list(map(int,detection_kept[0])))
    video_predict.write(frame_predict)
    """for i in detection:
        draw_BBox(frame_predict,list(map(int,i[0])))
        print(i)
    video_predict.write(frame_predict)"""

close_video_writer(video_gt)
close_video_writer(video_predict)

os.chdir(original_path)