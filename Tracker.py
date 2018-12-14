import cv2
import os
import numpy as np
from tqdm import tqdm as tq
from darkpy import Yolo

####################### Global Variables #############################

path = "Sequences/"
os.chdir(path)

################### Basic uselful functions ##########################

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

folder = "camel/"
weights_file = "../YOLO weights/yolov3.weights"
detector = Yolo(weights_file)

frames = getFrames(folder)
masks  = getMasks(folder)
height, width, channel = cv2.imread(frames[0]).shape
video_name_gt = folder.split("/")[0]+"_groundtruth"
video_name_predict = folder.split("/")[0]+"predict"

video_gt = init_video(video_name_gt,width,height)

for i in tq(range(len(frames))):
    frame_gt = cv2.imread(frames[i])
    mask = cv2.imread(masks[i])
    corner1_gt,corner3_gt = getBBox_Mask(mask)
    cv2.rectangle(frame_gt,corner1_gt,corner3_gt,(0,255,0),2)

    video_gt.write(frame_gt)

close_video_writer(video_gt)