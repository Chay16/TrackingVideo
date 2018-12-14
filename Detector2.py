import torch
from torch.autograd import Variable
from darknet import Darknet
import cv2 
import numpy as np
from util import load_classes,write_results
class YoloTorch:
    def __init__(self,cfgfile="darknet/cfg/yolov3.cfg",weightsfile="../YOLO weights/yolov3.weights",reso="416",confidence = 0.5,nms_thesh = 0.4):
        self.CUDA = torch.cuda.is_available()
        self.classes = load_classes('data/coco.names') 
        self.nb_c=len(self.classes)
        self.Td=confidence
        self.Tnms=nms_thesh
        
        
        
        self.model = Darknet(cfgfile)
        self.model.load_weights(weightsfile)
        self.model.net_info["height"] = reso
        self.inp_dim = int(self.model.net_info["height"])
        assert self.inp_dim % 32 == 0 
        assert self.inp_dim > 32
        if self.CUDA:
            self.model.cuda()
        self.model.eval()
    def preprocess(self,img):
        img = cv2.resize(img, (self.inp_dim, self.inp_dim)) 
        img_ =  img[:,:,::-1].transpose((2,0,1))
        img_ = img_[np.newaxis,:,:,:]/255.0
        img_ = torch.from_numpy(img_).float()
        img_ = Variable(img_)
        if self.CUDA:
            img_ = img_.cuda()
        return img_
    def predict(self,img):
        img=self.preprocess(img)
        with torch.no_grad():
            dets=self.model(img, self.CUDA)
            
        res=write_results(dets, self.Td,  self.nb_c , nms = True, nms_conf = self.Tnms)
        detection=[]
        for r in res:
            detection.append([r[1:5].tolist(),self.classes[int(r[-1])],float(r[5])])
        return  detection
