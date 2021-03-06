from ctypes import *
import pkg_resources
import numpy as np

resource_package = __name__
lib_file = pkg_resources.resource_stream(resource_package, 'libdarknet.so').name

config_file_classic = pkg_resources.resource_stream(resource_package, "yolov3_416.cfg").name
config_file_9000 = pkg_resources.resource_stream(resource_package, "yolo9000.cfg").name

weights_file_classic = pkg_resources.resource_stream(resource_package, "yolov3.weights").name
weights_file_9000 = pkg_resources.resource_stream(resource_package, "yolo9000.weights").name

meta_file_classic = pkg_resources.resource_stream(resource_package, "coco.data").name
meta_file_9000 = pkg_resources.resource_stream(resource_package, "combine9k.data").name

def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


lib = CDLL(lib_file, RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def detect_np(net, meta, np_img, thresh=.5, hier_thresh=.5, nms=.45):
    im, image = array_to_image(np_img)
    rgbgr_image(im)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh,
                             hier_thresh, None, 0, pnum)
    num = pnum[0]
    if nms: do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        a = dets[j].prob[0:meta.classes]
        if any(a):
            ai = np.array(a).nonzero()[0]
            for i in ai:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i],
                            (b.x, b.y, b.w, b.h)))

    res = sorted(res, key=lambda x: -x[1])
    if isinstance(image, bytes): free_image(im)
    free_detections(dets, num)
    return res


def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


class Yolo:
    """ Detector is a simple API to the darknet framework for python.
        Use the function detect to detect from image objects.
        To redefine the format of the detections outputed edit the function dark_detection2detection in this repo.
    Parameters
    ----------
    cfg_file : str
        Darknet configuration file ( .cfg) of the network, (network descreption).
    weights_file : str
        Darknet weight file (.weight).
    meta_file : str
        Darknet metadata file (.data) describing the liste of classes and other metadata.
    """

    def __init__(self, cfg_file=config_file_9000, weights=weights_file_9000, meta=meta_file_9000):
        cfg_file = cfg_file.encode('utf-8')
        weights_file = weights.encode('utf-8')
        meta_file = meta.encode('utf-8')

        self.net = load_net(cfg_file, weights_file, 0)
        self.meta = load_meta(meta_file)

    def detect(self, image,thres = 0.2):

        out = detect_np(self.net, self.meta, image,thres)
        dets=[]
        for e in out:
            d = list(e)
            d[2] = list(d[2])
            dets.append([d[2], d[0].decode("utf-8"), d[1]])
        return dets