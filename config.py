from easydict import EasyDict as edict
import os

def get_config():
    conf=edict()
    conf.cfg='cfg/char.cfg'
    conf.names='data/char.names'
    conf.weights='weights/char.pt'
    conf.img_size=320
    conf.conf_thres=0.4 #object confidence threshold
    conf.iou_thres=0.5 #IOU threshold for NMS
    conf.half = False 
    conf.device='' # =>gpu  or if cpu => cpu
    conf.view_img=False
    conf.save_txt=False
    conf.classes=None
    conf.agnostic_nms=False
    conf.augment=False
    return conf
