from __future__ import print_function
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from data import VOCroot
from data import AnnotationTransform,VOCDetection, BaseTransform, VOC_Config
from models.RFB_Net_vgg import build_net
import torch.utils.data as data
from layers.functions import Detect,PriorBox
from utils.nms_wrapper import nms
from utils.timer import Timer
from utils.visualize import print_info
from tqdm import tqdm
cfg = VOC_Config
testset = VOCDetection(VOCroot, [('2007', 'person_test')], None, AnnotationTransform())
det_file='eval/VOC/detections.pkl'
save_folder='eval/VOC'
with open(det_file, 'rb') as f:
    all_boxes = pickle.load(f)

print('Evaluating detections')
testset.evaluate_detections(all_boxes, save_folder)