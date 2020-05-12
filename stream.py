from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import struct
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from multiprocessing import shared_memory

from PIL import Image


FRAMES_SIZE = 128
INDEX_SIZE = 20
HEADER_SIZE = 8
BUFFER_SIZE = 100 * 1024 * 1024
CONTENT_OFFSET = 4096
CONTENT_SIZE = BUFFER_SIZE - CONTENT_OFFSET
IMAGE_SIZE = 416
CONF_THRES = .8
NMS_THRES = .4
CLASS_PATH = 'data/coco.names'


CLASSES = None

def init_yolo():
    global CLASSES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: %s' % device)
    model = Darknet('config/yolov3.cfg', img_size=IMAGE_SIZE).to(device)
    model.load_darknet_weights('weights/yolov3.weights')
    CLASSES = load_classes(CLASS_PATH)
    model.eval()
    return model


def save_image(tensor, path='test.jpg'):
    _, height, width = tensor.shape
    tensor *= 255
    bytes = tensor.cpu().byte().numpy()
    bytes = np.rollaxis(bytes, 0, 3).copy()
    print(bytes.shape)
    img = Image.frombuffer('RGB', [width, height], bytes)
    img.save(path)


def log_time(name):
    def wrapping(func):
        def wrapped(*args, **kwargs):
            ts = time.time()
            func(*args, **kwargs)
            diff = (time.time() - ts) * 1000 # in millisecond
            print('%s causes %f ms' % (name, diff))
        return wrapped
    return wrapping


@log_time("Process image")
def process_image(model, index, data, width, height, timestamp):
    print("Process image #%d of size (%dx%d) captured at %d" % (index, width, height, timestamp))
    data = np.frombuffer(data, dtype=np.uint8).reshape((height, width, -1))
    data = data[:, :, [2, 1, 0]]
    data = np.rollaxis(data, 2)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    image = torch.Tensor(data).type(Tensor) / 255
    image, _ = pad_to_square(image, 0)
    image = resize(image, IMAGE_SIZE)
    #save_image(image)
    detections = model(image.unsqueeze(0))
    detections = non_max_suppression(detections, CONF_THRES, NMS_THRES)
    for detection in detections :
        if detection is not None:
            detection = rescale_boxes(detection, IMAGE_SIZE, [height, width])
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                print("\t+ Label: %s, Conf: %.5f at [%dx%d - %dx%d]" % (CLASSES[int(cls_pred)], cls_conf.item(), x1, y1, x2, y2))
        else:
            print('Nothing is detected')


def read_shared_mem(model):
    shm = shared_memory.SharedMemory(name='/webrtc_frames', create=False)
    assert shm.size == BUFFER_SIZE

    def get_size():
        return struct.unpack('I', bytes(shm.buf[:4]))[0]

    def get_offset():
        return struct.unpack('I', bytes(shm.buf[:4]))[0]

    def get_frame_info(i):
        return struct.unpack('IIHHIi', bytes(shm.buf[HEADER_SIZE + i * INDEX_SIZE : HEADER_SIZE + (i + 1) * INDEX_SIZE]))

    size = get_size()
    print("Number of frames: %d" % size)
    index = size - 1 if size > 0 else 0
    while True:
        if index < get_size():
            i = index % FRAMES_SIZE
            frame_info = get_frame_info(i)
            if get_size() > index:
                frame_info_next = get_frame_info((i + 1) % FRAMES_SIZE)
                if frame_info_next[5] == 1:
                    print('New frame is ready, skip frame #%d, captured at %d' % (index, frame_info_next[4]))
                    index += 1
                    continue
            if frame_info[5] == 1: # check the finished tag
                offset = frame_info[0]
                length = frame_info[1]
                process_image(model, index, bytes(shm.buf[CONTENT_OFFSET + offset : CONTENT_OFFSET + offset + length]), *frame_info[2:5])
                index += 1
    shm.close()


if __name__ == "__main__":
    model = init_yolo()
    read_shared_mem(model)
