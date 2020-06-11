#!/usr/bin/env python3

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
import asyncio
import json
import multiprocessing
import threading

from PIL import Image

import torch
import socket
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from multiprocessing import shared_memory, Process, Manager
from multiprocessing.managers import SyncManager

from PIL import Image


FRAMES_SIZE = 128
INDEX_SIZE = 24
HEADER_SIZE = 8
BUFFER_SIZE = 100 * 1024 * 1024
CONTENT_OFFSET = 4096
CONTENT_SIZE = BUFFER_SIZE - CONTENT_OFFSET
IMAGE_SIZE = 416
CONF_THRES = .8
NMS_THRES = .4
CLASS_PATH = 'data/coco.names'
CLASSES = None
CLIENTS = Manager().dict()
SERVER_BARRIER = multiprocessing.Barrier(2)
UNIX_SOCKET_NAME = '/tmp/yolo_stream'


def init_yolo():
    print('Init yolo')
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
def process_image(model, index, data, width, height, timestamp, frame_sequence):
    print("Process image #%d[%d] of size (%dx%d) captured at %d" % (index, frame_sequence, width, height, timestamp))
    data = np.frombuffer(data, dtype=np.uint8).reshape((height, width, -1))
    data = data[:, :, [2, 1, 0]]
    data = np.rollaxis(data, 2)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    image = torch.Tensor(data).type(Tensor) / 255
    image, _ = pad_to_square(image, 0)
    image = resize(image, IMAGE_SIZE)
    # save_image(image)
    detections = model(image.unsqueeze(0))
    detections = non_max_suppression(detections, CONF_THRES, NMS_THRES)
    for detection in detections :
        if detection is not None:
            detection = rescale_boxes2(detection, IMAGE_SIZE, [height, width])
            for detc in detection:
                x1, y1, x2, y2, conf, cls_conf, cls_pred = detc.cpu().detach().numpy().tolist()
                print("\t+ Label: %s, Conf: %.5f at [%dx%d - %dx%d]" % (CLASSES[int(cls_pred)], cls_conf, x1, y1, x2, y2))
                # on_result({'frame_sequence': frame_sequence, 'frame_timestamp': timestamp, 'yolo_timestamp': time.time(), 'detection': detc.cpu().detach().numpy().tolist()})
                on_result({'frame_sequence': frame_sequence, 'frame_timestamp': timestamp, 'yolo_timestamp': time.time(), 'detection': {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'conf': conf, 'cls_conf': cls_conf, 'cls_pred': cls_pred, 'cls_pred_name': CLASSES[int(cls_pred)]
                    }})
        else:
            print('Nothing is detected')


def read_shared_mem(model):
    print('Read shared memory')
    while True:
        try:
            shm = shared_memory.SharedMemory(name='/webrtc_frames', create=False)
            break
        except:
            print("The shared memory is not ready, sleep 1s")
            time.sleep(1)
    assert shm.size == BUFFER_SIZE

    def get_size():
        return struct.unpack('I', bytes(shm.buf[:4]))[0]

    def get_offset():
        return struct.unpack('I', bytes(shm.buf[:4]))[0]

    def get_frame_info(i):
        return struct.unpack('IIHHIIi', bytes(shm.buf[HEADER_SIZE + i * INDEX_SIZE : HEADER_SIZE + (i + 1) * INDEX_SIZE]))

    size = get_size()
    print("Number of frames: %d" % size)
    index = size - 1 if size > 0 else 0
    while True:
        if index < get_size():
            i = index % FRAMES_SIZE
            offset, length, width, height, timestamp, frame_sequence, finished = get_frame_info(i)
            if get_size() > index:
                _, _, _, _, next_timestamp, next_frame_sequence, next_finished = get_frame_info((i + 1) % FRAMES_SIZE)
                if next_finished == 1:
                    print('New frame is ready, skip frame #%d[%d], captured at %d' % (index, next_frame_sequence, next_timestamp))
                    index += 1
                    continue
            if finished == 1: # check the finished tag
                process_image(model, index,
                        bytes(shm.buf[CONTENT_OFFSET + offset: CONTENT_OFFSET + offset + length]),
                        width, height, timestamp, frame_sequence)
                index += 1
    shm.close()


def on_result(result):
    result = json.dumps(result)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(UNIX_SOCKET_NAME)
    sock.send(result.encode())


class UdpServerProtocol:
    def __init__(self):
        self.port = 4400

    def connection_made(self, transport):
        self.transport = transport

    def connection_lost(self, transport):
        pass

    def on_result(self, result):
        for client in CLIENTS.keys():
            print('Send result to: %s, result: %s' % (client, result))
            self.transport.sendto(result.encode(), client)

    def datagram_received(self, data, addr):
        message = data.decode()
        if addr not in CLIENTS:
            print("Register new client: %s" % (addr, ))
            CLIENTS[addr] = 1


class UnixServerProtocol(asyncio.BaseProtocol):
    def connection_made(self, transport):
        self._transport = transport

    def data_received(self, data):
        data = data.decode().strip()
        SERVER_PROTOCOL.on_result(data)

    def eof_received(self):
        pass


async def start_udp_server():
    print('Starting UDP server')
    loop = asyncio.get_running_loop()
    transport1, _ = await loop.create_datagram_endpoint(lambda: SERVER_PROTOCOL, local_addr=('0.0.0.0', SERVER_PROTOCOL.port))
    unix_server = await loop.create_unix_server(UnixServerProtocol, path=UNIX_SOCKET_NAME)
    try:
        await asyncio.sleep(5)
        SERVER_BARRIER.wait()
        await asyncio.sleep(3600)  # Serve for 1 hour.
    finally:
        transport1.close()
        unix_server.close()


def object_detection():
    model = init_yolo()
    SERVER_BARRIER.wait()
    read_shared_mem(model)


SERVER_PROTOCOL = UdpServerProtocol()


def main():
    process = Process(target=object_detection)
    process.start()
    asyncio.run(start_udp_server())
    process.join()


if __name__ == "__main__":
    main()
