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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        print(input_imgs.shape)
        print(type(input_imgs))
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            print(input_imgs.shape)
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()


if __name__ == "__main__":
    #main()
    model = init_yolo()
    read_shared_mem(model)
