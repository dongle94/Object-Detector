import os
import sys
import cv2
import numpy as np
import torch
import torch.nn


from core.yolov8.models.yolo import YOLOV8


if __name__ == "__main__":
    model = YOLOV8('yolov8n.pt')

    result = model("/home/dongle94/Pictures/exercise.jpg")
    # print(result)
