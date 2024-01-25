# Ultralytics YOLO 🚀, AGPL-3.0 license

from core.yolov8.models.yolo import detect
# from core.yolov8.models.yolo import obb

from .model import YOLO

__all__ = "detect", "obb", "YOLO"
