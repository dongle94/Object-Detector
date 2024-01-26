# Ultralytics YOLO 🚀, AGPL-3.0 license
from .base import BasePredictor
from .predict import DetectionPredictor
# from .train import DetectionTrainer
# from .val import DetectionValidator

__all__ = "BasePredictor", "DetectionPredictor",     # "DetectionTrainer", "DetectionValidator"


