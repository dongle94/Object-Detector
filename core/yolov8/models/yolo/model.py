# Ultralytics YOLO 🚀, AGPL-3.0 license

from core.yolov8.engine.model import Model
from core.yolov8.models import yolo
from core.yolov8.nn.tasks import DetectionModel, OBBModel


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""
    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }
