import os
import sys

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
os.chdir(ROOT)

from obj_detectors.yolov5_pt import YoloDetector


class ObjectDetector(object):
    def __init__(self, cfg=None):
        # Detection model configuration
        if os.path.abspath(cfg.DET_MODEL_PATH) != cfg.DET_MODEL_PATH:
            weight = os.path.abspath(os.path.join(ROOT, cfg.DET_MODEL_PATH))
        else:
            weight = os.path.abspath(cfg.DET_MODEL_PATH)
        self.detector_type = cfg.DET_MODEL_TYPE.lower()

        if self.detector_type == "yolo":
            device = cfg.DEVICE
            fp16 = cfg.HALF
            img_size = cfg.IMG_SIZE
            self.im_shape = None
            self.im0_shape = None

            # model load with weight
            self.detector = YoloDetector(weight=weight, device=device, img_size=img_size, fp16=fp16)

            # warm up
            self.detector.warmup(imgsz=(1, 3, img_size, img_size))

    def preprocess(self, img):
        if self.detector_type == "yolo":
            img, orig_img = self.detector.preprocess(img)
            self.im_shape = img.shape
            self.im0_shape = orig_img.shape

        return img

    def detect(self, img):
        preds = None
        if self.detector_type == "yolo":
            preds = self.detector.forward(img)

        return preds

    def postprocess(self, ret):
        preds, dets = None, None
        if self.detector_type == 'yolo':
            max_det = 100
            preds, dets = self.detector.postprocess(pred=ret, im_shape=self.im_shape,
                                                    im0_shape=self.im0_shape, max_det=max_det)
            preds = preds.cpu().numpy()
            dets = dets.cpu().numpy()

        return preds, dets


if __name__ == "__main__":
    import time
    import cv2
    from utils.config import _C as cfg
    from utils.config import update_config
    from utils.medialoader import MediaLoader

    update_config(cfg, args='./config.yaml')
    detector = ObjectDetector(cfg=cfg)

    s = sys.argv[1]
    media_loader = MediaLoader(s)
    time.sleep(1)
    while True:
        frame = media_loader.get_frame()

        im = detector.preprocess(frame)
        _pred = detector.detect(im)
        _pred, _det = detector.postprocess(_pred)

        for d in _det:
            x1, y1, x2, y2 = map(int, d[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)

        cv2.imshow('_', frame)
        if cv2.waitKey(1) == ord('q'):
            print("-- CV2 Stop --")
            break

    media_loader.stop()
    print("-- Stop program --")
