import os
import cv2
import math
import numpy as np
from pathlib import Path
import time

from core.medialoader import load_images, load_video, load_stream


def check_sources(source):
    img_formats = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'
    vid_formats = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'
    is_imgs, is_vid, is_stream = False, False, False
    if os.path.isdir(source) or '*' in source:
        is_imgs = True
    elif os.path.isfile(source) and Path(source).suffix[1:] in img_formats:
        is_imgs = True
    elif os.path.isfile(source) and Path(source).suffix[1:] in vid_formats:
        is_vid = True
    elif source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
        is_stream = True
    elif source.isnumeric() or source.endswith('.streams') or source.startswith('/dev'):
        is_stream = True

    return is_imgs, is_vid, is_stream


class MediaLoader(object):
    def __init__(self, source, stride=1, logger=None, realtime=False, opt=None, bgr=True):

        self.stride = stride
        self.logger = logger
        self.realtime = realtime
        self.opt = opt
        self.bgr = bgr

        self.is_imgs, self.is_vid, self.is_stream = check_sources(source)

        if self.is_imgs:
            dataset = load_images.LoadImages(source, bgr=self.bgr)
        elif self.is_vid:
            dataset = load_video.LoadVideo(source, stride=self.stride, realtime=self.realtime, bgr=self.bgr)
        elif self.is_stream:
            dataset = load_stream.LoadStream(source, stride=self.stride, opt=self.opt, bgr=self.bgr)
        else:
            raise NotImplementedError(f'Invalid input: {source}')

        self.dataset = dataset

        # self.alive = True
        # self.bpause = False

    def get_frame(self):
        im = self.dataset.__next__()
        return im

    def show_frame(self, title: str = 'frame', wait_sec: int = 0):
        frame = self.get_frame()
        if self.bgr is False:
            frame = frame[..., ::-1]
        cv2.imshow(title, frame)
        if cv2.waitKey(wait_sec) == ord('q'):
            if self.logger is not None:
                self.logger.info("-- Quit Show frames")
            raise StopIteration

    def __del__(self):
        if hasattr(self, 'dataset'):
            del self.dataset


if __name__ == "__main__":
    # import sys
    from utils.config import set_config, get_config

    # s = sys.argv[1]      # video file, webcam, rtsp stream... 0etc
    set_config('./configs/config.yaml')
    _cfg = get_config()
    _bgr = getattr(_cfg, 'media_bgr', True)
    _realtime = getattr(_cfg, 'media_realtime', False)

    # _media_loader = MediaLoader(s, bgr=True)
    _media_loader = MediaLoader(_cfg.media_source, bgr=_bgr, realtime=_realtime, opt=_cfg)
    print("-- MediaLoader is ready")

    _title = 'frame'
    wt = 30

    _frame = _media_loader.get_frame()
    print("-- Frame Metadata:", _frame.shape, _frame.dtype)

    cv2.imshow(_title, _frame[..., ::-1])
    cv2.waitKey(wt)

    while True:
        _frame = _media_loader.show_frame(title=_title, wait_sec=wt)
