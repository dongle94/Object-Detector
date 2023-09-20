# -*- coding: utf-8 -*-
"""
datavoucher p1 - 와이이노베이션
Product object 10 classes
Manual labeling script
"""

import json
import os
import sys
import argparse
from datetime import datetime
from collections import defaultdict
import cv2
import shutil
import numpy as np
import platform

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
os.chdir(ROOT)

from utils.config import _C as cfg
from utils.config import update_config
from utils.logger import get_logger, init_logger


img = None
mouseX, mouseY = 0, 0
box_point = []


def get_box_point(pt1, pt2):
    """
    return box point xyxy with 2 points
    :param pt1:
    :param pt2:
    :return new_pt1, new_pt2:
    """
    x1, y1 = pt1
    x2, y2 = pt2
    new_pt1 = (min(x1, x2), min(y1, y2))
    new_pt2 = (max(x1, x2), max(y1, y2))
    return new_pt1, new_pt2


def draw_event(event, x, y, flags, param):
    global mouseX, mouseY, img, box_point
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(img, (x, y), 5, (32, 216, 32), -1)
        box_point.append((x, y))
        if len(box_point) % 2 == 0:
            box_pt1, box_pt2 = get_box_point(box_point[-2], box_point[-1])
            cv2.rectangle(img, box_pt1, box_pt2, (32, 32, 216), 2, cv2.LINE_AA)
        cv2.imshow(param, img)
    if event == cv2.EVENT_MOUSEMOVE:
        im = img.copy()
        img_size = im.shape
        cv2.line(im, (x, 0), (x, img_size[0]), (0, 0, 0), 2, cv2.LINE_AA)
        cv2.line(im, (0, y), (img_size[1], y), (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow(param, im)


def main(opt=None):
    get_logger().info(f"Start dv p1 annotation script. Object class is {opt.class_num}")
    IMGS_DIR = opt.imgs_dir
    get_logger().info(f"Input Directory is {IMGS_DIR}")

    obj_classes = defaultdict(int)

    if os.path.exists(opt.json_file):
        with open(opt.json_file, 'r') as file:
            basic_fmt = json.load(file)
        get_logger().info(f"{opt.json_file} is exist. append annotation file.")
        img_ids = 0
        if len(basic_fmt['images']) != 0:
            img_ids = int(basic_fmt['images'][-1]['id']) + 1
            get_logger().info(f"last {img_ids}th image file name: {basic_fmt['images'][-1]['file_name']}")
        anno_ids = 0
        if len(basic_fmt['annotations']) != 0:
            anno_ids = int(basic_fmt["annotations"][-1]['id']) + 1
            for anno in basic_fmt['annotations']:
                # for img_anno in basic_fmt['images']:
                #     if img_anno['id'] == anno['image_id'] and anno['category_id'] == 3:
                #         print(img_anno['file_name'])
                obj_classes[int(anno['category_id'])] += 1
            get_logger().info(f"old object classes: {obj_classes}")
    else:
        get_logger().info(f"{opt.json_file} is not exist. Create new annotation file")
        basic_fmt = {
            "info": {"year": "2023", "version": "1",
                     "description": "datavoucher industrial garbage detection dataset",
                     "contributor": "",
                     "url": "",
                     "date_created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")},
            "licenses": [{"id": 1, "url": "", "name": "Unknown"}],
            "categories": [
                {"id": 0, "name": "background", "supercategory": "background"},
                {"id": 1, "name": "battery_cell", "supercategory": "garbage"},
                {"id": 2, "name": "earphone", "supercategory": "garbage"},
                {"id": 3, "name": "metal_box", "supercategory": "garbage"},
                {"id": 4, "name": "monitor", "supercategory": "garbage"},
                {"id": 5, "name": "phone", "supercategory": "garbage"},
                {"id": 6, "name": "remote_controller", "supercategory": "garbage"},
                {"id": 7, "name": "smart_watch", "supercategory": "garbage"},
                {"id": 8, "name": "speaker", "supercategory": "garbage"},
                {"id": 9, "name": "tablet", "supercategory": "garbage"},
                {"id": 10, "name": "tumbler", "supercategory": "garbage"},
            ],
            "images": [],
            "annotations": []
        }
        img_ids = 0
        anno_ids = 0

    image_extension = ['.jpg', '.png', '.jpeg', '.bmp']

    IMGS = [i for i in os.listdir(IMGS_DIR) if os.path.splitext(i)[-1].lower() in image_extension]
    IMGS.sort()

    for idx, i in enumerate(IMGS):
        img_file = os.path.join(IMGS_DIR, i)
        get_logger().info(f"process {img_file}.")
        f0 = cv2.imread(img_file)
        if os.path.exists(img_file) is True and f0 is None:      # File 경로에 한글
            f0 = open(img_file.encode("utf8"), mode="rb")
            bs = bytearray(f0.read())
            arr = np.asarray(bs, dtype=np.uint8)
            f0 = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

        # Connect click event
        winname = f"{idx+1}/{len(IMGS)}"
        cv2.namedWindow(winname)
        cv2.setMouseCallback(winname, draw_event, winname)

        img_info = {
            "id": img_ids,
            "license": 1,
            "file_name": os.path.join(opt.type, i),
            "height": f0.shape[0],
            "width": f0.shape[1],
            "data_captured": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        }

        tmp_annos = []

        # image resize
        f1 = f0.copy()
        orig_img_size = (f0.shape[0], f0.shape[1])
        edit_img_size = orig_img_size
        global img
        img = f1
        while f1.shape[0] >= 1080:
            f1 = cv2.resize(f1, (int(f1.shape[1] * 0.8), int(f1.shape[0] * 0.8)))
            img = f1
            edit_img_size = (f1.shape[0], f1.shape[1])

        # winname에 한글 입력 불가
        cv2.imshow(winname, f1)

        k = cv2.waitKey(0)
        if k == ord('q'):
            get_logger().info("-- CV2 Stop --")
            break
        elif k == ord(" "):
            global box_point
            get_logger().info(f"Boxes list: {box_point}")
            if len(box_point) % 2 == 0:
                for box_i in range(0, len(box_point), 2):
                    pt1, pt2 = get_box_point(box_point[box_i], box_point[box_i+1])
                    rel_pt1 = (pt1[0]/edit_img_size[1], pt1[1]/edit_img_size[0])
                    rel_pt2 = (pt2[0]/edit_img_size[1], pt2[1]/edit_img_size[0])
                    orig_pt1 = (int(rel_pt1[0] * orig_img_size[1]), int(rel_pt1[1] * orig_img_size[0]))
                    orig_pt2 = (int(rel_pt2[0] * orig_img_size[1]), int(rel_pt2[1] * orig_img_size[0]))
                    w = int(orig_pt2[0] - orig_pt1[0])
                    h = int(orig_pt2[1] - orig_pt1[1])

                    anno_info = {
                        "id": anno_ids,
                        "image_id": img_ids,
                        "category_id": opt.class_num,
                        "bbox": [orig_pt1[0], orig_pt1[1], w, h],
                        "area": w * h,
                        "segmentation": [],
                        "iscrowd": 0
                    }
                    tmp_annos.append(anno_info)
                    anno_ids += 1
                    obj_classes[opt.class_num] += 1

                # add annotation
                basic_fmt["images"].append(img_info)
                for _anno_info in tmp_annos:
                    basic_fmt["annotations"].append(_anno_info)
                box_point = []

                new_path = os.path.join(IMGS_DIR, opt.type, i)
                if not os.path.exists(os.path.dirname(new_path)):
                    os.makedirs(os.path.dirname(new_path))
                shutil.move(img_file, new_path)
                img_ids += 1

                get_logger().info(
                    f"Save label {img_file}. Add {len(box_point)} boxes."
                )

            else:
                print("2 points not clicked!")
                break
        else:
            continue

        cv2.destroyAllWindows()

    with open(os.path.join(opt.json_file), 'w') as outfile:
        json.dump(basic_fmt, outfile, indent=1, ensure_ascii=False)
    get_logger().info(f"Stop Annotation. Obj classes: {obj_classes}")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imgs_dir', required=True,
                        help='image directory path')
    parser.add_argument('-j', '--json_file', required=True,
                        help="if write this file, append annotations")
    parser.add_argument('-t', '--type', default='train',
                        help='type is in [train, val]. this option write file_path {type}/img_file')
    parser.add_argument('-cn', '--class_num', required=True, type=int,
                        help="object class number 1~10")
    parser.add_argument('-c', '--config', default='./configs/annotate.yaml',
                        help="annotate.yaml config file path")
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = args_parse()

    update_config(cfg, args=args.config)
    init_logger(cfg)

    main(args)
