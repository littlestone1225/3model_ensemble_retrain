#!/usr/bin/env python3
import os, sys
import csv
import cv2
import numpy as np
import json
from collections import OrderedDict
import logging

aoi_dir_name = os.getenv('AOI_DIR_NAME')
assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"

current_dir = os.path.dirname(os.path.abspath(__file__))
idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
aoi_dir = current_dir[:idx]

sys.path.append(os.path.join(aoi_dir, "config"))
from config import read_config_yaml

sys.path.append(os.path.join(aoi_dir, "utils"))
from logger import get_logger

# Read config_file
config_file = os.path.join(aoi_dir, 'config/config.yaml')
config = read_config_yaml(config_file)

# Get logger
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
logger = get_logger(name=__file__, console_handler_level=logging.DEBUG, file_handler_level=None)

category_to_num_dict = {'bridge': 0, 'appearance_less': 1, 'excess_solder': 2, 'appearance': 3}

def csv_to_dashboard_txt(label_array, dashboard_txt_dir, coord_type="xmin_ymin_xmax_ymax"):
    """
    Convert csv format to dashboard txt format.

    Args:
        label_array (list[list] or np.ndarray): annotations in [[file_name, error_type, xmin, ymin, xmax, ymax, score, timestamp], ...] format.
        dashboard_txt_dir (str): directory in which txt file is stored
        coord_type (str): "xmin_ymin_w_h" or "xmin_ymin_xmax_ymax"
    Output:
        [index, xmin, ymin, w, h, score, error_num, timestamp]
    """
    dashboard_list = list()
    if len(label_array) > 0:
        image_file_name  = label_array[0][0]

        for idx, label in enumerate(label_array):
            error_type = label[1]
            error_num = category_to_num_dict[error_type]

            if coord_type == "xmin_ymin_w_h":
                xmin, ymin, w, h = [int(i) for i in label[2:6]]
            elif coord_type == "xmin_ymin_xmax_ymax":
                xmin, ymin, xmax, ymax = [int(i) for i in label[2:6]]
                w = xmax - xmin
                h = ymax - ymin
            else:
                assert False, 'coord_type should be either xmin_ymin_w_h or xmin_ymin_xmax_ymax'

            score = label[6]
            timestamp = label[7]
            dashboard_list.append([idx+1, xmin, ymin, w, h, score, error_num, timestamp])

        txt_file_name = os.path.splitext(image_file_name)[0] + '.txt'
        txt_file_path = os.path.join(dashboard_txt_dir, txt_file_name)
        with open(txt_file_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(dashboard_list)

def csv_to_json(label_array, image_dir, label_json_dir, coord_type="xmin_ymin_w_h", store_score=False):
    """
    Convert csv format to labelme json format.

    Args:
        label_array (list[list] or np.ndarray): annotations in [[file_name, error_type, xmin, ymin, xmax, ymax, score], ...] format.
                                                (score is optional and is controlled by store_score)
        image_dir (str): directory of image file (file_name = xxx.jpg).
        label_json_dir (str): directory in which json file is stored
        coord_type (str): "xmin_ymin_w_h" or "xmin_ymin_xmax_ymax"
        store_score (boolean): determine if you want to store score in json file or not
    """
    json_dict = OrderedDict()

    if len(label_array) > 0:
        for idx, label in enumerate(label_array):
            file_name  = label[0]
            error_type = label[1]

            if coord_type == "xmin_ymin_w_h":
                xmin, ymin, w, h = [int(i) for i in label[2:6]]
                xmax = xmin + w
                ymax = ymin + h
            elif coord_type == "xmin_ymin_xmax_ymax":
                xmin, ymin, xmax, ymax = [int(i) for i in label[2:6]]
            else:
                assert False, 'coord_type should be either xmin_ymin_w_h or xmin_ymin_xmax_ymax'

            if idx==0:
                json_dict["version"] = "4.5.6"
                json_dict["flags"] = dict()
                json_dict["shapes"] = list()
                json_dict["imagePath"] = file_name
                json_dict["imageData"] = None

                image_file_path = os.path.join(image_dir, file_name)

                if os.path.isfile(image_file_path):
                    image = cv2.imread(image_file_path)
                    json_dict["imageHeight"] = image.shape[0]
                    json_dict["imageWidth"] = image.shape[1]
                else:
                    logger.warning("{} does not exist".format(image_file_path))
                    return

            shapes = OrderedDict()
            shapes["label"] = error_type
            shapes["points"] = [[xmin, ymin], [xmax, ymax]]
            shapes["group_id"] = None
            shapes["shape_type"] = "rectangle"
            shapes["flags"] = dict()
            if store_score and len(label) >= 7:
                score = float(label[6])
                shapes["score"] = score
            json_dict["shapes"].append(shapes)

        json_file_name = os.path.splitext(file_name)[0] + '.json'
        json_file_path = os.path.join(label_json_dir, json_file_name)
        with open(json_file_path, 'w') as json_file:
            json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '))

def csv_to_json_one_by_one(xmin, ymin, w, h, error_type, jpg_dir, jpg_file_name, json_file_path):
    xmax = xmin + w
    ymax = ymin + h

    if os.path.isfile(json_file_path):
        with open(json_file_path, 'r') as json_file:
            json_dict = json.load(json_file, object_pairs_hook=OrderedDict)

            shapes = OrderedDict()
            shapes["label"] = error_type
            shapes["points"] = [[xmin, ymin], [xmax, ymax]]
            shapes["group_id"] = None
            shapes["shape_type"] = "rectangle"
            shapes["flags"] = dict()
            json_dict["shapes"].append(shapes)
    else:
        json_dict = OrderedDict()

        json_dict["version"] = "4.5.6"
        json_dict["flags"] = dict()
        json_dict["shapes"] = list()
        json_dict["imagePath"] = jpg_file_name
        json_dict["imageData"] = None

        jpg_file_path = os.path.join(jpg_dir, jpg_file_name)
        jpg_crop = cv2.imread(jpg_file_path)
        json_dict["imageHeight"] = jpg_crop.shape[0]
        json_dict["imageWidth"] = jpg_crop.shape[1]

        shapes = OrderedDict()
        shapes["label"] = error_type
        shapes["points"] = [[xmin, ymin], [xmax, ymax]]
        shapes["group_id"] = None
        shapes["shape_type"] = "rectangle"
        shapes["flags"] = dict()
        json_dict["shapes"].append(shapes)

    with open(json_file_path, 'w') as json_file:
        json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '))

def json_to_bbox(json_file_path, store_score=False):  # return [[file_name, error_type, xmin, ymin, xmax, ymax], ...]
    """
    Convert labelme json format to csv format.

    Args:
        json_file_path (str): json file path.
        store_score (boolean): determine if you want to store score in bbox_list or not
    Returns:
        bbox_list (list[list]): annotations in [[file_name, error_type, xmin, ymin, xmax, ymax, score], ...] format.
                                (score is optional and is controlled by store_score)
    """
    bbox_list = list()
    with open(json_file_path, 'r') as json_file:
        json_dict = json.load(json_file)
        file_name = os.path.basename(json_dict["imagePath"])
        for element in json_dict["shapes"]:
            error_type = element["label"]
            xmin = int(element["points"][0][0])
            ymin = int(element["points"][0][1])
            xmax = int(element["points"][1][0])
            ymax = int(element["points"][1][1])
            if store_score:
                score = element.get("score")
                bbox_list.append([file_name, error_type, xmin, ymin, xmax, ymax, score])
            else:
                bbox_list.append([file_name, error_type, xmin, ymin, xmax, ymax])
    return bbox_list

def get_width_height_from_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        json_dict = json.load(json_file)
        width = json_dict['imageWidth']
        height = json_dict['imageHeight']
    return width, height

