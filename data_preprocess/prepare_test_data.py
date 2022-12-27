#!/usr/bin/env python3
import os, sys
import cv2
import numpy as np
import logging

aoi_dir_name = os.getenv('AOI_DIR_NAME')
assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"

current_dir = os.path.dirname(os.path.abspath(__file__))
idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
aoi_dir = current_dir[:idx]

from remove_black_border import remove_black_border_specific_data
from crop_small_image import random_crop_test_data
from labelme_coco_conversion import labelme_to_coco

sys.path.append(os.path.join(aoi_dir, "config"))
from config import read_config_yaml

sys.path.append(os.path.join(aoi_dir, "utils"))
from utils import os_makedirs
from logger import get_logger

# Read config_file
config_file = os.path.join(aoi_dir, 'config/config.yaml')
config = read_config_yaml(config_file)

# Get logger
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
logger = get_logger(name=__file__, console_handler_level=logging.INFO, file_handler_level=None)

if __name__ == '__main__':
    test_data_dir = config['test_data_dir']
    retrain_data_val_dir = config['retrain_data_val_dir']
    pcb_data_dir = config['pcb_data_dir']

    # # # # # # # # # # # # # # # # # # # # #
    #   Remove black border                 #
    #   Output: test_data/images_wo_border  #
    #           test_data/labels_wo_border  #
    # # # # # # # # # # # # # # # # # # # # #
    test_data_dict = {
        'data_dir': test_data_dir,
        'mask_foreground_object': False,
        'save_shrink_image': False
    }
    remove_black_border_specific_data(test_data_dict)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Random_crop                                                               #
    #   Output: test_data/images_random_crop (eliminate_bbox_in_crop_rect=True)   #
    #           test_data/labels_random_crop (eliminate_bbox_in_crop_rect=True)   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    random_crop_test_data(test_data_dir)

    # # # # # # # # # # # # # # # # # # # # # # #
    #   Convert labelme labels to coco format   #
    #   Output: pcb_data/annotations/test.json  #
    #           pcb_data/test                   #
    #           pcb_data/test_json              #
    # # # # # # # # # # # # # # # # # # # # # # #
    test_data_dict = {
        'crop_image_dir_list': [ os.path.join(test_data_dir, 'images_random_crop'),
                                 os.path.join(retrain_data_val_dir, 'images_random_crop') ], # .jpg
        'crop_label_dir_list': [ os.path.join(test_data_dir, 'labels_random_crop'),
                                 os.path.join(retrain_data_val_dir, 'labels_random_crop') ], # .json
        'pcb_data_annotations_dir': os.path.join(pcb_data_dir, 'annotations'),
        'pcb_data_images_dir': os.path.join(pcb_data_dir, 'test'),
        'pcb_data_labels_dir': None, # os.path.join(pcb_data_dir, 'test_json'),
        'pcb_data_json_file_path': os.path.join(pcb_data_dir, 'annotations/test.json')
    }

    labelme_to_coco(test_data_dict)
