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
from crop_small_image import random_crop_and_aug_train_data, crop_and_save_sliding_window
from labelme_coco_conversion import labelme_to_coco

sys.path.append(os.path.join(aoi_dir, "config"))
from config import read_config_yaml

sys.path.append(os.path.join(aoi_dir, "utils"))
from logger import get_logger

# Read config_file
config_file = os.path.join(aoi_dir, 'config/config.yaml')
config = read_config_yaml(config_file)

# Get logger
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
logger = get_logger(name=__file__, console_handler_level=logging.INFO, file_handler_level=None)

if __name__ == '__main__':
    train_data_dir = config['train_data_dir']
    retrain_data_train_dir = config['retrain_data_train_dir']
    pcb_data_dir = config['pcb_data_dir']

    # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Remove black border and mask foreground bbox  #
    #   Output: train_data/images_wo_border           #
    #           train_data/labels_wo_border           #
    # # # # # # # # # # # # # # # # # # # # # # # # # #
    train_data_dict = {
        'data_dir': train_data_dir,
        'mask_foreground_object': False,
        'save_shrink_image': True
    }
    remove_black_border_specific_data(train_data_dict)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Random_crop                                   #
    #   Output: train_data/images_random_crop_w_aug   #
    #           train_data/labels_random_crop_w_aug   #
    # # # # # # # # # # # # # # # # # # # # # # # # # #
    # v5
    aug_num_dict = {'bridge': 2,
                    'empty': 2,
                    'appearance_less': 2,
                    'appearance_hole': 2,
                    'excess_solder': 10,
                    'appearance': 100}
    random_crop_and_aug_train_data(aug_num_dict, train_data_dir)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Sliding_window and mask those samples across margin   #
    #   Output: train_data/images_sliding_crop_mask_margin    #
    #           train_data/labels_sliding_crop_mask_margin    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    train_data_mask_fg_across_margin_dict = {
        'image_wo_border_dir': os.path.join(train_data_dir, 'images_wo_border'),
        'label_wo_border_dir': os.path.join(train_data_dir, 'labels_wo_border'),
        'image_sliding_crop_dir': os.path.join(train_data_dir, 'images_sliding_crop_mask_margin'),
        'label_sliding_crop_dir': os.path.join(train_data_dir, 'labels_sliding_crop_mask_margin'),
        'mask_fg_across_margin': True
    }
    crop_and_save_sliding_window(train_data_mask_fg_across_margin_dict)

    # # # # # # # # # # # # # # # # # # # # # # # #
    #   Convert labelme labels to coco format     #
    #   Output: pcb_data/annotations/train.json   #
    #           pcb_data/train                    #
    # # # # # # # # # # # # # # # # # # # # # # # #
    train_data_dict = {
        'crop_image_dir_list': [ os.path.join(train_data_dir, 'images_random_crop_w_aug'), 
                                 os.path.join(train_data_dir, 'images_sliding_crop_mask_margin'),
                                 os.path.join(retrain_data_train_dir, 'images_random_crop_w_aug') ], # .jpg
        'crop_label_dir_list': [ os.path.join(train_data_dir, 'labels_random_crop_w_aug'), 
                                 os.path.join(train_data_dir, 'labels_sliding_crop_mask_margin'),
                                 os.path.join(retrain_data_train_dir, 'labels_random_crop_w_aug') ], # .json
        'pcb_data_annotations_dir': os.path.join(pcb_data_dir, 'annotations'),
        'pcb_data_images_dir': os.path.join(pcb_data_dir, 'train'),
        'pcb_data_labels_dir': None,
        'pcb_data_json_file_path': os.path.join(pcb_data_dir, 'annotations/train.json')
    }
    labelme_to_coco(train_data_dict)
