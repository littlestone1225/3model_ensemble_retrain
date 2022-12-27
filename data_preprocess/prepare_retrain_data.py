#!/usr/bin/env python3
import os, sys
import logging

aoi_dir_name = os.getenv('AOI_DIR_NAME')
assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"

current_dir = os.path.dirname(os.path.abspath(__file__))
idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
aoi_dir = current_dir[:idx]

from crop_small_image import crop_retrain_data

sys.path.append(os.path.join(aoi_dir, "config"))
from config import read_config_yaml

sys.path.append(os.path.join(aoi_dir, "inference"))
from inference_fn import inference

sys.path.append(os.path.join(aoi_dir, "utils"))
from utils import os_makedirs, shutil_copyfile
from logger import get_logger

# Read config_file
config_file = os.path.join(aoi_dir, 'config/config.yaml')
config = read_config_yaml(config_file)

# Get logger
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
logger = get_logger(name=__file__, console_handler_level=logging.INFO, file_handler_level=None)

def convert_to_original_preprocess():
    retrain_data_org_dir = config['retrain_data_org_dir']
    retrain_data_org_pre_dir = config['retrain_data_org_pre_dir']

    fn_image_dir = os.path.join(retrain_data_org_pre_dir, 'images')
    fn_label_dir = os.path.join(retrain_data_org_pre_dir, 'labels')

    os_makedirs(fn_image_dir)
    os_makedirs(fn_label_dir)

    for root, dirs, files in os.walk(retrain_data_org_dir):
        for file_name in files:
            src_file_path = os.path.join(root, file_name)

            if os.path.splitext(file_name)[1] == '.jpg':
                dst_file_path = os.path.join(fn_image_dir, file_name)
            elif os.path.splitext(file_name)[1] == '.json':
                dst_file_path = os.path.join(fn_label_dir, file_name)

            shutil_copyfile(src_file_path, dst_file_path)


if __name__ == '__main__':
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Rearrange retraining images and labels            #
    #   Input:  retrain_data/original                     #
    #   Output: retrain_data/original_preprocess/images   #
    #           retrain_data/original_preprocess/labels   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    convert_to_original_preprocess()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Inference retrain data for old model                                      #
    #   Output: retrain_data/original_preprocess/inference_result/result_images   #
    #           retrain_data/original_preprocess/inference_result/result_labels   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    retrain_data_org_pre_dir = config['retrain_data_org_pre_dir']

    retrain_data_dict = {
        'image_dir': os.path.join(retrain_data_org_pre_dir, 'images'),
        'centernet2_label_dir': os.path.join(retrain_data_org_pre_dir, 'inference_result/centernet2_json'),
        'retinanet_label_dir': os.path.join(retrain_data_org_pre_dir, 'inference_result/retinanet_json'),
        'yolov4_label_dir': os.path.join(retrain_data_org_pre_dir, 'inference_result/yolov4_json'),
        'inference_result_image_dir': os.path.join(retrain_data_org_pre_dir, 'inference_result/result_images'),
        'inference_result_label_dir': os.path.join(retrain_data_org_pre_dir, 'inference_result/result_labels'),
        'inference_result_txt_dir': os.path.join(retrain_data_org_pre_dir, 'inference_result/result_txts'),
        'image_wo_border_dir': None, #os.path.join(retrain_data_org_pre_dir, 'inference_result/images_wo_border'),
        'image_backup_dir': None,
        'inference_result_image_backup_dir': None
    }

    inference(retrain_data_dict)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Crop retrain data for training                        #
    #   Output: retrain_data/train/images_random_crop_w_aug   #
    #           retrain_data/train/labels_random_crop_w_aug   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # v5
    aug_num_dict = {'bridge': 20,
                    'empty': 20,
                    'appearance_less': 20,
                    'appearance_hole': 20,
                    'excess_solder': 20,
                    'appearance': 20}
    '''
    # v6
    aug_num_dict = {'bridge': 1,
                    'empty': 1,
                    'appearance_less': 1,
                    'appearance_hole': 1,
                    'excess_solder': 1,
                    'appearance': 1}
    '''
    crop_retrain_data('train',aug_num_dict)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Crop retrain data for validation and testing                                    #
    #   Output: retrain_data/val/images_random_crop (eliminate_bbox_in_crop_rect=True)  #
    #           retrain_data/val/labels_random_crop (eliminate_bbox_in_crop_rect=True)  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    crop_retrain_data('val')
