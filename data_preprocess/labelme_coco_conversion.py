#!/usr/bin/env python3
import os, sys
import cv2
import json
import logging

aoi_dir_name = os.getenv('AOI_DIR_NAME')
assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"

current_dir = os.path.dirname(os.path.abspath(__file__))
idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
aoi_dir = current_dir[:idx]

from crop_small_image import invert_bbox_rect
sys.path.append(os.path.join(aoi_dir, "config"))
from config import read_config_yaml

sys.path.append(os.path.join(aoi_dir, "utils"))
from utils import os_makedirs, shutil_copyfile
from logger import get_logger

# Read config_file
config_file = os.path.join(aoi_dir, 'config/config.yaml')
config = read_config_yaml(config_file)

# Get logger
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
logger = get_logger(name=__file__, console_handler_level=logging.DEBUG, file_handler_level=None)

category_mapping = {'bridge': 'bridge',
                    'empty': 'appearance_less',
                    'appearance_less': 'appearance_less',
                    'appearance_hole': 'appearance_less',
                    'excess_solder': 'excess_solder',
                    'appearance': 'appearance'
                   }
category_dict = {'bridge': 1, 'appearance_less': 2, 'excess_solder': 3, 'appearance': 4}

def get_key_from_value(dict_obj, val):
    for key, value in dict_obj.items():
         if value == val:
             return key
    return "other"

def labelme_to_coco(input_dict):
    crop_image_dir_list = input_dict['crop_image_dir_list']
    crop_label_dir_list = input_dict['crop_label_dir_list']
    pcb_data_annotations_dir = input_dict['pcb_data_annotations_dir']
    pcb_data_images_dir = input_dict['pcb_data_images_dir']
    pcb_data_labels_dir = input_dict['pcb_data_labels_dir']
    pcb_data_json_file_path = input_dict['pcb_data_json_file_path']

    os_makedirs(pcb_data_annotations_dir, keep_exists=True)
    os_makedirs(pcb_data_images_dir)
    if pcb_data_labels_dir:
        os_makedirs(pcb_data_labels_dir)

    images, categories, annotations = [], [], []
    anno_id = 0
    fg = 0
    bg = 0
    offset = 0
    shift = -1
    for category in category_dict:
        categories.append({"supercategory": "pcb_defect", "id": category_dict[category], "name": category})

    for idx, (crop_image_dir, crop_label_dir) in enumerate(zip(crop_image_dir_list, crop_label_dir_list)):
        offset += (shift + 1)
        logger.debug('\noffset = {}'.format(offset))
        for image_id, image_file_name in enumerate(os.listdir(crop_image_dir)):
            logger.debug('image_id = {} ; offset + image_id = {}'.format(image_id, offset + image_id))
            shift = image_id

            image_file_path = os.path.join(crop_image_dir ,image_file_name)
            dst_file_path = os.path.join(pcb_data_images_dir ,image_file_name)
            shutil_copyfile(image_file_path, dst_file_path)

            json_file_name = os.path.splitext(image_file_name)[0] + '.json'
            json_file_path = os.path.join(crop_label_dir ,json_file_name)
            if pcb_data_labels_dir:
                dst_file_path = os.path.join(pcb_data_labels_dir ,json_file_name)
                shutil_copyfile(json_file_path, dst_file_path)

            image = cv2.imread(image_file_path)
            try:
                image_h, image_w, _ = image.shape
            except:
                print(image_file_name)
                continue
            images.append({"file_name": image_file_name, "height": image_h, "width": image_w, "id": offset + image_id})

            if os.path.isfile(json_file_path):
                fg += 1
                with open(json_file_path, 'r') as json_file:
                    json_dict = json.load(json_file)
                    file_name = os.path.basename(json_dict["imagePath"])
                    for element in json_dict["shapes"]:
                        error_type = element["label"]
                        if error_type in category_mapping:
                            category_id = category_dict[category_mapping[error_type]]
                            xmin = int(element["points"][0][0])
                            ymin = int(element["points"][0][1])
                            xmax = int(element["points"][1][0])
                            ymax = int(element["points"][1][1])
                            xmin, ymin, xmax, ymax = invert_bbox_rect([xmin, ymin, xmax, ymax])
                            w = xmax - xmin
                            h = ymax - ymin
                            bbox = [xmin, ymin, w, h]
                            segment = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, xmin, ymin]
                            area = w * h
                            anno_info = {'id': anno_id,
                                         'category_id': category_id,
                                         'bbox': bbox,
                                         'segmentation': [segment],
                                         'area': area,
                                         'iscrowd': 0,
                                         'image_id': offset + image_id
                                        }
                            annotations.append(anno_info)
                            anno_id += 1
                        else:
                            logger.info('error_type = {} is not supported'.format(error_type))
                            continue
            else:
                bg += 1
    logger.info("fg image = {}; bg image = {}".format(fg, bg))
    all_json_dict = {"images": images, "annotations": annotations, "categories": categories}

    with open(pcb_data_json_file_path, 'w') as json_file:
        json.dump(all_json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '))

if __name__ == '__main__':
    train_data_dir = config['train_data_dir']
    test_data_dir = config['test_data_dir']
    retrain_data_train_dir = config['retrain_data_train_dir']

    train_data_dict = {
        'crop_image_dir_list': [os.path.join(train_data_dir, 'crop_images_fg_bg')], # .jpg
        'crop_label_dir_list': [os.path.join(train_data_dir, 'crop_labels_fg_bg')], # .json
        'pcb_data_annotations_dir': os.path.join(train_data_dir, 'pcb_data/annotations'),
        'pcb_data_images_dir': os.path.join(train_data_dir, 'pcb_data/train'),
        'pcb_data_labels_dir': os.path.join(train_data_dir, 'pcb_data/train_json'),
        'pcb_data_json_file_path': os.path.join(train_data_dir, 'pcb_data/annotations/train.json')
    }
    test_data_dict = {
        'crop_image_dir_list': [os.path.join(test_data_dir, 'images_random_crop_wo_aug')], # .jpg
        'crop_label_dir_list': [os.path.join(test_data_dir, 'labels_random_crop_wo_aug')], # .json
        'pcb_data_annotations_dir': os.path.join(test_data_dir, 'pcb_data/annotations'),
        'pcb_data_images_dir': os.path.join(test_data_dir, 'pcb_data/test'),
        'pcb_data_labels_dir': os.path.join(train_data_dir, 'pcb_data/test_json'),
        'pcb_data_json_file_path': os.path.join(test_data_dir, 'pcb_data/annotations/test.json')
    }

    retrain_data_dict = {
        'crop_image_dir_list': [os.path.join(retrain_data_train_dir, 'images_random_crop_w_aug')], # .jpg
        'crop_label_dir_list': [os.path.join(retrain_data_train_dir, 'labels_random_crop_w_aug')], # .json
        'pcb_data_annotations_dir': os.path.join(retrain_data_train_dir, 'pcb_data/annotations'),
        'pcb_data_images_dir': os.path.join(retrain_data_train_dir, 'pcb_data/retrain'),
        'pcb_data_json_file_path': os.path.join(retrain_data_train_dir, 'pcb_data/annotations/retrain.json')
    }

    janet_data_dict = {
        'crop_image_dir_list': ['/home/aoi/AOI_PCB_Retrain_detectron2/original_aug/JPEGImages'], # .jpg
        'crop_label_dir_list': ['/home/aoi/AOI_PCB_Retrain_detectron2/original_aug/labelme'], # .json
        'pcb_data_annotations_dir': '/home/aoi/AOI_PCB_Retrain_detectron2/original_aug/pcb_data_janet/annotations',
        'pcb_data_images_dir': '/home/aoi/AOI_PCB_Retrain_detectron2/original_aug/pcb_data_janet/train',
        'pcb_data_json_file_path': '/home/aoi/AOI_PCB_Retrain_detectron2/original_aug/pcb_data_janet/annotations/train.json'
    }

    labelme_to_coco(retrain_data_dict)