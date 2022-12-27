#!/usr/bin/env python3
import os, sys
import random
import subprocess
import logging
import glob

aoi_dir_name = os.getenv('AOI_DIR_NAME')
assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"

current_dir = os.path.dirname(os.path.abspath(__file__))
idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
aoi_dir = current_dir[:idx]

from labelme_coco_conversion import labelme_to_coco

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
logger = get_logger(name=__file__, console_handler_level=logging.INFO, file_handler_level=None)

sample_list = [
    'LKA3534786_LDIA019DA_9697Q6C711E_700',
    'LKA3534790_LDIA019DA_9697Q6C711E_700',
    'LKA3534799_LDIA019DA_9697Q6C711E_700',
    'LKA3534800_LDIA019DA_9697Q6C711E_700',
    'LKA3534803_LDIA019DA_9697Q6C711E_700',
    'LKA3534853_LDIA019DA_9697Q6C711E_700',
    'LKA3534859_LDIA019DA_9697Q6C711E_700',
    'LKA3534899_LDIA019DA_9697Q6C711E_700',
    'LKA3534936_LDIA019DA_9697Q6C711E_700',
    'LKA3534945_LDIA019DA_9697Q6C711E_700',
    'LKA3535024_LDIA019DA_9697Q6C711E_700',
    'LKA3535041_LDIA019DA_9697Q6C711E_700',
    'LKA3535043_LDIA019DA_9697Q6C711E_700',
    'LKA3535063_LDIA019DA_9697Q6C711E_700',
    'LKA3535086_LDIA019DA_9697Q6C711E_700',
    'LKA3535092_LDIA019DA_9697Q6C711E_700',
    'LKA3535113_LDIA019DA_9697Q6C711E_700',
    'LKA3535130_LDIA019DA_9697Q6C711E_700',
    'LKA3535134_LDIA019DA_9697Q6C711E_700',
    'LKA3535159_LDIA019DA_9697Q6C711E_700',
    'LKA3535160_LDIA019DA_9697Q6C711E_700',
    'LKA3535172_LDIA019DA_9697Q6C711E_700',
    'LKA3535201_LDIA019DA_9697Q6C711E_700',
    'LKA3535208_LDIA019DA_9697Q6C711E_700',
    'LKA3535223_LDIA019DA_9697Q6C711E_700',
    'LKA3535252_LDIA019DA_9697Q6C711E_700',
    'LKA3535256_LDIA019DA_9697Q6C711E_700',
    'LKA3535266_LDIA019DA_9697Q6C711E_700',
    'LKA3535306_LDIA019DA_9697Q6C711E_700',
    'LKA3535319_LDIA019DA_9697Q6C711E_700',
    'LKA3535331_LDIA019DA_9697Q6C711E_700',
    'LKA3535361_LDIA019DA_9697Q6C711E_700',
    'LKA3535367_LDIA019DA_9697Q6C711E_700',
    'LKA3535385_LDIA019DA_9697Q6C711E_700',
    'LKA3535412_LDIA019DA_9697Q6C711E_700',
    'LKA3535425_LDIA019DA_9697Q6C711E_700',
    'LKA3535464_LDIA019DA_9697Q6C711E_700',
    'LKA3535468_LDIA019DA_9697Q6C711E_700',
    'LKA3535471_LDIA019DA_9697Q6C711E_700',
    'LKA3542791_LDIA001DA_9697Q1B101E-D01_470',
    'LKA3542889_LDIA001DA_9697Q1B101E-D01_470',
    'LKA3542913_LDIA001DA_9697Q1B101E-D01_470',
    'LKA3542920_LDIA001DA_9697Q1B101E-D01_470',
    'LKA3542947_LDIA001DA_9697Q1B101E-D01_470',
    'LKA3542977_LDIA001DA_9697Q1B101E-D01_470',
    'LKA3542994_LDIA001DA_9697Q1B101E-D01_470',
    'LKA3543002_LDIA001DA_9697Q1B101E-D01_470',
    'LKA3543010_LDIA001DA_9697Q1B101E-D01_470',
    'LKA3543129_LDIA001DA_9697Q1B101E-D01_470',
    'LKA3543141_LDIA001DA_9697Q1B101E-D01_470',
    'LKA3543144_LDIA001DA_9697Q1B101E-D01_470',
    'LKA3543146_LDIA001DA_9697Q1B101E-D01_470',
    'LKA3543154_LDIA001DA_9697Q1B101E-D01_470',
    'LKA3543203_LDIA001DA_9697Q1B101E-D01_470',
    'LKA3543234_LDIA001DA_9697Q1B101E-D01_470',
    'LKA3549154_LDIA018DA_9697Q1B101E-D02_101',
    'LKA3549181_LDIA018DA_9697Q1B101E-D02_101',
    'LKA3549199_LDIA018DA_9697Q1B101E-D02_101',
    'LKA3549222_LDIA018DA_9697Q1B101E-D02_101',
    'LKA3549235_LDIA018DA_9697Q1B101E-D02_101',
    'LKA3549246_LDIA018DA_9697Q1B101E-D02_101',
    'LKA3549715_LDIA012DA_9697Q1B101E-D02_301',
    'LKA3549728_LDIA012DA_9697Q1B101E-D02_301',
    'LKA3549754_LDIA012DA_9697Q1B101E-D02_301',
    'LKA3549763_LDIA012DA_9697Q1B101E-D02_301',
    'LKA3549764_LDIA012DA_9697Q1B101E-D02_301',
    'LKA3549798_LDIA012DA_9697Q1B101E-D02_301',
    'LKA3549812_LDIA012DA_9697Q1B101E-D02_301',
    'LKA3549887_LDIA012DA_9697Q1B101E-D02_301',
    'LKA3549937_LDIA012DA_9697Q1B101E-D02_301',
    'LKA3549940_LDIA012DA_9697Q1B101E-D02_301',
    'LKA3549956_LDIA012DA_9697Q1B101E-D02_301',
    'LKA3549962_LDIA012DA_9697Q1B101E-D02_301',
    'LKA3549970_LDIA012DA_9697Q1B101E-D02_301',
    'LKA3549988_LDIA012DA_9697Q1B101E-D02_301',
    'LKA3549989_LDIA012DA_9697Q1B101E-D02_301',
    'LKA3549997_LDIA012DA_9697Q1B101E-D02_301',
    'LKD0155290_LEIAP12DP_9696W25571E_960',
    'LKD0155334_LEIAP12DP_9696W25571E_960',
    'LKD0155464_LEIAP12DP_9696W25571E_960',
    'LKD0155470_LEIAP12DP_9696W25571E_960',
    'LKD0155578_LEIAP12DP_9696W25571E_960',
    'LKD0155592_LEIAP12DP_9696W25571E_960',
    'LKD0155606_LEIAP12DP_9696W25571E_960',
    'LKD0155642_LEIAP12DP_9696W25571E_960',
    'LKD0155705_LEIAP12DP_9696W25571E_960',
    'LKD0155737_LEIAP12DP_9696W25571E_960',
    'LKD0155785_LEIAP12DP_9696W25571E_960',
    'LKD0155805_LEIAP12DP_9696W25571E_960',
    'LKD0155863_LEIAP12DP_9696W25571E_960',
    'LKD0155871_LEIAP12DP_9696W25571E_960',
    'LKD0155906_LEIAP12DP_9696W25571E_960',
    'LKD0155958_LEIAP12DP_9696W25571E_960',
    'LKD0156046_LEIAP12DP_9696W25571E_960',
    'LKD0156101_LEIAP12DP_9696W25571E_960',
    'LKD0156110_LEIAP12DP_9696W25571E_960',
    'LKD0156148_LEIAP12DP_9696W25571E_960',
    'LKD0156165_LEIAP12DP_9696W25571E_960',
    'LKD0156214_LEIAP12DP_9696W25571E_960',
    'LKD0156217_LEIAP12DP_9696W25571E_960',
]

def sample_100val_from_test(input_dict):
    test_data_dir = input_dict['test_data_dir']
    val_data_dir = input_dict['val_data_dir']

    test_image_dir = os.path.join(test_data_dir, 'images_wo_border') # .jpg
    test_label_dir = os.path.join(test_data_dir, 'labels_wo_border') # .json
    # test_crop_image_dir = os.path.join(test_data_dir, 'images_random_crop') # .jpg
    # test_crop_label_dir = os.path.join(test_data_dir, 'labels_random_crop') # .json

    val_image_dir = os.path.join(val_data_dir, 'images_wo_border') # .jpg
    val_label_dir = os.path.join(val_data_dir, 'labels_wo_border') # .json
    os_makedirs(val_image_dir)
    os_makedirs(val_label_dir)

    # val_crop_image_dir = os.path.join(val_data_dir, 'images_random_crop') # .jpg
    # val_crop_label_dir = os.path.join(val_data_dir, 'labels_random_crop') # .json
    # os_makedirs(val_crop_image_dir)
    # os_makedirs(val_crop_label_dir)

    for sample_name in sample_list:
        sample_image_name = sample_name + '.jpg'
        sample_json_name = sample_name + '.json'
        test_image_path = os.path.join(test_image_dir, sample_image_name)
        test_label_path = os.path.join(test_label_dir, sample_json_name)
        val_image_path = os.path.join(val_image_dir, sample_image_name)
        val_label_path = os.path.join(val_label_dir, sample_json_name)
        shutil_copyfile(test_image_path, val_image_path)
        shutil_copyfile(test_label_path, val_label_path)

def copy_random_crop_from_test_to_val(test_data_dir, val_data_dir, dirname):
    src_dir = os.path.join(test_data_dir, dirname)
    dst_dir = os.path.join(val_data_dir, dirname)
    os_makedirs(dst_dir, keep_exists=False)

    for sample_name in sample_list:
        sample_img_list = glob.glob(os.path.join(src_dir, sample_name + '*'))
        for src_file_path in sample_img_list:
            dst_file_path = src_file_path.replace(src_dir, dst_dir)
            shutil_copyfile(src_file_path, dst_file_path)

if __name__ == '__main__':
    test_data_dir = config['test_data_dir']
    val_data_dir = config['val_data_dir']
    retrain_data_val_dir = config['retrain_data_val_dir']
    pcb_data_dir = config['pcb_data_dir']

    # # # # # # # # # # # # # # # # # # # # # # #
    #   Sample validation data from test data   #
    #   Output: val_data/images_wo_border       #
    #           val_data/labels_wo_border       #
    # # # # # # # # # # # # # # # # # # # # # # #
    data_dict = {
        'test_data_dir': test_data_dir,
        'val_data_dir' : val_data_dir
    }
    sample_100val_from_test(data_dict)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Copy images_random_crop and labels_random_crop from test_data to val_data   #
    #   according to sample_list                                                    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    copy_random_crop_from_test_to_val(test_data_dir, val_data_dir, 'images_random_crop')
    copy_random_crop_from_test_to_val(test_data_dir, val_data_dir, 'labels_random_crop')


    # # # # # # # # # # # # # # # # # # # # # # #
    #   Convert labelme labels to coco format   #
    #   Output: pcb_data/annotations/val.json   #
    #           pcb_data/val                    #
    #           pcb_data/val_json               #
    # # # # # # # # # # # # # # # # # # # # # # #
    val_data_dict = {
        'crop_image_dir_list': [ os.path.join(val_data_dir, 'images_random_crop'),
                                 os.path.join(retrain_data_val_dir, 'images_random_crop') ], # .jpg
        'crop_label_dir_list': [ os.path.join(val_data_dir, 'labels_random_crop'),
                                 os.path.join(retrain_data_val_dir, 'labels_random_crop') ], # .json
        'pcb_data_annotations_dir': os.path.join(pcb_data_dir, 'annotations'),
        'pcb_data_images_dir': os.path.join(pcb_data_dir, 'val'),
        'pcb_data_labels_dir': None, # os.path.join(pcb_data_dir, 'val_json'),
        'pcb_data_json_file_path': os.path.join(pcb_data_dir, 'annotations/val.json')
    }

    labelme_to_coco(val_data_dict)