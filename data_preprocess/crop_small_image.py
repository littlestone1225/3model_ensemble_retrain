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

from csv_json_conversion import csv_to_json, json_to_bbox

sys.path.append(os.path.join(aoi_dir, "config"))
from config import read_config_yaml

sys.path.append(os.path.join(aoi_dir, "utils"))
from utils import os_makedirs
from logger import get_logger

# Get logger
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
logger = get_logger(name=__file__, console_handler_level=logging.INFO, file_handler_level=None)

# Read config_file
config_file = os.path.join(aoi_dir, 'config/config.yaml')
config = read_config_yaml(config_file)

crop_w = config['crop_w']
crop_h = config['crop_h']
margin = config['margin']

def get_overlap(gt_rect, dt_rect, return_type): # rect = [xmin, ymin, xmax, ymax]
    xmin = max(min(gt_rect[0], gt_rect[2]), min(dt_rect[0], dt_rect[2]))
    ymin = max(min(gt_rect[1], gt_rect[3]), min(dt_rect[1], dt_rect[3]))
    xmax = min(max(gt_rect[0], gt_rect[2]), max(dt_rect[0], dt_rect[2]))
    ymax = min(max(gt_rect[1], gt_rect[3]), max(dt_rect[1], dt_rect[3]))
    if return_type =="area":
        # overlap area
        if xmin < xmax and ymin < ymax:
            return (xmax - xmin) * (ymax - ymin)
        # no overlap
        return 0
    elif return_type == "xyxy":
        # overlap rect
        if xmin < xmax and ymin < ymax:
            return xmin, ymin, xmax, ymax
        # no overlap
        return 0, 0, 0, 0


def check_partially_overlap(bbox, bbox_list, crop_rect):
    for other_bbox in bbox_list:
        if other_bbox != bbox:
            other_rect = [int(val) for val in other_bbox[2:6]]
            other_xmin, other_ymin, other_xmax, other_ymax = other_rect
            other_area = (other_xmax - other_xmin) * (other_ymax - other_ymin)

            overlap_area = get_overlap(crop_rect, other_rect, "area")
            if overlap_area == other_area or overlap_area == 0:
                continue
            else:
                return True
    return False

def get_bbox_in_crop_rect(bbox_list, crop_rect):
    """
    Convert labelme json format to csv format.

    Args:
        bbox_list (list[list]): annotations in [[file_name, error_type, xmin, ymin, xmax, ymax], ...] format.
        crop_rect (list): [xmin, ymin, xmax, ymax]
    Returns:
        bbox_in_crop_rect (list[list]): annotations in [[file_name, error_type, xmin, ymin, xmax, ymax], ...] format.
    """
    bbox_in_crop_rect = list()
    for bbox in bbox_list:
        bbox_rect = [int(val) for val in bbox[2:6]]
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox_rect
        bbox_area = (bbox_xmax - bbox_xmin) * (bbox_ymax - bbox_ymin)

        overlap_area = get_overlap(crop_rect, bbox_rect, "area")
        if overlap_area == bbox_area:
            bbox_in_crop_rect.append(bbox)
    return bbox_in_crop_rect

def get_bbox_in_and_across_crop_rect(bbox_list, crop_rect):
    """
    Convert labelme json format to csv format.

    Args:
        bbox_list (list[list]): annotations in [[file_name, error_type, xmin, ymin, xmax, ymax], ...] format.
        crop_rect (list): [xmin, ymin, xmax, ymax]
    Returns:
        bbox_in_crop_rect (list[list]): annotations in [[file_name, error_type, xmin, ymin, xmax, ymax], ...] format.
        bbox_across_crop_rect (list[list]): annotations in [[file_name, error_type, xmin, ymin, xmax, ymax], ...] format.
    """
    bbox_in_crop_rect = list()
    bbox_across_crop_rect = list()
    for bbox in bbox_list:
        bbox_rect = [int(val) for val in bbox[2:6]]
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox_rect
        bbox_area = (bbox_xmax - bbox_xmin) * (bbox_ymax - bbox_ymin)

        overlap_area = get_overlap(crop_rect, bbox_rect,"area")
        if overlap_area == bbox_area:
            bbox_in_crop_rect.append(bbox)
        elif overlap_area > 0 and overlap_area < bbox_area:
            bbox_across_crop_rect.append(bbox)
    return bbox_in_crop_rect, bbox_across_crop_rect

def filter_error_type_from_bbox_list(bbox_list, error_type_list):
    # bbox_list = [[file_name, error_type, xmin, ymin, xmax, ymax], ...]
    # error_type_list = ['error_type_1', 'error_type_2', ...]
    if len(bbox_list) == 0:
        return bbox_list
    bbox_list = np.array(bbox_list)

    for error_type in error_type_list:
        selected_rows = np.where(bbox_list==error_type)[0]
        if len(selected_rows) > 0:
            bbox_list = np.delete(bbox_list, selected_rows, axis=0)

    bbox_list = bbox_list.tolist()
    bbox_list = [ bbox[0:2]+[int(i) for i in bbox[2:]] for bbox in bbox_list ]
    return bbox_list

def convert_to_crop_image_coordinate(bbox_list, crop_rect, crop_image, name_suffix=0):
    crop_xmin, crop_ymin, crop_xmax, crop_ymax = crop_rect
    new_bbox_list = list()
    for idx, bbox in enumerate(bbox_list):
        file_name, error_type = bbox[0:2]
        file_id, ext = os.path.splitext(file_name)
        file_name = file_id + "_{}".format(name_suffix) + ext
        '''
        file_path = os.path.join(crop_image_dir, file_name)
        if idx == 0:
            cv2.imwrite(file_path, crop_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        '''
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox[2:6]
        bbox_xmin = bbox_xmin - crop_xmin
        bbox_ymin = bbox_ymin - crop_ymin
        bbox_xmax = bbox_xmax - crop_xmin
        bbox_ymax = bbox_ymax - crop_ymin
        new_bbox_list.append([file_name, error_type, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])
    return new_bbox_list

def convert_to_black_border_image_coordinate(bbox_list, crop_rect, delta_rect, black_border_image, name_suffix=0):
    crop_xmin, crop_ymin, crop_xmax, crop_ymax = crop_rect
    delta_xmin, delta_ymin, delta_xmax, delta_ymax = delta_rect
    new_bbox_list = list()
    for idx, bbox in enumerate(bbox_list):
        file_name, error_type = bbox[0:2]
        file_id, ext = os.path.splitext(file_name)
        file_name = file_id + "_{}".format(name_suffix) + ext
        '''
        file_path = os.path.join(crop_image_dir, file_name)
        if idx == 0:
            cv2.imwrite(file_path, black_border_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        '''
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox[2:6]
        bbox_xmin = bbox_xmin - crop_xmin + delta_xmin
        bbox_ymin = bbox_ymin - crop_ymin + delta_ymin
        bbox_xmax = bbox_xmax - crop_xmin + delta_xmin
        bbox_ymax = bbox_ymax - crop_ymin + delta_ymin
        new_bbox_list.append([file_name, error_type, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])
    return new_bbox_list

def invert_bbox_rect(bbox_rect):
    _xmin, _ymin, _xmax, _ymax = bbox_rect
    if _xmin < _xmax and _ymin < _ymax:
        return bbox_rect
    else:
        xmin = min(_xmin, _xmax)
        xmax = max(_xmin, _xmax)
        ymin = min(_ymin, _ymax)
        ymax = max(_ymin, _ymax)
        return [xmin, ymin, xmax, ymax]

def truncate_bbox_rect(bbox_rect, image_rect):
    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox_rect
    img_xmin, img_ymin, img_xmax, img_ymax = image_rect

    if bbox_xmin < img_xmin:
        bbox_xmin = img_xmin
    if bbox_ymin < img_ymin:
        bbox_ymin = img_ymin
    if bbox_xmax > img_xmax:
        bbox_xmax = img_xmax
    if bbox_ymax > img_ymax:
        bbox_ymax = img_ymax

    return [bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax]

num_solder_ball = 0
num_out_of_range = 0
def crop_small_image(bbox_list, image_file_path, crop_image_dir, crop_label_dir,
                     mask_bbox_list=[], eliminate_bbox_in_crop_rect=False, first_random_shift=False):
    global num_solder_ball, num_out_of_range
    image = cv2.imread(image_file_path)
    image_h, image_w, _ = image.shape
    image_rect = [0, 0, image_w, image_h]

    for mask_bbox in mask_bbox_list:
        xmin, ymin, xmax, ymax = mask_bbox[2:6]
        image[ymin:ymax, xmin:xmax] = [0,0,0]

    num = 1
    eliminate_bbox_list = []
    for bbox in bbox_list: # bbox_list = [[file_name, error_type, xmin, ymin, xmax, ymax], ...]
        if eliminate_bbox_in_crop_rect and bbox in eliminate_bbox_list:
            continue
        error_type = bbox[1]
        bbox_rect = invert_bbox_rect([int(val) for val in bbox[2:6]]) # [xmin, ymin, xmax, ymax]
        bbox_rect = truncate_bbox_rect(bbox_rect, image_rect)
        bbox[2:6] = bbox_rect

        xmin, ymin, xmax, ymax = bbox_rect

        xmid = (xmin + xmax)//2
        ymid = (ymin + ymax)//2
        logger.debug("error_type = {} ; xmin = {} ; ymin = {} ; xmax = {} ; ymax = {}".format(error_type, xmin, ymin, xmax, ymax))
        logger.debug("xmid = {} ; ymid = {}".format(xmid, ymid))

        if error_type == 'solder_ball':
            num_solder_ball = num_solder_ball + 1
            continue

        # Check if bbox_rect in image_rect to avoid mislabel case
        if get_overlap(bbox_rect, image_rect, "area") == 0:
            logger.info("{} : {} not in {}".format(os.path.basename(image_file_path), bbox_rect, image_rect))
            num_out_of_range = num_out_of_range + 1
            continue

        fixed_crop_xmin = xmid - crop_w//2
        fixed_crop_ymin = ymid - crop_h//2
        fixed_crop_xmax = fixed_crop_xmin + crop_w
        fixed_crop_ymax = fixed_crop_ymin + crop_h
        range_w = xmin - fixed_crop_xmin - 10 # TODO: Need to check if bbox touches the crop_rect
        range_h = ymin - fixed_crop_ymin - 10 # TODO: Need to check if bbox touches the crop_rect

        crop_xmin, crop_ymin, crop_xmax, crop_ymax = fixed_crop_xmin, fixed_crop_ymin, fixed_crop_xmax, fixed_crop_ymax
        crop_rect = [crop_xmin, crop_ymin, crop_xmax, crop_ymax]

        logger.debug("crop_rect 1 = {}".format(crop_rect))
        logger.debug("range_w = {} ; range_h = {}".format(range_w, range_h))

        random_shift_flag = first_random_shift
        # Check if other bbox partially in crop_rect
        while random_shift_flag or check_partially_overlap(bbox, bbox_list, crop_rect):
            if random_shift_flag:
                logger.debug("random shift crop_rect due to first_random_shift")
                random_shift_flag = False
            else:
                logger.debug("random shift crop_rect due to other bbox partially in crop_rect")
            random_shift_w = np.random.random_integers(0, range_w*2) - range_w
            random_shift_h = np.random.random_integers(0, range_h*2) - range_h
            crop_xmin = fixed_crop_xmin + random_shift_w
            crop_ymin = fixed_crop_ymin + random_shift_h
            crop_xmax = crop_xmin + crop_w
            crop_ymax = crop_ymin + crop_h
            crop_rect = [crop_xmin, crop_ymin, crop_xmax, crop_ymax]

        # Padding black border if crop range is out of image range
        if crop_xmin < 0:
            delta_xmin = np.absolute(crop_xmin)
            crop_xmin = 0
        else:
            delta_xmin = 0

        if crop_ymin < 0:
            delta_ymin = np.absolute(crop_ymin)
            crop_ymin = 0
        else:
            delta_ymin = 0

        if crop_xmax > image_w:
            delta_xmax = image_w - crop_xmin
            crop_xmax = image_w
        else:
            delta_xmax = crop_w

        if crop_ymax > image_h:
            delta_ymax = image_h - crop_ymin
            crop_ymax = image_h
        else:
            delta_ymax = crop_h

        if [delta_xmin, delta_ymin, delta_xmax, delta_ymax] != [0, 0, crop_w, crop_h]:
            crop_rect = [crop_xmin, crop_ymin, crop_xmax, crop_ymax]
            delta_rect = [delta_xmin, delta_ymin, delta_xmax, delta_ymax]

            bbox_in_crop_rect = get_bbox_in_crop_rect(bbox_list, crop_rect) # original image coordinate
            bbox_in_crop_rect = filter_error_type_from_bbox_list(bbox_in_crop_rect, ['solder_ball'])
            black_border_image = np.zeros([crop_h, crop_w, 3], dtype=np.uint8)
            black_border_image[delta_ymin:delta_ymax, delta_xmin:delta_xmax] = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            crop_image = black_border_image

            suffix = "{:03d}_{}_{}_{}_{}".format(num, xmin, ymin, xmax, ymax)
            bbox_crop_coord = convert_to_black_border_image_coordinate(bbox_in_crop_rect, crop_rect, delta_rect, black_border_image, name_suffix=suffix) # crop image coordinate
        else:
            bbox_in_crop_rect = get_bbox_in_crop_rect(bbox_list, crop_rect) # original image coordinate
            bbox_in_crop_rect = filter_error_type_from_bbox_list(bbox_in_crop_rect, ['solder_ball'])
            crop_image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

            suffix = "{:03d}_{}_{}_{}_{}".format(num, xmin, ymin, xmax, ymax)
            bbox_crop_coord = convert_to_crop_image_coordinate(bbox_in_crop_rect, crop_rect, crop_image, name_suffix=suffix) # crop image coordinate

        if eliminate_bbox_in_crop_rect:
            eliminate_bbox_list.extend(bbox_in_crop_rect)
        logger.debug("crop_rect = {}".format(crop_rect))
        logger.debug("bbox_in_crop_rect = {}".format(bbox_in_crop_rect))
        logger.debug("bbox_crop_coord = {}\n".format(bbox_crop_coord))

        assert len(bbox_crop_coord) > 0, 'len(bbox_crop_coord) must be greater than 0'

        crop_image_file_name = bbox_crop_coord[0][0]
        crop_image_file_path = os.path.join(crop_image_dir, crop_image_file_name)
        cv2.imwrite(crop_image_file_path, crop_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        csv_to_json(bbox_crop_coord, crop_image_dir, crop_label_dir, coord_type="xmin_ymin_xmax_ymax")
        num = num + 1

def crop_small_image_w_aug(bbox_list, image_file_path, crop_image_dir, crop_label_dir, aug_num_dict, mask_bbox_list=[],
                           enhance_bbox_list=[], enhance_aug_num=0, first_random_shift=False):
    global num_solder_ball, num_out_of_range
    image = cv2.imread(image_file_path)
    image_h, image_w, _ = image.shape
    image_rect = [0, 0, image_w, image_h]

    for mask_bbox in mask_bbox_list:
        xmin, ymin, xmax, ymax = mask_bbox[2:6]
        image[ymin:ymax, xmin:xmax] = [0,0,0]

    for bbox in bbox_list: # bbox_list = [[file_name, error_type, xmin, ymin, xmax, ymax], ...]
        num = 1
        file_name, error_type = bbox[0:2]
        bbox_rect = invert_bbox_rect([int(val) for val in bbox[2:6]]) # [xmin, ymin, xmax, ymax]
        bbox_rect = truncate_bbox_rect(bbox_rect, image_rect)
        bbox[2:6] = bbox_rect

        xmin, ymin, xmax, ymax = bbox_rect

        xmid = (xmin + xmax)//2
        ymid = (ymin + ymax)//2
        logger.debug("error_type = {} ; xmin = {} ; ymin = {} ; xmax = {} ; ymax = {}".format(error_type, xmin, ymin, xmax, ymax))
        logger.debug("xmid = {} ; ymid = {}".format(xmid, ymid))

        if error_type == 'solder_ball':
            num_solder_ball = num_solder_ball + 1
            continue

        # Check if bbox_rect in image_rect to avoid mislabel case
        if get_overlap(bbox_rect, image_rect,"area") == 0:
            logger.info("{} : {} not in {}".format(os.path.basename(image_file_path), bbox_rect, image_rect))
            num_out_of_range = num_out_of_range + 1
            continue

        fixed_crop_xmin = xmid - crop_w//2
        fixed_crop_ymin = ymid - crop_h//2
        fixed_crop_xmax = fixed_crop_xmin + crop_w
        fixed_crop_ymax = fixed_crop_ymin + crop_h
        range_w = xmin - fixed_crop_xmin - 10 # TODO: Need to check if bbox touches the crop_rect
        range_h = ymin - fixed_crop_ymin - 10 # TODO: Need to check if bbox touches the crop_rect

        crop_xmin, crop_ymin, crop_xmax, crop_ymax = fixed_crop_xmin, fixed_crop_ymin, fixed_crop_xmax, fixed_crop_ymax
        crop_rect = [crop_xmin, crop_ymin, crop_xmax, crop_ymax]

        logger.debug("crop_rect 1 = {}".format(crop_rect))
        logger.debug("range_w = {} ; range_h = {}".format(range_w, range_h))

        random_shift_flag = first_random_shift
        if bbox in enhance_bbox_list:
            total_aug_num = aug_num_dict.get(error_type, 1) + enhance_aug_num
        else:
            total_aug_num = aug_num_dict.get(error_type, 1)

        for aug_num in range(total_aug_num):
            # Check if other bbox partially in crop_rect
            random_shift_flag = random_shift_flag or aug_num > 0
            while random_shift_flag or check_partially_overlap(bbox, bbox_list, crop_rect):
                if random_shift_flag:
                    logger.debug("random shift crop_rect due to first_random_shift or aug_num > 0")
                    random_shift_flag = False
                else:
                    logger.debug("random shift crop_rect due to other bbox partially in crop_rect")
                random_shift_w = np.random.random_integers(0, range_w*2) - range_w
                random_shift_h = np.random.random_integers(0, range_h*2) - range_h
                crop_xmin = fixed_crop_xmin + random_shift_w
                crop_ymin = fixed_crop_ymin + random_shift_h
                crop_xmax = crop_xmin + crop_w
                crop_ymax = crop_ymin + crop_h
                crop_rect = [crop_xmin, crop_ymin, crop_xmax, crop_ymax]

            # Padding black border if crop range is out of image range
            if crop_xmin < 0:
                delta_xmin = np.absolute(crop_xmin)
                crop_xmin = 0
            else:
                delta_xmin = 0

            if crop_ymin < 0:
                delta_ymin = np.absolute(crop_ymin)
                crop_ymin = 0
            else:
                delta_ymin = 0

            if crop_xmax > image_w:
                delta_xmax = image_w - crop_xmin
                crop_xmax = image_w
            else:
                delta_xmax = crop_w

            if crop_ymax > image_h:
                delta_ymax = image_h - crop_ymin
                crop_ymax = image_h
            else:
                delta_ymax = crop_h

            if [delta_xmin, delta_ymin, delta_xmax, delta_ymax] != [0, 0, crop_w, crop_h]:
                crop_rect = [crop_xmin, crop_ymin, crop_xmax, crop_ymax]
                delta_rect = [delta_xmin, delta_ymin, delta_xmax, delta_ymax]

                bbox_in_crop_rect = get_bbox_in_crop_rect(bbox_list, crop_rect) # original image coordinate
                bbox_in_crop_rect = filter_error_type_from_bbox_list(bbox_in_crop_rect, ['solder_ball'])
                black_border_image = np.zeros([crop_h, crop_w, 3], dtype=np.uint8)
                black_border_image[delta_ymin:delta_ymax, delta_xmin:delta_xmax] = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
                crop_image = black_border_image

                suffix = "{:03d}_{}_{}_{}_{}".format(num, xmin, ymin, xmax, ymax)
                bbox_crop_coord = convert_to_black_border_image_coordinate(bbox_in_crop_rect, crop_rect, delta_rect, black_border_image, name_suffix=suffix) # crop image coordinate
            else:
                bbox_in_crop_rect = get_bbox_in_crop_rect(bbox_list, crop_rect) # original image coordinate
                bbox_in_crop_rect = filter_error_type_from_bbox_list(bbox_in_crop_rect, ['solder_ball'])
                crop_image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

                suffix = "{:03d}_{}_{}_{}_{}".format(num, xmin, ymin, xmax, ymax)
                bbox_crop_coord = convert_to_crop_image_coordinate(bbox_in_crop_rect, crop_rect, crop_image, name_suffix=suffix) # crop image coordinate

            logger.debug("crop_rect = {}".format(crop_rect))
            logger.debug("bbox_in_crop_rect = {}".format(bbox_in_crop_rect))
            logger.debug("bbox_crop_coord = {}\n".format(bbox_crop_coord))

            assert len(bbox_crop_coord) > 0, 'len(bbox_crop_coord) must be greater than 0'

            crop_image_file_name = bbox_crop_coord[0][0]
            crop_image_file_path = os.path.join(crop_image_dir, crop_image_file_name)
            cv2.imwrite(crop_image_file_path, crop_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            csv_to_json(bbox_crop_coord, crop_image_dir, crop_label_dir, coord_type="xmin_ymin_xmax_ymax")
            num = num + 1

def crop_retrain_data(set_type, aug_num_dict=None):
    retrain_data_org_pre_dir = config['retrain_data_org_pre_dir']
    fn_image_dir = os.path.join(retrain_data_org_pre_dir, 'images')
    fn_label_dir = os.path.join(retrain_data_org_pre_dir, 'labels')
    inference_result_label_dir = os.path.join(retrain_data_org_pre_dir, 'inference_result/result_labels')

    if set_type == 'train':
        retrain_data_train_dir = config['retrain_data_train_dir']
        crop_image_dir = os.path.join(retrain_data_train_dir, 'images_random_crop_w_aug') # .jpg
        crop_label_dir = os.path.join(retrain_data_train_dir, 'labels_random_crop_w_aug') # .json
    elif set_type == 'val':
        retrain_data_val_dir = config['retrain_data_val_dir']
        crop_image_dir = os.path.join(retrain_data_val_dir, 'images_random_crop') # .jpg
        crop_label_dir = os.path.join(retrain_data_val_dir, 'labels_random_crop') # .json
    else:
        logger.error('retrain data dir: {} is ilegal.'.format(set_type))

    os_makedirs(crop_image_dir)
    os_makedirs(crop_label_dir)

    for idx, json_file_name in enumerate(os.listdir(fn_label_dir)):
        logger.info('{:>4d}, {}'.format(idx+1, json_file_name))
        fn_json_file_path = os.path.join(fn_label_dir, json_file_name)
        infer_json_file_path = os.path.join(inference_result_label_dir, json_file_name)

        # [[file_name, error_type, xmin, ymin, xmax, ymax], ...]
        fn_bbox_list = json_to_bbox(fn_json_file_path) 
        if  os.path.isfile(infer_json_file_path):
            infer_bbox_list = json_to_bbox(infer_json_file_path) 
        else:
            infer_bbox_list = list()

        for fn_bbox in fn_bbox_list:
            for infer_bbox in list(infer_bbox_list):
                if get_overlap(fn_bbox[2:6], infer_bbox[2:6], "area") > 0:
                    infer_bbox_list.remove(infer_bbox)

        image_file_name = os.path.splitext(json_file_name)[0] + '.jpg'
        image_file_path = os.path.join(fn_image_dir, image_file_name)
        if aug_num_dict == None:
            crop_small_image(fn_bbox_list, image_file_path, crop_image_dir, crop_label_dir,
                             mask_bbox_list=infer_bbox_list, eliminate_bbox_in_crop_rect=True,
                             first_random_shift=False)
        else:
            crop_small_image_w_aug(fn_bbox_list, image_file_path, crop_image_dir, crop_label_dir, aug_num_dict,
                                   mask_bbox_list=infer_bbox_list, first_random_shift=False)

def random_crop_and_aug_train_data(aug_num_dict, train_data_dir):
    '''
    image_dir = os.path.join(train_data_dir, 'images') # .jpg
    label_dir = os.path.join(train_data_dir, 'labels') # .json
    '''
    image_dir = os.path.join(train_data_dir, 'images_wo_border') # .jpg
    label_dir = os.path.join(train_data_dir, 'labels_wo_border') # .json
    enhance_label_dir = os.path.join(train_data_dir, 'labels_wo_border_micro_bridge') # .json

    # Crop image with foreground object (random shift if bbox overlaps the border)
    crop_image_dir = os.path.join(train_data_dir, 'images_random_crop_w_aug') # .jpg
    crop_label_dir = os.path.join(train_data_dir, 'labels_random_crop_w_aug') # .json

    os_makedirs(crop_image_dir)
    os_makedirs(crop_label_dir)

    for idx, json_file_name in enumerate(os.listdir(label_dir)):
        logger.info('{:>4d}, {}'.format(idx+1, json_file_name))
        json_file_path = os.path.join(label_dir, json_file_name) # json file path
        bbox_list = json_to_bbox(json_file_path) # [[file_name, error_type, xmin, ymin, xmax, ymax], ...]

        if os.path.isdir(enhance_label_dir) and json_file_name in os.listdir(enhance_label_dir):
            enhance_json_file_path = os.path.join(enhance_label_dir, json_file_name)
            enhance_bbox_list = json_to_bbox(enhance_json_file_path) # [[file_name, error_type, xmin, ymin, xmax, ymax], ...]
            for enhance_bbox in enhance_bbox_list:
                assert enhance_bbox in bbox_list, "enhance_bbox {} not in bbox_list".format(enhance_bbox)
            enhance_aug_num = 20
        else:
            enhance_bbox_list = list()
            enhance_aug_num = 0

        if len(bbox_list) > 0:
            image_file_name = os.path.splitext(json_file_name)[0] + '.jpg'
            image_file_path = os.path.join(image_dir, image_file_name)
            crop_small_image_w_aug(bbox_list, image_file_path, crop_image_dir, crop_label_dir, aug_num_dict,
                                   enhance_bbox_list=enhance_bbox_list, enhance_aug_num=enhance_aug_num, first_random_shift=True)

    logger.info("num_solder_ball = {}, num_out_of_range = {}".format(num_solder_ball, num_out_of_range))

def random_crop_test_data(test_data_dir):
    image_dir = os.path.join(test_data_dir, 'images') # .jpg
    label_dir = os.path.join(test_data_dir, 'labels') # .json

    # Crop image with foreground object (random shift if bbox overlaps the border)
    crop_image_dir = os.path.join(test_data_dir, 'images_random_crop') # .jpg
    crop_label_dir = os.path.join(test_data_dir, 'labels_random_crop') # .json

    os_makedirs(crop_image_dir)
    os_makedirs(crop_label_dir)

    for idx, json_file_name in enumerate(os.listdir(label_dir)):
        logger.info('{:>4d}, {}'.format(idx+1, json_file_name))
        json_file_path = os.path.join(label_dir, json_file_name) # json file path
        bbox_list = json_to_bbox(json_file_path) # [[file_name, error_type, xmin, ymin, xmax, ymax], ...]

        if len(bbox_list) > 0:
            image_file_name = os.path.splitext(json_file_name)[0] + '.jpg'
            image_file_path = os.path.join(image_dir, image_file_name)
            crop_small_image(bbox_list, image_file_path, crop_image_dir, crop_label_dir, \
                             eliminate_bbox_in_crop_rect=True, first_random_shift=False)

    logger.debug("num_solder_ball = {}, num_out_of_range = {}".format(num_solder_ball, num_out_of_range))

def crop_sliding_window(image, crop_h=crop_h, crop_w=crop_w, margin=margin):
    
    crop_rect_list = list()
    crop_image_list = list()

    image_h, image_w, _ = image.shape

    # Check image size is equal to crop size
    if image_h==crop_h and image_w==crop_w:
        crop_rect_list = [[0, 0, image_w, image_h]]
        crop_image_list = [image]
        return crop_rect_list, crop_image_list


    count_h = image_h // (crop_h - margin) + 1
    count_w = image_w // (crop_w - margin) + 1

    ymax_1 = (count_h-1)*(crop_h - margin) + crop_h
    ymax_2 = (count_h-2)*(crop_h - margin) + crop_h
    if (ymax_1 >= image_h and ymax_2 >= image_h):
        count_h = count_h - 1

    xmax_1 = (count_w-1)*(crop_w - margin) + crop_w
    xmax_2 = (count_w-2)*(crop_w - margin) + crop_w
    if (xmax_1 >= image_w and xmax_2 >= image_w):
        count_w = count_w - 1

    for h in range(0, count_h):
        for w in range(0, count_w):
            crop_ymin = (crop_h - margin) * h
            crop_xmin = (crop_w - margin) * w

            crop_ymax = crop_ymin + crop_h
            crop_xmax = crop_xmin + crop_w

            if crop_ymax > image_h:
                crop_ymax = image_h
                crop_ymin = image_h - crop_h
            if crop_xmax > image_w:
                crop_xmax = image_w
                crop_xmin = image_w - crop_w

            # Store rect and image
            crop_rect = [crop_xmin, crop_ymin, crop_xmax, crop_ymax]
            crop_rect_list.append(crop_rect)

            crop_image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            crop_image_list.append(crop_image)
    return crop_rect_list, crop_image_list

def convert_bbox_list_to_coordinate_list(bbox_list, crop_rect):
    """
    Convert bbox_list to coordinate_list.

    Args:
        bbox_list (list[list]): annotations in [[file_name, error_type, xmin, ymin, xmax, ymax], ...] format.
    Returns:
        coordinate_list (list[list]): annotations in [[xmin, ymin, xmax, ymax], ...] format.
    """
    crop_xmin, crop_ymin, crop_xmax, crop_ymax = crop_rect
    coordinate_list = list()
    for bbox in bbox_list:
        xmin, ymin, xmax, ymax = [int(val) for val in bbox[2:6]]
        bbox_rect = [xmin-crop_xmin, ymin-crop_ymin, xmax-crop_xmin, ymax-crop_ymin]
        coordinate_list.append(bbox_rect)
    return coordinate_list

def mask_fg_across_crop_image_margin(crop_image, margin_bbox_list):
    image = crop_image.copy()
    for margin_bbox in margin_bbox_list:
        xmin, ymin, xmax, ymax = margin_bbox
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax > crop_w:
            xmax = crop_w
        if ymax > crop_h:
            ymax = crop_h
        image[ymin:ymax, xmin:xmax] = [0,0,0]
    return image

def crop_and_save_sliding_window(input_dict):
    image_wo_border_dir = input_dict['image_wo_border_dir']
    label_wo_border_dir = input_dict['label_wo_border_dir']
    image_sliding_crop_dir = input_dict['image_sliding_crop_dir']
    label_sliding_crop_dir = input_dict['label_sliding_crop_dir']
    mask_fg_across_margin = input_dict['mask_fg_across_margin']

    os_makedirs(image_sliding_crop_dir)
    os_makedirs(label_sliding_crop_dir)

    for idx, image_file_name in enumerate(os.listdir(image_wo_border_dir)):
        image_file_path = os.path.join(image_wo_border_dir, image_file_name)
        image = cv2.imread(image_file_path)
        image_h, image_w, _ = image.shape

        json_file_name = os.path.splitext(image_file_name)[0] + '.json'
        json_file_path = os.path.join(label_wo_border_dir, json_file_name)
        bbox_list = json_to_bbox(json_file_path)

        image_file_name_wo_ext, ext = os.path.splitext(image_file_name)

        count_h = image_h // (crop_h - margin) + 1
        count_w = image_w // (crop_w - margin) + 1

        tag = ""
        ymax_1 = (count_h-1)*(crop_h - margin) + crop_h
        ymax_2 = (count_h-2)*(crop_h - margin) + crop_h
        if (ymax_1 >= image_h and ymax_2 >= image_h):
            count_h = count_h - 1
            tag = tag + 'h'
        else:
            tag = tag + ' '

        xmax_1 = (count_w-1)*(crop_w - margin) + crop_w
        xmax_2 = (count_w-2)*(crop_w - margin) + crop_w
        if (xmax_1 >= image_w and xmax_2 >= image_w):
            count_w = count_w - 1
            tag = tag + 'w'
        else:
            tag = tag + ' '

        logger.info('{:>4d}, {}, {:>4d}, {}'.format(idx+1, tag, count_h * count_w, image_file_name))
        num = 0
        for h in range(0, count_h):
            for w in range(0, count_w):
                num += 1
                crop_ymin = (crop_h - margin) * h
                crop_xmin = (crop_w - margin) * w

                crop_ymax = crop_ymin + crop_h
                crop_xmax = crop_xmin + crop_w

                if crop_ymax > image_h:
                    crop_ymax = image_h
                    crop_ymin = image_h - crop_h
                if crop_xmax > image_w:
                    crop_xmax = image_w
                    crop_xmin = image_w - crop_w

                # Save image
                crop_rect = [crop_xmin, crop_ymin, crop_xmax, crop_ymax]
                crop_image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

                # Get bbox_in_crop_rect and bbox_across_crop_rect on original image coordinate
                bbox_in_crop_rect, bbox_across_crop_rect = get_bbox_in_and_across_crop_rect(bbox_list, crop_rect)
                # Filter out some anomaly
                bbox_in_crop_rect = filter_error_type_from_bbox_list(bbox_in_crop_rect, ['solder_ball'])

                if mask_fg_across_margin:
                    if len(bbox_in_crop_rect) > 0:
                        suffix = "{:03d}_fg_".format(num)
                    else:
                        suffix = "{:03d}_bg_".format(num)
                    # margin_bbox_list is on crop image coordinate
                    margin_bbox_list = convert_bbox_list_to_coordinate_list(bbox_across_crop_rect, crop_rect)
                    if len(margin_bbox_list) > 0:
                        crop_image = mask_fg_across_crop_image_margin(crop_image, margin_bbox_list)
                        suffix = suffix +"mask_margin_{}_{}_{}_{}".format(*crop_rect)
                    else:
                        suffix = suffix +"{}_{}_{}_{}".format(*crop_rect)
                else:
                    suffix = "{:03d}_{}_{}_{}_{}".format(num, *crop_rect)

                crop_image_file_name = "{}_{}{}".format(image_file_name_wo_ext, suffix, ext)
                crop_image_file_path = os.path.join(image_sliding_crop_dir, crop_image_file_name)
                cv2.imwrite(crop_image_file_path, crop_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                # Save label
                # bbox_crop_coord is on crop image coordinate
                bbox_crop_coord = convert_to_crop_image_coordinate(bbox_in_crop_rect, crop_rect, crop_image, name_suffix=suffix)

                if len(bbox_crop_coord) > 0:
                    csv_to_json(bbox_crop_coord, image_sliding_crop_dir, label_sliding_crop_dir, coord_type="xmin_ymin_xmax_ymax")



if __name__ == '__main__':
    # crop_train_data
    aug_num_dict = {'bridge': 2,
                    'empty': 2,
                    'appearance_less': 2,
                    'appearance_hole': 2,
                    'excess_solder': 10,
                    'appearance': 100}
    train_data_dir = config['train_data_dir']
    random_crop_and_aug_train_data(aug_num_dict, train_data_dir)

    # crop_test_data
    test_data_dir = config['test_data_dir']
    random_crop_test_data(test_data_dir)


    # windows sliding crop
    train_data_dict = {
        'image_wo_border_dir': os.path.join(train_data_dir, 'images_wo_border'),
        'label_wo_border_dir': os.path.join(train_data_dir, 'labels_wo_border'),
        'image_sliding_crop_dir': os.path.join(train_data_dir, 'images_wo_border_sliding_crop'),
        'label_sliding_crop_dir': os.path.join(train_data_dir, 'labels_wo_border_sliding_crop'),
        'mask_fg_across_margin': False
    }

    test_data_dict = {
        'image_wo_border_dir': os.path.join(test_data_dir, 'images_wo_border'),
        'label_wo_border_dir': os.path.join(test_data_dir, 'labels_wo_border'),
        'image_sliding_crop_dir': os.path.join(test_data_dir, 'images_wo_border_sliding_crop'),
        'label_sliding_crop_dir': os.path.join(test_data_dir, 'labels_wo_border_sliding_crop'),
        'mask_fg_across_margin': False
    }

    train_data_mask_fg_across_margin_dict = {
        'image_wo_border_dir': os.path.join(train_data_dir, 'images_wo_border'),
        'label_wo_border_dir': os.path.join(train_data_dir, 'labels_wo_border'),
        'image_sliding_crop_dir': os.path.join(train_data_dir, 'images_wo_border_mask_margin_sliding_crop'),
        'label_sliding_crop_dir': os.path.join(train_data_dir, 'labels_wo_border_mask_margin_sliding_crop'),
        'mask_fg_across_margin': True
    }

    crop_and_save_sliding_window(train_data_dict)
    crop_and_save_sliding_window(test_data_dict)
    crop_and_save_sliding_window(train_data_mask_fg_across_margin_dict)