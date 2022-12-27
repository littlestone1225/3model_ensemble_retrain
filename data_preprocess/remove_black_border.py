#!/usr/bin/env python3
import os, sys
import cv2
import numpy as np
from operator import add
import logging

aoi_dir_name = os.getenv('AOI_DIR_NAME')
assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"

current_dir = os.path.dirname(os.path.abspath(__file__))
idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
aoi_dir = current_dir[:idx]

from csv_json_conversion import csv_to_json, json_to_bbox
from crop_small_image import get_overlap

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


def mask_bbox_in_image(bbox_list, image):
    image_h, image_w, _ = image.shape
    image_rect = [0, 0, image_w-1, image_h-1]

    for bbox in bbox_list:
        xmin, ymin, xmax, ymax = [int(val) for val in bbox[2:6]]
        bbox_rect = [xmin, ymin, xmax, ymax]

        if get_overlap(bbox_rect, image_rect, "area") == (xmax-xmin)*(ymax-ymin):
            image[ymin:ymax, xmin:xmax] = [0,0,0]
    return image

def sort_rect_by_area(rect_list):
    area_list = list()
    for rect in rect_list:
        xmin, ymin, xmax, ymax = rect
        area = (xmax - xmin)*(ymax - ymin)
        area_list.append(area)
    area_array = np.array(area_list)
    sort_idx = np.argsort(area_array)[::-1]
    new_rect_array = np.array(rect_list)[sort_idx]
    return new_rect_array

def sort_rect_by_white_point(rect_list, maskBGR):
    white_point_list = list()
    for rect in rect_list:
        xmin, ymin, xmax, ymax = rect
        mask = maskBGR[ymin:ymax, xmin:xmax]
        white_point_sum = np.sum(mask>0)
        white_point_list.append(white_point_sum)
    white_point_array = np.array(white_point_list)
    sort_idx = np.argsort(white_point_array)[::-1]
    new_white_point_array = np.array(rect_list)[sort_idx]
    return new_white_point_array


def remove_black_border_ori(imgRaw, shrink_image_wo_border_path=None):

    shrink_margin = 50
    margin = 200
    rawH, rawW, rawCH = imgRaw.shape

    defaultWidth = 640.0
    shrinkFactor = defaultWidth/rawW
    imgShrink = cv2.resize(imgRaw, (0,0), fx=shrinkFactor, fy=shrinkFactor)
    shrinkH, shrinkW, shrinkCH = imgShrink.shape

    H, W = [x[0]+x[1] for x in zip(imgShrink.shape[:2], [shrink_margin*2, shrink_margin*2])]
    black_img = np.zeros([H,W,3], dtype=np.uint8)
    black_img[shrink_margin:shrink_margin+shrinkH, shrink_margin:shrink_margin+shrinkW] = imgShrink
    imgShrink = black_img

    imgLab = cv2.cvtColor(imgShrink, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(imgLab)

    ret, thresh = cv2.threshold(L, 10, 255, cv2.THRESH_BINARY)

    maskBGR = cv2.inRange(imgLab,(0,124,124),(100,132,132))
    maskBGR = cv2.bitwise_not(maskBGR)

    kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))

    low_sigma = cv2.GaussianBlur(thresh,(3,3),0)
    high_sigma = cv2.GaussianBlur(thresh,(5,5),0)

    # Calculate the DoG by subtracting
    dog = low_sigma - high_sigma

    # Find the contours
    contours, hierarchy = cv2.findContours(dog, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bbox_list = list()
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        if (w<=30 and h<=30) or w/h > 10 or h/w>10:
            continue
        xmin, ymin, xmax, ymax = x, y, x+w, y+h
        bbox_list.append([xmin, ymin, xmax, ymax])

    if len(bbox_list)==0:
        return 0, 0, 0, 0

    rect_array = sort_rect_by_white_point(bbox_list, maskBGR)

    best_ratio = 0.6
    best_idx = 0
    best_sum = 0
    for idx, rect in enumerate(rect_array):
        xmin, ymin, xmax, ymax = rect
        mask = maskBGR[ymin:ymax, xmin:xmax]
        sum = np.sum(mask>0)
        sum_all = np.sum(mask>=0)
        ratio = sum/sum_all
        if idx ==0:
            first_sum = sum
            first_sum_all = sum_all
            first_ratio = ratio
            first_idx = idx
            logger.debug('case 0 => {}: first_sum = {} ; first_sum_all = {} ; first_ratio = {}'.
                         format(idx, first_sum, first_sum_all, first_ratio))
            if ratio > best_ratio:
                best_sum = sum
                best_sum_all = sum_all
                best_ratio = ratio
                best_idx = idx
                logger.debug('case 1 => {}: best_sum = {} ; best_sum_all = {} ; best_ratio = {}'.
                             format(idx, best_sum, best_sum_all, best_ratio))
        else:
            if best_sum == 0:
                if ratio > first_ratio and (first_sum - sum)/first_sum < 0.1:
                    best_sum = sum
                    best_sum_all = sum_all
                    best_ratio = ratio
                    best_idx = idx
                    logger.debug('case 2 => {}: best_sum = {} ; best_sum_all = {} ; best_ratio = {}'.
                                 format(idx, best_sum, best_sum_all, best_ratio))
                else:
                    logger.debug('case 4 => {}: sum = {} ; sum_all = {} ; ratio = {}'.
                                 format(idx, sum, sum_all, ratio))
            else:
                overlap_xmin, overlap_ymin, overlap_xmax, overlap_ymax = get_overlap(rect_array[best_idx], rect_array[idx],"xyxy")
                overlap_mask = maskBGR[overlap_ymin:overlap_ymax, overlap_xmin:overlap_xmax]
                overlap_sum = np.sum(overlap_mask>0)
                overlap_sum_all = np.sum(overlap_mask>=0)

                if (best_sum_all - overlap_sum_all)/best_sum_all < 0.3:
                    logger.debug('case 5 => {}: sum = {} ; sum_all = {} ; ratio = {}'.
                                 format(idx, sum, sum_all, ratio))
                elif ratio > best_ratio and (best_sum - sum)/best_sum < 0.1:
                    best_sum = sum
                    best_sum_all = sum_all
                    best_ratio = ratio
                    best_idx = idx
                    logger.debug('case 3 => {}: best_sum = {} ; best_sum_all = {} ; best_ratio = {}'.
                                 format(idx, best_sum, best_sum_all, best_ratio))
                else:
                    logger.debug('case 6 => {}: sum = {} ; sum_all = {} ; ratio = {}'.
                                 format(idx, sum, sum_all, ratio))

    logger.debug("best_idx = {}".format(best_idx))
    xmin, ymin, xmax, ymax = rect_array[best_idx]

    xmin = xmin - shrink_margin
    ymin = ymin - shrink_margin
    xmax = xmax - shrink_margin
    ymax = ymax - shrink_margin

    xmin = int(xmin/shrinkFactor) - margin
    ymin = int(ymin/shrinkFactor) - margin
    xmax = int(xmax/shrinkFactor) + margin
    ymax = int(ymax/shrinkFactor) + margin

    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax >= rawW:
        xmax = rawW - 1
    if ymax >= rawH:
        ymax = rawH - 1

    if shrink_image_wo_border_path != None:
        imgOrg = imgRaw.copy()
        cv2.rectangle(imgOrg, (xmin, ymin), (xmax, ymax), (0, 0, 255), 10)
        xmin_2 = xmin - 100
        ymin_2 = ymin - 100
        xmax_2 = xmax + 100
        ymax_2 = ymax + 100
        if xmin_2 < 0:
            xmin_2 = 1
        if ymin_2 < 0:
            ymin_2 = 1
        if xmax_2 > rawW:
            xmax_2 = rawW-1
        if ymax_2 > rawH:
            ymax_2 = rawH-1
        shrink_image_wo_border = cv2.resize(imgOrg[ymin_2:ymax_2, xmin_2:xmax_2], (0,0), fx=0.6, fy=0.6)
        cv2.imwrite(shrink_image_wo_border_path, shrink_image_wo_border, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    return xmin, ymin, xmax, ymax

def remove_border(imgRaw, shrink_image_wo_border_path=None):
    save_step_image = False
    shrink_margin = 0
    margin = 200
    rawH, rawW, rawCH = imgRaw.shape

    # why is width decided?
    defaultWidth = 640.0
    shrinkFactor = defaultWidth/rawW
    imgShrink = cv2.resize(imgRaw, (0,0), fx=shrinkFactor, fy=shrinkFactor)
    shrinkH, shrinkW, shrinkCH = imgShrink.shape

    H, W = [x[0]+x[1] for x in zip(imgShrink.shape[:2], [shrink_margin*2, shrink_margin*2])]
    black_img = np.zeros([H,W,3], dtype=np.uint8)
    black_img[shrink_margin:shrink_margin+shrinkH, shrink_margin:shrink_margin+shrinkW] = imgShrink
    imgShrink = black_img

    imgLab = cv2.cvtColor(imgShrink, cv2.COLOR_BGR2Lab)

    
    
    # filter the black pixel which is not in pcb area
    maskBGR_black = cv2.inRange(imgLab,(0,124,124),(255,132,132)) #################?
    if save_step_image: cv2.imwrite("0_1black.png",maskBGR_black)
    maskBGR_iron = cv2.inRange(imgLab,(100,118,118),(150,138,138)) ################?
    if save_step_image: cv2.imwrite("0_2iron.png",maskBGR_iron)
    maskBGR = cv2.bitwise_or(maskBGR_black,maskBGR_iron)
    if save_step_image: cv2.imwrite("0_3union.png",maskBGR)
    maskBGR = cv2.bitwise_not(maskBGR)
    if save_step_image: cv2.imwrite("0_4last.png",maskBGR)
    

    
    L, a, b = cv2.split(imgLab)
    if save_step_image: cv2.imwrite("0_00L.png",L)
    ret, thresh = cv2.threshold(L, 100, 255, cv2.THRESH_TOZERO_INV)###############?
    ret, thresh = cv2.threshold(L, 10, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_and(thresh,maskBGR)
    if save_step_image: cv2.imwrite("0_0thres.png",thresh)


    low_sigma = cv2.GaussianBlur(thresh,(3,3),0)
    high_sigma = cv2.GaussianBlur(thresh,(5,5),0)

    # Calculate the DoG by subtracting
    dog = low_sigma - high_sigma
    if save_step_image: cv2.imwrite("0_5DoG.png",dog)

    # Find the contours
    contours, hierarchy = cv2.findContours(dog, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bbox_list = list()
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(imgShrink, (x, y), (x+w, y+h), (0,255,0), 3, cv2.LINE_AA)
        
        if (w<=30 and h<=30) or w/h > 10 or h/w>10:
            continue
        xmin, ymin, xmax, ymax = x, y, x+w, y+h
        bbox_list.append([xmin, ymin, xmax, ymax])
        
        cv2.rectangle(imgShrink, (xmin, ymin), (xmax, ymax), (0,0,255), 3, cv2.LINE_AA)
        
    if save_step_image: cv2.imwrite("0_yrect.png",imgShrink)

    if len(bbox_list)==0:
        return 0, 0, 0, 0

    rect_array = sort_rect_by_white_point(bbox_list, maskBGR)
    best_ratio = 0.6
    best_idx = 0
    best_sum = 0
    for idx, rect in enumerate(rect_array):
        xmin, ymin, xmax, ymax = rect
        mask = maskBGR[ymin:ymax, xmin:xmax]
        sum = np.sum(mask>0)
        sum_all = np.sum(mask>=0)
        ratio = sum/sum_all

        if idx ==0:
            first_sum = sum
            first_sum_all = sum_all
            first_ratio = ratio
            logger.debug('case 0 => {}: first_sum = {} ; first_sum_all = {} ; first_ratio = {}'.
                         format(idx, first_sum, first_sum_all, first_ratio))
            if ratio > best_ratio:
                best_sum = sum
                best_sum_all = sum_all
                best_ratio = ratio
                best_idx = idx
                logger.debug('case 1 => {}: best_sum = {} ; best_sum_all = {} ; best_ratio = {}'.
                             format(idx, best_sum, best_sum_all, best_ratio))
        else:
            if best_sum == 0:
                if ratio > first_ratio and (first_sum - sum)/first_sum < 0.1:
                    best_sum = sum
                    best_sum_all = sum_all
                    best_ratio = ratio
                    best_idx = idx
                    logger.debug('case 2 => {}: best_sum = {} ; best_sum_all = {} ; best_ratio = {}'.
                                 format(idx, best_sum, best_sum_all, best_ratio))
                else:
                    logger.debug('case 4 => {}: sum = {} ; sum_all = {} ; ratio = {}'.
                                 format(idx, sum, sum_all, ratio))
            else:
                overlap_xmin, overlap_ymin, overlap_xmax, overlap_ymax = get_overlap(rect_array[best_idx], rect_array[idx], "xyxy")
                overlap_mask = maskBGR[overlap_ymin:overlap_ymax, overlap_xmin:overlap_xmax]
                overlap_sum_all = np.sum(overlap_mask>=0)

                if (best_sum_all - overlap_sum_all)/best_sum_all < 0.3:
                    logger.debug('case 5 => {}: sum = {} ; sum_all = {} ; ratio = {}'.
                                 format(idx, sum, sum_all, ratio))
                elif ratio > best_ratio and (best_sum - sum)/best_sum < 0.1:
                    best_sum = sum
                    best_sum_all = sum_all
                    best_ratio = ratio
                    best_idx = idx
                    logger.debug('case 3 => {}: best_sum = {} ; best_sum_all = {} ; best_ratio = {}'.
                                 format(idx, best_sum, best_sum_all, best_ratio))
                else:
                    logger.debug('case 6 => {}: sum = {} ; sum_all = {} ; ratio = {}'.
                                 format(idx, sum, sum_all, ratio))

    logger.debug("best_idx = {}".format(best_idx))
    xmin, ymin, xmax, ymax = rect_array[best_idx]

    xmin = xmin - shrink_margin
    ymin = ymin - shrink_margin
    xmax = xmax - shrink_margin
    ymax = ymax - shrink_margin

    xmin = int(xmin/shrinkFactor) - margin
    ymin = int(ymin/shrinkFactor) - margin
    xmax = int(xmax/shrinkFactor) + margin
    ymax = int(ymax/shrinkFactor) + margin

    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax >= rawW:
        xmax = rawW - 1
    if ymax >= rawH:
        ymax = rawH - 1

    if shrink_image_wo_border_path != None:
        imgOrg = imgRaw.copy()
        cv2.rectangle(imgOrg, (xmin, ymin), (xmax, ymax), (0, 0, 255), 10)
        xmin_2 = xmin - 100
        ymin_2 = ymin - 100
        xmax_2 = xmax + 100
        ymax_2 = ymax + 100
        if xmin_2 < 0:
            xmin_2 = 1
        if ymin_2 < 0:
            ymin_2 = 1
        if xmax_2 > rawW:
            xmax_2 = rawW-1
        if ymax_2 > rawH:
            ymax_2 = rawH-1
        shrink_image_wo_border = cv2.resize(imgOrg[ymin_2:ymax_2, xmin_2:xmax_2], (0,0), fx=0.6, fy=0.6)
        cv2.imwrite(shrink_image_wo_border_path, shrink_image_wo_border, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    return xmin, ymin, xmax, ymax

def convert_to_border_removal_image_coordinate(bbox_list, crop_rect):
    crop_xmin, crop_ymin, crop_xmax, crop_ymax = crop_rect
    new_bbox_list = list()
    for idx, bbox in enumerate(bbox_list):
        file_name, error_type = bbox[0:2]
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox[2:6]
        bbox_xmin = bbox_xmin - crop_xmin
        bbox_ymin = bbox_ymin - crop_ymin
        bbox_xmax = bbox_xmax - crop_xmin
        bbox_ymax = bbox_ymax - crop_ymin
        new_bbox_list.append([file_name, error_type, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])
    return new_bbox_list

def remove_black_border_specific_data(input_dict):
    data_dir = input_dict['data_dir']
    mask_foreground_object = input_dict['mask_foreground_object']
    save_shrink_image = input_dict['save_shrink_image']

    image_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')

    image_wo_border_dir = os.path.join(data_dir, 'images_wo_border')
    label_wo_border_dir = os.path.join(data_dir, 'labels_wo_border')
    os_makedirs(image_wo_border_dir)
    os_makedirs(label_wo_border_dir)

    if mask_foreground_object:
        image_wo_border_mask_fg_dir = os.path.join(data_dir, 'images_wo_border_mask_fg')
        label_wo_border_mask_fg_dir = os.path.join(data_dir, 'labels_wo_border_mask_fg')
        os_makedirs(image_wo_border_mask_fg_dir)
        os_makedirs(label_wo_border_mask_fg_dir)

    # Save the shrink image for debug only
    if save_shrink_image:
        shrink_image_wo_border_dir = os.path.join(data_dir, 'shrink_images_wo_border')
        os_makedirs(shrink_image_wo_border_dir)

    # Mask foreground object and remove black border
    for idx, image_file_name in enumerate(os.listdir(image_dir)):
        image_file_path = os.path.join(image_dir, image_file_name)

        json_file_name = os.path.splitext(image_file_name)[0] + '.json'
        json_file_path = os.path.join(label_dir, json_file_name)
        bbox_list = json_to_bbox(json_file_path)

        image = cv2.imread(image_file_path)

        # Remove black border
        if save_shrink_image:
            shrink_image_wo_border_path = os.path.join(shrink_image_wo_border_dir, image_file_name)
            xmin, ymin, xmax, ymax = remove_black_border_ori(image, shrink_image_wo_border_path)
        else:
            xmin, ymin, xmax, ymax = remove_black_border_ori(image, None)

        # Save the without-border image to image_wo_border_dir
        image_wo_border = image[ymin:ymax,xmin:xmax]
        image_wo_border_file_path = os.path.join(image_wo_border_dir, image_file_name)
        cv2.imwrite(image_wo_border_file_path, image_wo_border, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # Save the without-border json to label_wo_border_dir
        crop_rect = [xmin, ymin, xmax, ymax]
        logger.debug('{:>4d}, {}, {}'.format(idx+1, image_file_name, crop_rect))
        bbox_wo_border_list = convert_to_border_removal_image_coordinate(bbox_list, crop_rect)
        csv_to_json(bbox_wo_border_list, image_wo_border_dir, label_wo_border_dir, coord_type="xmin_ymin_xmax_ymax")

        # Mask foreground object
        if mask_foreground_object:
            image_mask_fg = mask_bbox_in_image(bbox_wo_border_list, image_wo_border)
            image_wo_border_mask_fg_file_path = os.path.join(image_wo_border_mask_fg_dir, image_file_name)
            cv2.imwrite(image_wo_border_mask_fg_file_path, image_mask_fg, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            json_file_name = os.path.splitext(image_file_name)[0] + '.json'
            label_wo_border_file_path = os.path.join(label_wo_border_dir, json_file_name)
            label_wo_border_mask_fg_file_path = os.path.join(label_wo_border_mask_fg_dir, json_file_name)
            shutil_copyfile(label_wo_border_file_path, label_wo_border_mask_fg_file_path)


if __name__ == '__main__':
    # Remove black border for train_data and test_data
    train_data_dict = {
        'data_dir': config['train_data_dir'],
        'mask_foreground_object': False,
        'save_shrink_image': False
    }

    test_data_dict = {
        'data_dir': config['test_data_dir'],
        'mask_foreground_object': False,
        'save_shrink_image': False
    }
    #remove_black_border_specific_data(train_data_dict)
    
    input_path = "../../big_img/"
    for image_file_path in os.listdir(input_path):
        if not image_file_path.endswith("jpg"): continue
        print(image_file_path)
        imgRaw = cv2.imread(input_path+image_file_path)
        remove_black_border_ori(imgRaw, "../../test_crop_border/0_zcut_"+image_file_path.replace(".jpg","_ori.jpg"))
        remove_border(imgRaw, "../../test_crop_border/0_zcut_"+image_file_path)
