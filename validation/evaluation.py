#!/usr/bin/env python3
import os, sys
import logging
from collections import OrderedDict
import numpy as np

aoi_dir_name = os.getenv('AOI_DIR_NAME')
assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"

current_dir = os.path.dirname(os.path.abspath(__file__))
idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
aoi_dir = current_dir[:idx]

sys.path.append(os.path.join(aoi_dir, "config"))
from config import read_config_yaml, write_config_yaml

sys.path.append(os.path.join(aoi_dir, "utils"))
from utils import os_makedirs, os_remove, shutil_copyfile
from logger import get_logger

sys.path.append(os.path.join(aoi_dir, "data_preprocess"))
from csv_json_conversion import json_to_bbox
from crop_small_image import get_overlap
from labelme_coco_conversion import category_mapping as category_mapping_new
# from validation import sort_model_name_by_iter
from csv_json_conversion import csv_to_json

# Read config_file
config_file = os.path.join(aoi_dir, 'config/config.yaml')
config = read_config_yaml(config_file)

# Get logger
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
log_file_name = 'result.log'
logger = get_logger(name=__file__, console_handler_level=logging.INFO, file_handler_level=logging.INFO, file_name=log_file_name)

def sort_model_name_by_iter(model_file_name):
    model_file_id, ext = os.path.splitext(model_file_name)
    iter_str =  model_file_id.split('_')[1]
    return iter_str

def empty_all_statistics():
    global min_score
    global category_mapping
    global label_dict, label_tp_dict, label_tp_wrong_category_dict, label_fn_dict
    global truth_dict, truth_tp_dict, truth_tp_wrong_category_dict, truth_fn_dict
    global infer_dict, infer_tp_dict, infer_tp_wrong_category_dict, infer_fp_dict

    min_score = 1.0
    category_mapping = category_mapping_new
    """
    Ground Truth Label Statistics
    1. label_dict                   : all of the ground truth in original category
    2. label_tp_dict                : those ground truth which are detected by detector with correct category which is mapped by category_mapping()
    3. label_tp_wrong_category_dict : those ground truth which are detected by detector with wrong category which is mapped by category_mapping()
    4. label_fn_dict                : those ground truth which are not detected by detector

    [Note] label_dict = label_tp_dict + label_tp_wrong_category_dict + label_fn_dict
    """
    label_dict                   = OrderedDict({'bridge': 0, 'empty': 0, 'appearance_less': 0, 'appearance_hole': 0, 'excess_solder': 0, 'appearance': 0})
    label_tp_dict                = OrderedDict({'bridge': 0, 'empty': 0, 'appearance_less': 0, 'appearance_hole': 0, 'excess_solder': 0, 'appearance': 0})
    label_tp_wrong_category_dict = OrderedDict({'bridge': 0, 'empty': 0, 'appearance_less': 0, 'appearance_hole': 0, 'excess_solder': 0, 'appearance': 0})
    label_fn_dict                = OrderedDict({'bridge': 0, 'empty': 0, 'appearance_less': 0, 'appearance_hole': 0, 'excess_solder': 0, 'appearance': 0})

    """
    Ground Truth Statistics
    1. truth_dict                   : all of the ground truth
    2. truth_tp_dict                : those ground truth which are detected by detector with correct category (the same as infer_tp_dict)
    3. truth_tp_wrong_category_dict : those ground truth which are detected by detector with wrong category
    4. truth_fn_dict                : those ground truth which are not detected by detector

    [Note] truth_dict = truth_tp_dict + truth_tp_wrong_category_dict + truth_fn_dict
           the sum of truth_tp_wrong_category_dict = the sum of infer_tp_wrong_category_dict
           Combine 'empty', 'appearance_less' and 'appearance_hole' of ground truth label statistics into 'appearance_less' of ground truth statistics
    """
    truth_dict                   = OrderedDict({'bridge': 0, 'appearance_less': 0, 'excess_solder': 0, 'appearance': 0})
    truth_tp_dict                = OrderedDict({'bridge': 0, 'appearance_less': 0, 'excess_solder': 0, 'appearance': 0})
    truth_tp_wrong_category_dict = OrderedDict({'bridge': 0, 'appearance_less': 0, 'excess_solder': 0, 'appearance': 0})
    truth_fn_dict                = OrderedDict({'bridge': 0, 'appearance_less': 0, 'excess_solder': 0, 'appearance': 0})

    """
    Inference Result Statistics
    1. infer_dict                   : all of the detected object
    2. infer_tp_dict                : those detected objects which match the ground truth with correct category (the same as truth_tp_dict)
    3. infer_tp_wrong_category_dict : those detected objects which match the ground truth with wrong category
    4. infer_fp_dict                : those detected objects which does not match the ground truth

    [Note] infer_dict = infer_tp_dict + infer_tp_wrong_category_dict + infer_fp_dict
           the sum of truth_tp_wrong_category_dict = the sum of infer_tp_wrong_category_dict
           Combine 'empty', 'appearance_less' and 'appearance_hole' of ground truth label statistics into 'appearance_less' of inference result statistics
    """
    infer_dict                   = OrderedDict({'bridge': 0, 'appearance_less': 0, 'excess_solder': 0, 'appearance': 0})
    infer_tp_dict                = OrderedDict({'bridge': 0, 'appearance_less': 0, 'excess_solder': 0, 'appearance': 0})
    infer_tp_wrong_category_dict = OrderedDict({'bridge': 0, 'appearance_less': 0, 'excess_solder': 0, 'appearance': 0})
    infer_fp_dict                = OrderedDict({'bridge': 0, 'appearance_less': 0, 'excess_solder': 0, 'appearance': 0})

def get_best_infer_candidate(candidate_dict):
    """
    Choose best candidate from candidate_dict.
    First, choose the highest overlap area ratio. Second, choose the highest confidence score

    Args:
        candidate_dict (dict):  { overlap_area_ratio_1: [[file_name, error_type, xmin, ymin, xmax, ymax, score], ...],
                                  overlap_area_ratio_2: [[file_name, error_type, xmin, ymin, xmax, ymax, score], ...],
                                  ...
                                }
    Returns:
        candidate (list): best candidate in [file_name, error_type, xmin, ymin, xmax, ymax, score] format.
    """
    max_key = max(candidate_dict.keys())
    candidate_list = candidate_dict[max_key] # [[file_name, error_type, xmin, ymin, xmax, ymax, score], ...]
    if len(candidate_list) > 1:
        candidate_list.sort(key=lambda x: x[6], reverse=True)
    return candidate_list[0]

def dict_to_str(dict_obj):
    """
    Conver dict to formatted string

    Args:
        dict_obj (dict): dict.
    Returns:
        str_obj (str): formatted string.
    """
    str_obj = ""
    for key in dict_obj:
        str_add = "%s = %6d" % (key, dict_obj[key])
        str_obj = str_obj + str_add + "; "
    return str_obj

def get_critical_recall(label_dict, label_tp_dict, label_tp_wrong_category_dict, label_fn_dict, critical_error=['bridge', 'empty']):
    TP = 0
    FN = 0
    for key in label_dict:
        if key in critical_error:
            TP = TP + label_tp_dict[key] + label_tp_wrong_category_dict[key]
            FN = FN + label_fn_dict[key]
    try:
        recall = round(TP / (TP + FN), 4)
    except:
        recall = -1
    return recall

def get_normal_recall(truth_dict, truth_tp_dict, truth_tp_wrong_category_dict, truth_fn_dict):
    TP = 0
    FN = 0
    for key in truth_dict:
        TP = TP + truth_tp_dict[key] + truth_tp_wrong_category_dict[key]
        FN = FN + truth_fn_dict[key]
    try:
        recall = round(TP / (TP + FN), 4)
    except:
        recall = -1
    return recall

def get_normal_precision(infer_dict, infer_tp_dict, infer_tp_wrong_category_dict, infer_fp_dict):
    TP = 0
    FP = 0
    for key in infer_dict:
        TP = TP + infer_tp_dict[key] + infer_tp_wrong_category_dict[key]
        FP = FP + infer_fp_dict[key]
    try:
        precision = round(TP / (TP + FP), 4)
    except:
        precision = -1
    return precision

def get_fp_rate(precision):
    if precision == -1:
        return precision
    else:
        return round(1 - precision, 4)

def compare_ground_truth_with_inference_result(ground_truth_image_dir_list, ground_truth_label_dir_list, \
                                               inference_result_label_dir_list, score_threshold, \
                                               eval_result=None, eval_dict=None, save_fn_bbox=False):
    """
    Compare ground truth with inference result.

    Args:
        ground_truth_image_dir (str): directory of ground truth image which is used only if save_fn_bbox is True
        ground_truth_label_dir (str): directory of ground truth label
        inference_result_label_dir (str): directory of inference result label
        score_threshold (float): if score is less than score_threshold, filter that out
        save_fn_bbox (bool): choose to save FN bounding box or not
    """
    global min_score
    global category_mapping
    global label_dict, label_tp_dict, label_tp_wrong_category_dict, label_fn_dict
    global truth_dict, truth_tp_dict, truth_tp_wrong_category_dict, truth_fn_dict
    global infer_dict, infer_tp_dict, infer_tp_wrong_category_dict, infer_fp_dict

    for ground_truth_image_dir, ground_truth_label_dir, inference_result_label_dir in \
        zip(ground_truth_image_dir_list, ground_truth_label_dir_list, inference_result_label_dir_list):

        # Iterate through ground_truth_label_dir
        for json_idx, json_file_name in enumerate(os.listdir(ground_truth_label_dir)):
            fn_bbox_list = list()

            # Get truth_bbox_list = [[file_name, error_type, xmin, ymin, xmax, ymax], ...]
            # error_type = 'bridge' / 'empty' / 'appearance_less' / 'appearance_hole' / 'excess_solder' / 'appearance'
            truth_json_file_path = os.path.join(ground_truth_label_dir, json_file_name)
            truth_bbox_list = json_to_bbox(truth_json_file_path, store_score=False)
            # truth_bbox_list = non_max_suppression_no_score(truth_bbox_list, 0.5)

            # Get infer_bbox_list = [[file_name, error_type, xmin, ymin, xmax, ymax, score], ...]
            # error_type = 'bridge' / 'appearance_less' / 'excess_solder' / 'appearance'
            infer_json_file_path = os.path.join(inference_result_label_dir, json_file_name)
            if os.path.isfile(infer_json_file_path):
                infer_bbox_list = json_to_bbox(infer_json_file_path, store_score=True)
                infer_bbox_list = filter(lambda bbox: bbox[6] >= score_threshold, infer_bbox_list)
                infer_bbox_list = list(infer_bbox_list)
            else:
                infer_bbox_list = list()

            # Iterate through truth_bbox_list and compare with infer_bbox_list
            for idx, truth_bbox in enumerate(truth_bbox_list):
                truth_file_name, truth_error_type = truth_bbox[0:2]
                truth_rect = truth_bbox[2:6] # [xmin, ymin, xmax, ymax]
                truth_area = (truth_rect[2] - truth_rect[0]) * (truth_rect[3] - truth_rect[1])
                if truth_error_type in category_mapping:
                    label_error_type = truth_error_type
                    label_dict[label_error_type] += 1
                    truth_error_type = category_mapping.get(truth_error_type)
                    truth_dict[truth_error_type] += 1
                else:
                    logger.debug("{} is undefined".format(truth_error_type))
                    continue

                # Get candidate_dict from inference bounding box list
                # candidate_dict = { overlap_area_ratio_1: [[file_name, error_type, xmin, ymin, xmax, ymax, score], ...],
                #                    overlap_area_ratio_2: [[file_name, error_type, xmin, ymin, xmax, ymax, score], ...],
                #                    ...
                #                  }
                candidate_dict = dict()
                for infer_bbox in infer_bbox_list:
                    infer_file_name, infer_error_type = infer_bbox[0:2]
                    if infer_error_type == 'elh':
                        infer_error_type = 'appearance_less'
                    infer_rect = infer_bbox[2:6] # [xmin, ymin, xmax, ymax]
                    infer_score = infer_bbox[6]
                    if infer_score < min_score:
                        min_score = infer_score
                    if idx == 0:
                        infer_dict[infer_error_type] += 1
                    overlap_area = get_overlap(truth_rect, infer_rect, "area")
                    if overlap_area > 0:
                        overlap_area_ratio = round(overlap_area / truth_area, 4)
                        if candidate_dict.get(overlap_area_ratio) == None:
                            candidate_dict[overlap_area_ratio] = list()
                        candidate_dict[overlap_area_ratio].append(infer_bbox)

                if len(candidate_dict) > 0:
                    # candidate = [file_name, error_type, xmin, ymin, xmax, ymax, score]
                    candidate = get_best_infer_candidate(candidate_dict)
                    candidate_error_type = candidate[1]
                    if candidate_error_type == 'elh':
                        candidate_error_type = 'appearance_less'
                    if candidate_error_type == truth_error_type:
                        truth_tp_dict[truth_error_type] += 1
                        label_tp_dict[label_error_type] += 1
                        infer_tp_dict[truth_error_type] += 1
                    else:
                        truth_tp_wrong_category_dict[truth_error_type] += 1
                        label_tp_wrong_category_dict[label_error_type] += 1
                        infer_tp_wrong_category_dict[candidate_error_type] += 1

                    infer_bbox_list.remove(candidate)
                else:
                    truth_fn_dict[truth_error_type] += 1
                    label_fn_dict[label_error_type] += 1
                    fn_bbox_list.append(truth_bbox)

            if save_fn_bbox:
                inference_fn_label_dir = os.path.join(os.path.dirname(inference_result_label_dir), 'fn_labels')
                if json_idx == 0:
                    os_makedirs(inference_fn_label_dir)

                if len(fn_bbox_list) > 0:
                    for fn_bbox in fn_bbox_list:
                        error_type = fn_bbox[1]
                        if error_type == 'bridge' or error_type == 'empty':
                            csv_to_json(fn_bbox_list, ground_truth_image_dir, inference_fn_label_dir, coord_type="xmin_ymin_xmax_ymax", store_score=False)
                            break

            # Calculate FP from remaining infer_bbox_list
            for infer_bbox in infer_bbox_list:
                infer_error_type = infer_bbox[1]
                if infer_error_type == 'elh':
                    infer_error_type = 'appearance_less'
                infer_fp_dict[infer_error_type] += 1

    critical_recall = get_critical_recall(label_dict, label_tp_dict, label_tp_wrong_category_dict, label_fn_dict, critical_error=['bridge', 'empty'])
    normal_recall = get_normal_recall(truth_dict, truth_tp_dict, truth_tp_wrong_category_dict, truth_fn_dict)
    normal_precision = get_normal_precision(infer_dict, infer_tp_dict, infer_tp_wrong_category_dict, infer_fp_dict)
    fp_rate = get_fp_rate(normal_precision)

    if eval_dict is not None:
        logger.info("score_threshold              : {}".format(score_threshold))
        logger.info("min_score                    : {}".format(min_score))
        logger.info("")
        logger.info("label_dict                   : {}".format(dict_to_str(label_dict)))
        logger.info("label_tp_dict                : {}".format(dict_to_str(label_tp_dict)))
        logger.info("label_tp_wrong_category_dict : {}".format(dict_to_str(label_tp_wrong_category_dict)))
        logger.info("label_fn_dict                : {}".format(dict_to_str(label_fn_dict)))
        logger.info("")
        logger.info("truth_dict                   : {}".format(dict_to_str(truth_dict)))
        logger.info("truth_tp_dict                : {}".format(dict_to_str(truth_tp_dict)))
        logger.info("truth_tp_wrong_category_dict : {}".format(dict_to_str(truth_tp_wrong_category_dict)))
        logger.info("truth_fn_dict                : {}".format(dict_to_str(truth_fn_dict)))
        logger.info("")
        logger.info("infer_dict                   : {}".format(dict_to_str(infer_dict)))
        logger.info("infer_tp_dict                : {}".format(dict_to_str(infer_tp_dict)))
        logger.info("infer_tp_wrong_category_dict : {}".format(dict_to_str(infer_tp_wrong_category_dict)))
        logger.info("infer_fp_dict                : {}".format(dict_to_str(infer_fp_dict)))
        logger.info("")
        logger.info("critical recall              : {}".format(critical_recall))
        logger.info("normal recall                : {}".format(normal_recall))
        logger.info("normal precision             : {}".format(normal_precision))
        logger.info("fp rate                      : {}".format(fp_rate))
        logger.info("= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ")

    if eval_result is not None:
        eval_result['score_threshold'] = score_threshold
        eval_result['min_score'] = min_score
        eval_result['label_dict'] = label_dict
        eval_result['label_tp_dict'] = label_tp_dict
        eval_result['label_tp_wrong_category_dict'] = label_tp_wrong_category_dict
        eval_result['label_fn_dict'] = label_fn_dict
        eval_result['truth_dict'] = truth_dict
        eval_result['truth_tp_dict'] = truth_tp_dict
        eval_result['truth_tp_wrong_category_dict'] = truth_tp_wrong_category_dict
        eval_result['truth_fn_dict'] = truth_fn_dict
        eval_result['infer_dict'] = infer_dict
        eval_result['infer_tp_dict'] = infer_tp_dict
        eval_result['infer_tp_wrong_category_dict'] = infer_tp_wrong_category_dict
        eval_result['infer_fp_dict'] = infer_fp_dict
        eval_result['critical_recall'] = critical_recall
        eval_result['normal_recall'] = normal_recall
        eval_result['normal_precision'] = normal_precision
        eval_result['fp_rate'] = fp_rate

        if eval_dict is not None:
            if critical_recall not in eval_dict:
                eval_dict[critical_recall] = list()
            eval_dict[critical_recall].append(eval_result)

def get_best_model_iters(eval_dict):
    # Get several models according to descending order of critical recall
    eval_dict_keys = sorted(eval_dict.keys(), reverse = True)
    best_model_iter_list = list()
    start = 0
    logger.info("")
    logger.info("get_best_model_iters")
    for idx, key in enumerate(eval_dict_keys):
        for element in eval_dict[key]:
            best_model_iter_list.append(element['model_iter'])
        logger.info("critical recall = {}; length = {}; model_iter = {}".
                    format(key, len(eval_dict[key]), best_model_iter_list[start:]))
        start = len(best_model_iter_list)
        if len(best_model_iter_list)>10 or (idx>0 and len(best_model_iter_list)>6):
            break
    logger.info("")
    logger.info("= "*50)
    return best_model_iter_list

def get_eval_result(eval_dict):
    # Sort in descending order of critical recall
    eval_dict_keys = sorted(eval_dict.keys(), reverse = True)
    eval_result_list = list()
    for key in eval_dict_keys:
        for element in eval_dict[key]:
            eval_result_list.append(element)
    return eval_result_list

def evaluation_through_model_iter_by_score_threshold(input_dict):
    ground_truth_image_dir_list = input_dict['ground_truth_image_dir_list']
    ground_truth_label_dir_list = input_dict['ground_truth_label_dir_list']
    inference_result_dir_list = input_dict['inference_result_dir_list']
    model_iter_list = input_dict['model_iter_list']
    score_threshold = input_dict['score_threshold']
    save_fn_bbox = input_dict['save_fn_bbox']

    if model_iter_list==None:
        for idx, inference_result_dir in enumerate(inference_result_dir_list):
            if idx==0:
                model_iter_list = os.listdir(inference_result_dir)
            else:
                model_iter_list = list(set(model_iter_list) & set(os.listdir(inference_result_dir)))

        # Sort the model_iter folder by iteration number
        model_iter_list.sort(key = sort_model_name_by_iter)

    eval_dict = dict()
    # Iterate through all model_iter folder
    for model_iter in model_iter_list:
        logger.info("model_iter                   : {}".format(model_iter))
        '''
        if iter_num % 1000 == 0 and iter_num < 19000:
            logger.info("model_iter                   : {}".format(model_iter))
        else:
            continue
        '''
        inference_result_label_dir_list = [os.path.join(inference_result_dir, model_iter, 'labels') for inference_result_dir in inference_result_dir_list]
        empty_all_statistics()
        eval_result = OrderedDict()
        eval_result['model_iter'] = model_iter
        compare_ground_truth_with_inference_result(ground_truth_image_dir_list, ground_truth_label_dir_list, \
                                                   inference_result_label_dir_list, score_threshold, \
                                                   eval_result=eval_result, eval_dict=eval_dict, save_fn_bbox=save_fn_bbox)

    return get_eval_result(eval_dict)

def evaluation_by_fp_rate(input_dict, eval_dict=None, refine_num=11):
    model_type = input_dict['model_type']
    ground_truth_image_dir_list = input_dict['ground_truth_image_dir_list']
    ground_truth_label_dir_list = input_dict['ground_truth_label_dir_list']
    inference_result_label_dir_list = input_dict['inference_result_label_dir_list']
    model_iter = input_dict['model_iter']
    fp_rate_threshold = input_dict['fp_rate_threshold']
    save_fn_bbox = input_dict['save_fn_bbox']

    logger.info("model_iter                   : {}".format(model_iter))
    # Iterate score_threshold from 0.1 to 1.0 with interval 0.1
    #score_threshold_coarse_list = [round(x * 0.01, 2) for x in range(1, 11)]
    if model_type == 'YOLOv4':
        score_threshold_coarse_list = list(np.linspace(0.0, 1.0, num=100, endpoint=False))
        score_threshold_coarse_list = [round(x, 3) for x in score_threshold_coarse_list]
    else: #model_type in ['RetinaNet','CenterNet2']
        score_threshold_coarse_list = list(np.linspace(0.1, 1.0, num=10, endpoint=True))
        score_threshold_coarse_list = [round(x, 2) for x in score_threshold_coarse_list]
    last_score_threshold = 0.0
    score_threshold_coarse_range = list()
    small_threshold = 1.0
    
    for score_threshold in score_threshold_coarse_list:
        empty_all_statistics()

        eval_result = OrderedDict()
        eval_result['model_iter'] = model_iter
        compare_ground_truth_with_inference_result(ground_truth_image_dir_list, ground_truth_label_dir_list, \
                                                   inference_result_label_dir_list, score_threshold, \
                                                   eval_result=eval_result, eval_dict=None, save_fn_bbox=False)

        if eval_result['fp_rate'] < fp_rate_threshold:
            score_threshold_coarse_range.append(last_score_threshold)
            score_threshold_coarse_range.append(score_threshold)
            break
        if eval_result['fp_rate'] < small_threshold:
            small_threshold = eval_result['fp_rate']
        last_score_threshold = score_threshold
    
    if score_threshold_coarse_range == []:
        score_threshold_coarse_range = [np.floor(small_threshold*10)/ 10.0, np.ceil(small_threshold*10)/ 10.0]
        score_threshold_coarse_range = [round(x, 2) for x in score_threshold_coarse_range]
    logger.info("score_threshold_coarse_range : {}".format(score_threshold_coarse_range))


    # Iterate score_threshold from score_threshold_coarse_range[0] to score_threshold_coarse_range[1] with interval 0.01
    score_threshold_refine_list = list(np.linspace(*score_threshold_coarse_range, num=refine_num))
    score_threshold_refine_list = [round(x, 3) for x in score_threshold_refine_list]
    last_fp_rate = 0.0
    score_threshold_refine_range = list()
    fp_rate_refine_range = list()
    for score_threshold in score_threshold_refine_list:
        empty_all_statistics()

        eval_result = OrderedDict()
        eval_result['model_iter'] = model_iter
        compare_ground_truth_with_inference_result(ground_truth_image_dir_list, ground_truth_label_dir_list, \
                                                   inference_result_label_dir_list, score_threshold, \
                                                   eval_result=eval_result, eval_dict=None, save_fn_bbox=False)
        
        score_threshold_refine_range = [last_score_threshold, score_threshold]
        fp_rate_refine_range = [last_fp_rate, eval_result['fp_rate']]
        
        if eval_result['fp_rate'] < fp_rate_threshold:
            break
        
        last_score_threshold = score_threshold
        last_fp_rate = eval_result['fp_rate']
    logger.info("score_threshold_refine_range : {}".format(score_threshold_refine_range))
    logger.info("fp_rate_refine_range         : {}".format(fp_rate_refine_range))


    # Get score_threshold according to specific fp_rate_threshold
    diff_abs = np.abs(np.array(fp_rate_refine_range) - fp_rate_threshold)
    idx = np.argmin(diff_abs)
    score_threshold = score_threshold_refine_range[idx]

    empty_all_statistics()

    eval_result = OrderedDict()
    eval_result['model_iter'] = model_iter
    if eval_dict == None:
        eval_dict = dict()
    compare_ground_truth_with_inference_result(ground_truth_image_dir_list, ground_truth_label_dir_list, \
                                               inference_result_label_dir_list, score_threshold, \
                                               eval_result=eval_result, eval_dict=eval_dict, save_fn_bbox=save_fn_bbox)
    return eval_result

def evaluation_through_model_iter_by_fp_rate(input_dict):
    inference_result_dir_list = input_dict['inference_result_dir_list']
    model_iter_list = input_dict['model_iter_list']
    return_type = input_dict['return_type']

    if model_iter_list==None:
        for idx, inference_result_dir in enumerate(inference_result_dir_list):
            if idx==0:
                model_iter_list = os.listdir(inference_result_dir)
            else:
                model_iter_list = list(set(model_iter_list) & set(os.listdir(inference_result_dir)))

        # Sort the model_iter folder by iteration number
        model_iter_list.sort(key = sort_model_name_by_iter)
    
    #model_iter_list = ["model_0000111","model_0000011"]

    eval_dict = dict()
    # Iterate through all model_iter folder
    for model_iter in model_iter_list:

        inference_result_label_dir_list = [os.path.join(inference_result_dir, model_iter, 'labels') for inference_result_dir in inference_result_dir_list]

        input_dict['inference_result_label_dir_list'] = inference_result_label_dir_list
        input_dict['model_iter'] = model_iter
        evaluation_by_fp_rate(input_dict, eval_dict=eval_dict)

    if return_type == "eval_result":
        return get_eval_result(eval_dict)
    elif return_type == "best_models":
        return get_best_model_iters(eval_dict)

def evaluation_ensemble_result(ensemble_result_dirname = 'ensemble_result', \
                               score_threshold = 0.01, \
                               test_and_fn_list = ['test', 'fn']):
    # global logger
    # log_file_path = os.path.join(aoi_dir, 'validation', log_file_name)
    # os_remove(log_file_path)
    # while logger.hasHandlers():
    #     logger.removeHandler(logger.handlers[0])
    # logger = get_logger(name=__file__, console_handler_level=logging.INFO, file_handler_level=logging.INFO, file_name=log_file_name)
    logger.info("ensemble_result_dirname      : {}".format(ensemble_result_dirname))
    test_data_dir = config['test_data_dir']
    retrain_data_val_dir = config['retrain_data_val_dir']
    ensemble_result_dir = config['ensemble_result_dir']

    ground_truth_image_dir_list = list()
    ground_truth_label_dir_list = list()
    inference_result_label_dir_list = list()

    ensemble_result = OrderedDict()

    for test_and_fn in test_and_fn_list:
        if test_and_fn == 'test':
            ground_truth_image_dir_list.append(os.path.join(test_data_dir, 'images_wo_border'))
            ground_truth_label_dir_list.append(os.path.join(test_data_dir, 'labels_wo_border'))
            inference_result_label_dir_list.append(os.path.join(ensemble_result_dir, ensemble_result_dirname, 'test'))
        elif test_and_fn == 'fn':
            ground_truth_image_dir_list.append(os.path.join(retrain_data_val_dir, 'images_random_crop'))
            ground_truth_label_dir_list.append(os.path.join(retrain_data_val_dir, 'labels_random_crop'))
            inference_result_label_dir_list.append(os.path.join(ensemble_result_dir, ensemble_result_dirname, 'fn'))
        else:
            continue

        # eval_result = OrderedDict()
        # eval_dict = OrderedDict()
        # empty_all_statistics()
        # logger.info("data_type                    : {}".format(test_and_fn))
        # compare_ground_truth_with_inference_result( [ground_truth_image_dir_list[-1]], [ground_truth_label_dir_list[-1]], \
        #                                             [inference_result_label_dir_list[-1]], score_threshold,
        #                                             eval_result=eval_result, eval_dict=eval_dict, save_fn_bbox=False)
        # ensemble_result[test_and_fn] = eval_result

    if len(inference_result_label_dir_list) > 1:
        eval_result = OrderedDict()
        eval_dict = OrderedDict()
        empty_all_statistics()
        logger.info("data_type                    : {}".format('test_and_fn'))
        compare_ground_truth_with_inference_result( ground_truth_image_dir_list, ground_truth_label_dir_list, \
                                                    inference_result_label_dir_list, score_threshold,
                                                    eval_result=eval_result, eval_dict=eval_dict, save_fn_bbox=False)
        ensemble_result['test_and_fn'] = eval_result

    ensemble_result_yaml_file_path = os.path.join(ensemble_result_dir, ensemble_result_dirname, '{}.yaml'.format(ensemble_result_dirname))
    write_config_yaml(ensemble_result_yaml_file_path, ensemble_result)

    log_file_path = os.path.join(aoi_dir, 'validation', log_file_name)
    ensemble_result_txt_file_path = os.path.join(ensemble_result_dir, ensemble_result_dirname, '{}.txt'.format(ensemble_result_dirname))
    shutil_copyfile(log_file_path ,ensemble_result_txt_file_path)

    return ensemble_result, ensemble_result_yaml_file_path, ensemble_result_txt_file_path


def take_total_fn(elem):
    # return elem[3]
    return elem[4]

def take_total_fp(elem):
    # return elem[4]
    return elem[5]

def take_total_fn_and_fp(elem):
    # return elem[3] + elem[4]
    return elem[4] + elem[5]

def take_critical_loss(elem):
    # return elem[1]+elem[2]
    return elem[2] + elem[3]

def take_normal_recall(elem):
    return elem[7]

def get_best_model(eval_result_list):
    stat_list = list()
    for idx, eval_result in enumerate(eval_result_list):
        model_iter = eval_result['model_iter']
        score_threshold = eval_result['score_threshold']
        min_score = eval_result['min_score']
        label_dict = eval_result['label_dict']
        label_tp_dict = eval_result['label_tp_dict']
        label_tp_wrong_category_dict = eval_result['label_tp_wrong_category_dict']
        label_fn_dict = eval_result['label_fn_dict']
        truth_dict = eval_result['truth_dict']
        truth_tp_dict = eval_result['truth_tp_dict']
        truth_tp_wrong_category_dict = eval_result['truth_tp_wrong_category_dict']
        truth_fn_dict = eval_result['truth_fn_dict']
        infer_dict = eval_result['infer_dict']
        infer_tp_dict = eval_result['infer_tp_dict']
        infer_tp_wrong_category_dict = eval_result['infer_tp_wrong_category_dict']
        infer_fp_dict = eval_result['infer_fp_dict']
        critical_recall = eval_result['critical_recall']
        normal_recall = eval_result['normal_recall']
        normal_precision = eval_result['normal_precision']
        fp_rate = eval_result['fp_rate']

        bridge_fn = label_fn_dict['bridge']
        empty_fn = label_fn_dict['empty']
        total_fn = sum([label_fn_dict[key] for key in label_fn_dict])
        total_gt = sum([label_dict[key] for key in label_dict])

        total_fp = sum([infer_fp_dict[key] for key in infer_fp_dict])
        total_dt = sum([infer_dict[key] for key in infer_dict])

        # stat_list.append([model_iter, bridge_fn, empty_fn, total_fn, total_fp])
        stat_list.append([model_iter, min_score,\
                          bridge_fn, empty_fn, total_fn, total_fp,\
                          critical_recall, normal_recall, normal_precision, fp_rate])

    logger.info("Sort by critical_loss")
    # logger.info("[model_iter, bridge_fn, empty_fn, total_fn, total_fp]")
    logger.info("{:>25s}, {:>9s}, {:>9s}, {:>8s}, {:>8s}, {:>8s}, {:>15s}, {:>13s}, {:>16s}, {:>8s}"
                .format('model_iter', 'threshold',
                        'bridge_fn', 'empty_fn', 'total_fn', 'total_fp',
                        'critical_recall', 'normal_recall', 'normal_precision', 'fp_rate'))
    stat_list.sort(key=take_critical_loss)
    for stat in stat_list:
        logger.info("{:>25s}, {:>9f}, {:>9d}, {:>8d}, {:>8d}, {:>8d}, {:>15f}, {:>13f}, {:>16f}, {:>8f}"
                    .format(*stat))
    logger.info("")

    if len(stat_list)==2:
        logger.info("Select best one from two models")
        # logger.info("[model_iter, bridge_fn, empty_fn, total_fn, total_fp]")
        logger.info("{:>25s}, {:>9s}, {:>9s}, {:>8s}, {:>8s}, {:>8s}, {:>15s}, {:>13s}, {:>16s}, {:>8s}"
                    .format('model_iter', 'threshold',
                            'bridge_fn', 'empty_fn', 'total_fn', 'total_fp',
                            'critical_recall', 'normal_recall', 'normal_precision', 'fp_rate'))
        stat_best_1 = stat_list[0]
        stat_best_2 = stat_list[1]
        if stat_best_1[2] + stat_best_1[3] == stat_best_2[2] + stat_best_2[3]: # bridge_fn + empty_fn
            stat_list.sort(key=take_normal_recall, reverse=True)
            stat_best_1 = stat_list[0]

        logger.info("{:>25s}, {:>9f}, {:>9d}, {:>8d}, {:>8d}, {:>8d}, {:>15f}, {:>13f}, {:>16f}, {:>8f}"
                    .format(*stat_best_1))
        logger.info("")
        logger.info("Best model = {}".format(stat_best_1))
        return stat_best_1[0]
    else:
        logger.info("Filter half out")
        # logger.info("[model_iter, bridge_fn, empty_fn, total_fn, total_fp]")
        logger.info("{:>25s}, {:>9s}, {:>9s}, {:>8s}, {:>8s}, {:>8s}, {:>15s}, {:>13s}, {:>16s}, {:>8s}"
                    .format('model_iter', 'threshold',
                            'bridge_fn', 'empty_fn', 'total_fn', 'total_fp',
                            'critical_recall', 'normal_recall', 'normal_precision', 'fp_rate'))
        end = len(stat_list)/2
        if end.is_integer():
            end = int(end)
        else:
            end = int(end) + 1

        if end > 3:
            end = 3
        stat_list = stat_list[:end]
        for stat in stat_list:
            logger.info("{:>25s}, {:>9f}, {:>9d}, {:>8d}, {:>8d}, {:>8d}, {:>15f}, {:>13f}, {:>16f}, {:>8f}"
                        .format(*stat))
        logger.info("")

        stat_best_1 = stat_list[0]

    logger.info("Sort by take_normal_recall")
    # logger.info("[model_iter, bridge_fn, empty_fn, total_fn, total_fp]")
    logger.info("{:>25s}, {:>9s}, {:>9s}, {:>8s}, {:>8s}, {:>8s}, {:>15s}, {:>13s}, {:>16s}, {:>8s}"
                .format('model_iter', 'threshold',
                        'bridge_fn', 'empty_fn', 'total_fn', 'total_fp',
                        'critical_recall', 'normal_recall', 'normal_precision', 'fp_rate'))
    stat_list.sort(key=take_normal_recall, reverse=True)
    for stat in stat_list:
        logger.info("{:>25s}, {:>9f}, {:>9d}, {:>8d}, {:>8d}, {:>8d}, {:>15f}, {:>13f}, {:>16f}, {:>8f}"
                    .format(*stat))
    logger.info("")

    stat_best_2 = stat_list[0]

    if stat_best_1 == stat_best_2:
        logger.info("Best model = {}".format(stat_best_1))
        return stat_best_1[0]
    else:
        # compare normal recall
        if stat_best_2[7] >= stat_best_1[7] + 0.05:
            logger.info("Best model = {}".format(stat_best_2))
            return stat_best_2[0]
        else:
            logger.info("Best model = {}".format(stat_best_1))
            return stat_best_1[0]

def evaluate_old_model_on_test_and_fn_by_score_threshold(score_threshold):
    test_data_dir = config['test_data_dir']
    retrain_data_val_dir = config['retrain_data_val_dir']

    test_and_fn_data_dict = {
        'ground_truth_image_dir_list': [os.path.join(test_data_dir, 'images_wo_border'),
                                        os.path.join(retrain_data_val_dir, 'images_random_crop')],
        'ground_truth_label_dir_list': [os.path.join(test_data_dir, 'labels_wo_border'),
                                        os.path.join(retrain_data_val_dir, 'labels_random_crop')],
        'inference_result_dir_list': [os.path.join(test_data_dir, 'old_inference_result'),
                                      os.path.join(retrain_data_val_dir, 'old_inference_result')],
        'model_iter_list': os.listdir(os.path.join(test_data_dir, 'old_inference_result')),
        'score_threshold': score_threshold, # only for evaluation_through_model_iter_by_score_threshold()
        'fp_rate_threshold': None,          # only for evaluation_through_model_iter_by_fp_rate()
        'return_type': None,                # only for evaluation_through_model_iter_by_fp_rate()
        'save_fn_bbox': False,
    }

    eval_result_list = evaluation_through_model_iter_by_score_threshold(test_and_fn_data_dict)
    return eval_result_list

def evaluate_old_model_on_test_and_fn_by_fp_rate(model_type, fp_rate, return_type):
    test_data_dir = config['test_data_dir']
    retrain_data_val_dir = config['retrain_data_val_dir']

    if model_type == 'RetinaNet':
        old_model_file_path = config['retinanet_old_model_file_path']
    elif model_type == 'CenterNet2':
        old_model_file_path = config['centernet2_old_model_file_path']
    elif model_type == 'YOLOv4':
        old_model_file_path = config['yolov4_old_model_file_path']
    else:
        assert False, "model_type = {} is not supported".format(model_type)

    old_model_file_name = os.path.basename(old_model_file_path)
    old_model_file_id, ext = os.path.splitext(old_model_file_name)
    test_and_fn_data_dict = {
        'model_type': model_type,
        'ground_truth_image_dir_list': [os.path.join(test_data_dir, 'images_wo_border'),
                                        os.path.join(retrain_data_val_dir, 'images_random_crop')],
        'ground_truth_label_dir_list': [os.path.join(test_data_dir, 'labels_wo_border'),
                                        os.path.join(retrain_data_val_dir, 'labels_random_crop')],
        'inference_result_dir_list': [os.path.join(test_data_dir, '{}_old_inference_result'.format(model_type)),
                                      os.path.join(retrain_data_val_dir, '{}_old_inference_result'.format(model_type))],
        'model_iter_list': [old_model_file_id],
        'score_threshold': None,        # only for evaluation_through_model_iter_by_score_threshold()
        'fp_rate_threshold': fp_rate,   # only for evaluation_through_model_iter_by_fp_rate()
        'return_type': return_type,     # only for evaluation_through_model_iter_by_fp_rate()
        'save_fn_bbox': False,
    }

    eval_result_list = evaluation_through_model_iter_by_fp_rate(test_and_fn_data_dict)
    return eval_result_list

def evaluate_new_models_on_val(model_type, fp_rate, return_type):
    val_data_dir = config['val_data_dir']

    if model_type == 'RetinaNet':
        model_output_version = config['retinanet_model_output_version']
    elif model_type == 'CenterNet2':
        model_output_version = config['centernet2_model_output_version']
    elif model_type == 'YOLOv4':
        model_output_version = config['yolov4_model_output_version']
    else:
        assert False, "model_type = {} is not supported".format(model_type)

    val_data_dict = {
        'model_type': model_type,
        'ground_truth_image_dir_list': [os.path.join(val_data_dir, 'images_wo_border')],
        'ground_truth_label_dir_list': [os.path.join(val_data_dir, 'labels_wo_border')],
        'inference_result_dir_list': [os.path.join(val_data_dir, '{}_inference_result'.format(model_output_version))],
        'model_iter_list': None,
        'score_threshold': None,        # only for evaluation_through_model_iter_by_score_threshold()
        'fp_rate_threshold': fp_rate,   # only for evaluation_through_model_iter_by_fp_rate()
        'return_type': return_type,     # only for evaluation_through_model_iter_by_fp_rate()
        'save_fn_bbox': False,
    }

    val_best_model_iter_list = evaluation_through_model_iter_by_fp_rate(val_data_dict)
    return val_best_model_iter_list

def evaluate_new_models_on_test_and_fn_by_fp_rate(model_type, model_iter_list, fp_rate, return_type, test_and_fn = ['test', 'fn']):
    test_data_dir = config['test_data_dir']
    retrain_data_val_dir = config['retrain_data_val_dir']

    if model_type == 'RetinaNet':
        model_output_version = config['retinanet_model_output_version']
    elif model_type == 'CenterNet2':
        model_output_version = config['centernet2_model_output_version']
    elif model_type == 'YOLOv4':
        model_output_version = config['yolov4_model_output_version']
    else:
        assert False, "model_type = {} is not supported".format(model_type)

    ground_truth_image_dir_list = list()
    ground_truth_label_dir_list = list()
    inference_result_dir_list = list()

    if 'test' in test_and_fn:
        ground_truth_image_dir_list.append(os.path.join(test_data_dir, 'images_wo_border'))
        ground_truth_label_dir_list.append(os.path.join(test_data_dir, 'labels_wo_border'))
        inference_result_dir_list.append(os.path.join(test_data_dir, '{}_inference_result'.format(model_output_version)))

    if 'fn' in test_and_fn:
        ground_truth_image_dir_list.append(os.path.join(retrain_data_val_dir, 'images_random_crop'))
        ground_truth_label_dir_list.append(os.path.join(retrain_data_val_dir, 'labels_random_crop'))
        inference_result_dir_list.append(os.path.join(retrain_data_val_dir, '{}_inference_result'.format(model_output_version)))

    test_and_fn_data_dict = {
        'model_type': model_type,
        'ground_truth_image_dir_list': ground_truth_image_dir_list,
        'ground_truth_label_dir_list': ground_truth_label_dir_list,
        'inference_result_dir_list': inference_result_dir_list,
        'model_iter_list': model_iter_list,
        'score_threshold': None,        # only for evaluation_through_model_iter_by_score_threshold()
        'fp_rate_threshold': fp_rate,   # only for evaluation_through_model_iter_by_fp_rate()
        'return_type': return_type,     # only for evaluation_through_model_iter_by_fp_rate()
        'save_fn_bbox': False,
    }

    test_best_model_iter_list = evaluation_through_model_iter_by_fp_rate(test_and_fn_data_dict)
    return test_best_model_iter_list

def evaluate_new_models_on_test_and_fn_by_score_threshold(model_type, test_best_model_iter_list, score_threshold):
    test_data_dir = config['test_data_dir']
    retrain_data_val_dir = config['retrain_data_val_dir']

    if model_type == 'RetinaNet':
        model_output_version = config['retinanet_model_output_version']
    elif model_type == 'CenterNet2':
        model_output_version = config['centernet2_model_output_version']
    elif model_type == 'YOLOv4':
        model_output_version = config['yolov4_model_output_version']
    else:
        assert False, "model_type = {} is not supported".format(model_type)

    test_and_fn_data_dict = {
        'model_type': model_type,
        'ground_truth_image_dir_list': [os.path.join(test_data_dir, 'images_wo_border'),
                                        os.path.join(retrain_data_val_dir, 'images_random_crop')],
        'ground_truth_label_dir_list': [os.path.join(test_data_dir, 'labels_wo_border'),
                                        os.path.join(retrain_data_val_dir, 'labels_random_crop')],
        'inference_result_dir_list': [os.path.join(test_data_dir, '{}_inference_result'.format(model_output_version)),
                                      os.path.join(retrain_data_val_dir, '{}_inference_result'.format(model_output_version))],
        'model_iter_list': test_best_model_iter_list,
        'score_threshold': score_threshold, # only for evaluation_through_model_iter_by_score_threshold()
        'fp_rate_threshold': None,          # only for evaluation_through_model_iter_by_fp_rate()
        'return_type': None,                # only for evaluation_through_model_iter_by_fp_rate()
        'save_fn_bbox': False
    }

    eval_result_list = evaluation_through_model_iter_by_score_threshold(test_and_fn_data_dict)
    return eval_result_list

def evaluation_models_by_fp_rate(label_dir_list, new_model_file_dir_list, refine_num_list, fp_rate_threshold=0.6):
    global logger
    log_file_path = os.path.join(aoi_dir, 'validation', log_file_name)
    os_remove(log_file_path)
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    logger = get_logger(name=__file__, console_handler_level=logging.INFO, file_handler_level=logging.INFO, file_name=log_file_name)

    test_data_dir = config['test_data_dir']
    retrain_data_val_dir = config['retrain_data_val_dir']

    score_threshold_list = list()
    for label_dir, new_model_file_dir, refine_num in zip(label_dir_list, new_model_file_dir_list, refine_num_list):
        model_type = os.path.basename(new_model_file_dir)
        model_iter = os.path.basename(os.path.dirname(label_dir[0]))
        test_and_fn_data_dict = {
            'model_type': model_type,
            'ground_truth_image_dir_list': [os.path.join(test_data_dir, 'images_wo_border'),
                                            os.path.join(retrain_data_val_dir, 'images_random_crop')],
            'ground_truth_label_dir_list': [os.path.join(test_data_dir, 'labels_wo_border'),
                                            os.path.join(retrain_data_val_dir, 'labels_random_crop')],
            'inference_result_label_dir_list': [label_dir[0],
                                                label_dir[1]],
            'model_iter': model_type + ' / ' + model_iter,
            'fp_rate_threshold': fp_rate_threshold,
            'save_fn_bbox': False
        }

        eval_result = evaluation_by_fp_rate(test_and_fn_data_dict, refine_num=refine_num)
        score_threshold_list.append(eval_result['score_threshold'])

    return log_file_path, score_threshold_list

if __name__ == '__main__':
    # # # # # # # # # # # # # # # # # # # # #
    #   Evaluation YoloV4 inference result  #
    # # # # # # # # # # # # # # # # # # # # #
    '''
    test_data_dir = config['test_data_dir']
    retrain_data_val_dir = config['retrain_data_val_dir']

    base_dir = '/home/aoi/AOI_PCB_Retrain_detectron2/YOLOv4/result/after_retrain/test'
    model_dir_list = os.listdir(base_dir)
    for model_dir_name in model_dir_list:
        if os.path.isdir(os.path.join(base_dir, model_dir_name)):
            test_label_dir = os.path.join(base_dir, model_dir_name, 'test_label_json_dir')
            fn_label_dir = os.path.join(base_dir, model_dir_name, 'fn_label_json_dir')

            test_and_fn_data_dict = {
                'ground_truth_image_dir_list': [os.path.join(test_data_dir, 'images_wo_border'),
                                                os.path.join(retrain_data_val_dir, 'images_random_crop')],
                'ground_truth_label_dir_list': [os.path.join(test_data_dir, 'labels_wo_border'),
                                                os.path.join(retrain_data_val_dir, 'labels_random_crop')],
                'inference_result_label_dir_list': [test_label_dir, fn_label_dir],
                'model_iter': model_dir_name,
                'fp_rate_threshold': 0.6,
                'save_fn_bbox': False
            }

            eval_result = evaluation_by_fp_rate(test_and_fn_data_dict, refine_num=101)
    '''

    # # # # # # # # # # # # # # # # # # #
    #   Evaluation inference result     #
    # # # # # # # # # # # # # # # # # # #
    '''
    ground_truth_image_dir_list = list()
    ground_truth_label_dir_list = list()
    inference_result_label_dir_list = list()
    ground_truth_image_dir_list.append('/home/aoi/AOI_PCB_Retrain_detectron2/inference_result/Week_47_image')
    ground_truth_label_dir_list.append('/home/aoi/AOI_PCB_Retrain_detectron2/inference_result/Week_47_gt_label')
    inference_result_label_dir_list.append('/home/aoi/AOI_PCB_Retrain_detectron2/inference_result/Week_47_dt_label')

    eval_result = OrderedDict()
    eval_dict = OrderedDict()
    empty_all_statistics()
    score_threshold = 0.01
    compare_ground_truth_with_inference_result( ground_truth_image_dir_list, ground_truth_label_dir_list, \
                                                inference_result_label_dir_list, score_threshold,
                                                eval_result=eval_result, eval_dict=eval_dict, save_fn_bbox=False)
    '''

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Evaluation inference result for a specific FP rate          #
    #   label_dir = config['centernet2_best_model_label_dir'] /     #
    #               config['retinanet_best_model_label_dir'] /      #
    #               config['yolov4_best_model_label_dir']           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # '''
    label_dir_list = [  config['centernet2_best_model_label_dir'], \
                        config['retinanet_best_model_label_dir'], \
                        config['yolov4_best_model_label_dir'] ]
    new_model_file_dir_list = [ config['centernet2_new_model_file_dir'], \
                                config['retinanet_new_model_file_dir'], \
                                config['yolov4_new_model_file_dir'] ]
    refine_num_list = [51, 51, 101]
    evaluation_models_by_fp_rate(label_dir_list, new_model_file_dir_list, refine_num_list, fp_rate_threshold=0.7)
    # '''

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Evaluation through model iteration for a specific FP rate   #
    #   model_type = 'CenterNet2' / 'RetinaNet' / 'YOLOv4'          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    '''
    evaluate_old_model_on_test_and_fn_by_fp_rate(model_type='CenterNet2', fp_rate=0.6, return_type='eval_result')

    evaluate_new_models_on_test_and_fn_by_fp_rate(model_type='RetinaNet', model_iter_list=None, \
                                                  fp_rate=0.6, return_type='eval_result', test_and_fn = ['test', 'fn'])
    '''

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Evaluation through model iteration for a specific threshold   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    '''
    model_iter_list = ['model_0035999']
    evaluate_new_models_on_test_and_fn_by_score_threshold('RetinaNet', model_iter_list, 0.18)
    '''