#!/usr/bin/env python3
import os, sys
import numpy as np
import logging
import time
from multiprocessing import Pool, Lock

aoi_dir_name = os.getenv('AOI_DIR_NAME')
assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"

current_dir = os.path.dirname(os.path.abspath(__file__))
idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
aoi_dir = current_dir[:idx]

sys.path.append(os.path.join(aoi_dir, "data_preprocess"))
from csv_json_conversion import csv_to_dashboard_txt, csv_to_json, json_to_bbox

sys.path.append(os.path.join(aoi_dir, "validation"))
from evaluation import get_best_infer_candidate
from nms import non_max_suppression_slow

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
logger = get_logger(name=__file__, console_handler_level=logging.WARNING, file_handler_level=None)

def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%06d" % (data_head, data_secs)
    return time_stamp

def get_bbox_overlap_area(dt_bbox_1, dt_bbox_2): # [file_name, error_type, xmin, ymin, xmax, ymax, score]
    dt_bbox_1 = dt_bbox_1[2:6]
    dt_bbox_2 = dt_bbox_2[2:6]
    x1 = max(min(dt_bbox_1[0], dt_bbox_1[2]), min(dt_bbox_2[0], dt_bbox_2[2]))
    y1 = max(min(dt_bbox_1[1], dt_bbox_1[3]), min(dt_bbox_2[1], dt_bbox_2[3]))
    x2 = min(max(dt_bbox_1[0], dt_bbox_1[2]), max(dt_bbox_2[0], dt_bbox_2[2]))
    y2 = min(max(dt_bbox_1[1], dt_bbox_1[3]), max(dt_bbox_2[1], dt_bbox_2[3]))
    # overlap area
    if x1 < x2 and y1 < y2:
        # area
        return (x2 - x1) * (y2 - y1)
    # no overlap
    return 0

lock = Lock()
def ensemble(json_idx, image_dir, center_json_dir, retina_json_dir, yolov4_json_dir, \
             ensemble_json_dir, json_file_name, threshold, dashboard_txt_dir=None):
    logger.debug('{} {:>4d}, {:>50s} {}'.format('= '*10, json_idx+1, json_file_name, '= '*10))

    center_threshold, retina_threshold, yolov4_threshold = threshold

    # [[file_name, error_type, xmin, ymin, xmax, ymax, score], ...]
    center_json_file_path = os.path.join(center_json_dir, json_file_name)
    if os.path.isfile(center_json_file_path):
        center_bbox_list = json_to_bbox(center_json_file_path, store_score=True)
        center_bbox_list = filter(lambda bbox: bbox[6] >= center_threshold['3hit'], center_bbox_list)
        center_bbox_list = list(center_bbox_list)
        center_bbox_list.sort(key = lambda bbox: bbox[-1], reverse=True)
        center_bbox_list = non_max_suppression_slow(center_bbox_list, 0.5)
    else:
        center_bbox_list = list()

    retina_json_file_path = os.path.join(retina_json_dir, json_file_name)
    if os.path.isfile(retina_json_file_path):
        retina_bbox_list = json_to_bbox(retina_json_file_path, store_score=True)
        retina_bbox_list = filter(lambda bbox: bbox[6] >= retina_threshold['3hit'], retina_bbox_list)
        retina_bbox_list = list(retina_bbox_list)
        retina_bbox_list.sort(key = lambda bbox: bbox[-1], reverse=True)
        retina_bbox_list = non_max_suppression_slow(retina_bbox_list, 0.5)
    else:
        retina_bbox_list = list()

    yolov4_json_file_path = os.path.join(yolov4_json_dir, json_file_name)
    if os.path.isfile(yolov4_json_file_path):
        yolov4_bbox_list = json_to_bbox(yolov4_json_file_path, store_score=True)
        yolov4_bbox_list = filter(lambda bbox: bbox[6] >= yolov4_threshold['3hit'], yolov4_bbox_list)
        yolov4_bbox_list = list(yolov4_bbox_list)
        yolov4_bbox_list.sort(key = lambda bbox: bbox[-1], reverse=True)
        yolov4_bbox_list = non_max_suppression_slow(yolov4_bbox_list, 0.5)
    else:
        yolov4_bbox_list = list()


    ensemble_bbox_list = list()
    for center_bbox in list(center_bbox_list):
        logger.debug("")
        logger.debug("center_bbox = {}".format(center_bbox))
        candidate_bbox_list = list()
        candidate_bbox_list.append(center_bbox)

        center_xmin, center_ymin, center_xmax, center_ymax = center_bbox[2:6] # [xmin, ymin, xmax, ymax]
        assert center_xmin <= center_xmax, 'center_xmin > center_xmax'
        assert center_ymin <= center_ymax, 'center_ymin > center_ymax'
        center_area = (center_xmax - center_xmin) * (center_ymax - center_ymin)
        center_score = center_bbox[6]

        # RetinaNet
        retina_candidate_dict = dict()
        retina_candidate = None
        logger.debug("len(retina_bbox_list) = {}".format(len(retina_bbox_list)))
        for retina_bbox in list(retina_bbox_list):
            overlap_area = get_bbox_overlap_area(center_bbox, retina_bbox)
            if overlap_area > 0:
                overlap_area_ratio = round(overlap_area / center_area, 3)
                if retina_candidate_dict.get(overlap_area_ratio) == None:
                    retina_candidate_dict[overlap_area_ratio] = list()
                retina_candidate_dict[overlap_area_ratio].append(retina_bbox)
        if len(retina_candidate_dict) > 0:
            retina_candidate = get_best_infer_candidate(retina_candidate_dict)
            retina_bbox_list.remove(retina_candidate)
        candidate_bbox_list.append(retina_candidate)

        # Yolo V4
        yolov4_candidate_dict = dict()
        yolov4_candidate = None
        logger.debug("len(yolov4_bbox_list) = {}".format(len(yolov4_bbox_list)))
        for yolov4_bbox in list(yolov4_bbox_list):
            overlap_area = get_bbox_overlap_area(center_bbox, yolov4_bbox)
            if overlap_area > 0:
                overlap_area_ratio = round(overlap_area / center_area, 3)
                if yolov4_candidate_dict.get(overlap_area_ratio) == None:
                    yolov4_candidate_dict[overlap_area_ratio] = list()
                yolov4_candidate_dict[overlap_area_ratio].append(yolov4_bbox)
        if len(yolov4_candidate_dict) > 0:
            yolov4_candidate = get_best_infer_candidate(yolov4_candidate_dict)
            yolov4_bbox_list.remove(yolov4_candidate)
        candidate_bbox_list.append(yolov4_candidate)

        # ensemble
        num_hit = 0

        for candidate_bbox in candidate_bbox_list:
            if candidate_bbox != None:
                num_hit = num_hit + 1

        if num_hit == 3:
            ensemble_bbox_list.append(candidate_bbox_list)
        elif num_hit == 2:
            hit = False
            hit_count = 0
            for i, candidate_bbox in enumerate(candidate_bbox_list):
                if candidate_bbox != None:
                    if candidate_bbox[6] >= threshold[i]['1hit']:
                        hit = True
                        break
                    if candidate_bbox[6] >= threshold[i]['2hit']:
                        hit_count = hit_count + 1
            if hit or hit_count == 2:
                ensemble_bbox_list.append(candidate_bbox_list)
        elif num_hit == 1:
            hit = False
            for i, candidate_bbox in enumerate(candidate_bbox_list):
                if candidate_bbox != None:
                    if candidate_bbox[6] >= threshold[i]['1hit']:
                        hit = True
                    else:
                        hit = False
            if hit:
                ensemble_bbox_list.append(candidate_bbox_list)


    for retina_bbox in list(retina_bbox_list):
        logger.debug("")
        logger.debug("retina_bbox = {}".format(retina_bbox))
        candidate_bbox_list = list()
        candidate_bbox_list.append(None) # center_candidate = None
        candidate_bbox_list.append(retina_bbox)

        retina_xmin, retina_ymin, retina_xmax, retina_ymax = retina_bbox[2:6] # [xmin, ymin, xmax, ymax]
        assert retina_xmin <= retina_xmax, 'retina_xmin > retina_xmax'
        assert retina_ymin <= retina_ymax, 'retina_ymin > retina_ymax'
        retina_area = (retina_xmax - retina_xmin) * (retina_ymax - retina_ymin)
        retina_score = retina_bbox[6]

        # Yolo V4
        yolov4_candidate_dict = dict()
        yolov4_candidate = None
        logger.debug("len(yolov4_bbox_list) = {}".format(len(yolov4_bbox_list)))
        for yolov4_bbox in list(yolov4_bbox_list):
            overlap_area = get_bbox_overlap_area(retina_bbox, yolov4_bbox)
            if overlap_area > 0:
                overlap_area_ratio = round(overlap_area / retina_area, 3)
                if yolov4_candidate_dict.get(overlap_area_ratio) == None:
                    yolov4_candidate_dict[overlap_area_ratio] = list()
                yolov4_candidate_dict[overlap_area_ratio].append(yolov4_bbox)
        if len(yolov4_candidate_dict) > 0:
            yolov4_candidate = get_best_infer_candidate(yolov4_candidate_dict)
            yolov4_bbox_list.remove(yolov4_candidate)
        candidate_bbox_list.append(yolov4_candidate)

        # ensemble
        num_hit = 0
        for candidate_bbox in candidate_bbox_list:
            if candidate_bbox != None:
                num_hit = num_hit + 1

        assert num_hit <= 2, "num_hit should be less than or equal to 2"
        if num_hit == 2:
            hit = False
            hit_count = 0
            for i, candidate_bbox in enumerate(candidate_bbox_list):
                if candidate_bbox != None:
                    if candidate_bbox[6] >= threshold[i]['1hit']:
                        hit = True
                        break
                    if candidate_bbox[6] >= threshold[i]['2hit']:
                        hit_count = hit_count + 1
            if hit or hit_count == 2:
                ensemble_bbox_list.append(candidate_bbox_list)
        elif num_hit == 1:
            hit = False
            for i, candidate_bbox in enumerate(candidate_bbox_list):
                if candidate_bbox != None:
                    if candidate_bbox[6] >= threshold[i]['1hit']:
                        hit = True
                    else:
                        hit = False
            if hit:
                ensemble_bbox_list.append(candidate_bbox_list)


    for yolov4_bbox in list(yolov4_bbox_list):
        logger.debug("")
        logger.debug("yolov4_bbox = {}".format(yolov4_bbox))
        candidate_bbox_list = list()
        candidate_bbox_list.append(None) # center_candidate = None
        candidate_bbox_list.append(None)
        candidate_bbox_list.append(yolov4_bbox)

        yolov4_xmin, yolov4_ymin, yolov4_xmax, yolov4_ymax = yolov4_bbox[2:6] # [xmin, ymin, xmax, ymax]
        assert yolov4_xmin <= yolov4_xmax, 'yolov4_xmin > yolov4_xmax'
        assert yolov4_ymin <= yolov4_ymax, 'yolov4_ymin > yolov4_ymax'
        yolov4_area = (yolov4_xmax - yolov4_xmin) * (yolov4_ymax - yolov4_ymin)
        yolov4_score = yolov4_bbox[6]

        # ensemble
        num_hit = 0
        for candidate_bbox in candidate_bbox_list:
            if candidate_bbox != None:
                num_hit = num_hit + 1

        assert num_hit <= 1, "num_hit should be less than or equal to 1"

        if num_hit == 1:
            hit = False
            for i, candidate_bbox in enumerate(candidate_bbox_list):
                if candidate_bbox != None:
                    if candidate_bbox[6] >= threshold[i]['1hit']:
                        hit = True
                    else:
                        hit = False
            if hit:
                ensemble_bbox_list.append(candidate_bbox_list)


    final_bbox_list = list()
    final_num_hit = list()
    final_binary_hit = list()
    final_score = list()
    for ensemble_bbox in ensemble_bbox_list:
        num_hit = 0
        binary_hit = 0
        max_score = 0.0
        max_score_idx = 0
        score_list = list()
        for idx, bbox in enumerate(ensemble_bbox):
            if bbox != None:
                num_hit = num_hit + 1
                binary_hit = binary_hit + 10 ** (len(ensemble_bbox) - idx -1)
                score = bbox[6]
                if score > max_score:
                    max_score = score
                    max_score_idx = idx
                score_list.append(round(score, 3))
            else:
                score_list.append(0.0)
        if ensemble_bbox[max_score_idx][1] == "elh":
            ensemble_bbox[max_score_idx][1] = "appearance_less"
        final_bbox_list.append(ensemble_bbox[max_score_idx])
        final_bbox_list[-1].append(get_time_stamp())
        final_num_hit.append(num_hit)
        final_binary_hit.append(binary_hit)
        final_score.append(score_list)

    with lock:
        logger.info('')
        for idx, final_bbox in enumerate(final_bbox_list):
            logger.info('{:>4d}, {:>4d}, {:>2d}, {:>03d}, [{:>0.3f}, {:>0.3f}, {:>0.3f}], {}'.\
                        format(json_idx, idx+1, final_num_hit[idx], final_binary_hit[idx], \
                        *final_score[idx], final_bbox))

    csv_to_json(final_bbox_list, image_dir, ensemble_json_dir, coord_type="xmin_ymin_xmax_ymax", store_score=True)

    if dashboard_txt_dir!=None:
        # final_bbox_list = [[file_name, error_type, xmin, ymin, xmax, ymax, score, timestamp], ...]
        csv_to_dashboard_txt(final_bbox_list, dashboard_txt_dir, coord_type="xmin_ymin_xmax_ymax")

def concatenate_jsons(json_dir_1, json_dir_2, image_dir, concatenate_json_dir):
    os_makedirs(concatenate_json_dir)

    for idx, json_file_name in enumerate(os.listdir(json_dir_1)):
        logger.info('{:>4d}, {:>50s}'.format(idx+1, json_file_name))
        json_file_path_1 = os.path.join(json_dir_1, json_file_name)
        json_file_path_2 = os.path.join(json_dir_2, json_file_name)
        assert os.path.isfile(json_file_path_1), '{} does not exist.'.format(json_file_path_1)
        assert os.path.isfile(json_file_path_2), '{} does not exist.'.format(json_file_path_2)

        bbox_list_1 = json_to_bbox(json_file_path_1, store_score=False)
        bbox_list_2 = json_to_bbox(json_file_path_2, store_score=False)

        bbox_list = [*bbox_list_1, *bbox_list_2]
        csv_to_json(bbox_list, image_dir, concatenate_json_dir, coord_type="xmin_ymin_xmax_ymax", store_score=False)

def get_ensemble_result_on_test_and_fn(model_type, test_and_fn_list = ['test', 'fn']):
    concurrent_ensemble = True

    test_data_dir = config['test_data_dir']
    retrain_data_val_dir = config['retrain_data_val_dir']

    ground_truth_image_dir_list = list()
    ground_truth_label_dir_list = list()
    centernet2_label_dir_list = list()
    retinanet_label_dir_list = list()
    yolov4_label_dir_list = list()

    if 'test' in test_and_fn_list:
        ground_truth_image_dir_list.append(os.path.join(test_data_dir, 'images_wo_border'))
        ground_truth_label_dir_list.append(os.path.join(test_data_dir, 'labels_wo_border'))
        centernet2_label_dir_list.append(config['centernet2_{}_label_dir'.format(model_type)][0])
        retinanet_label_dir_list.append(config['retinanet_{}_label_dir'.format(model_type)][0])
        yolov4_label_dir_list.append(config['yolov4_{}_label_dir'.format(model_type)][0])
    if 'fn' in test_and_fn_list:
        ground_truth_image_dir_list.append(os.path.join(retrain_data_val_dir, 'images_random_crop'))
        ground_truth_label_dir_list.append(os.path.join(retrain_data_val_dir, 'labels_random_crop'))
        centernet2_label_dir_list.append(config['centernet2_{}_label_dir'.format(model_type)][1])
        retinanet_label_dir_list.append(config['retinanet_{}_label_dir'.format(model_type)][1])
        yolov4_label_dir_list.append(config['yolov4_{}_label_dir'.format(model_type)][1])


    ensemble_result_dir = config['ensemble_result_dir']
    center_threshold = {'3hit': 0.1, '2hit': 0.6, '1hit': 0.9}
    retina_threshold = {'3hit': 0.1, '2hit': 0.6, '1hit': 0.9}
    yolov4_threshold = {'3hit': 0.01, '2hit': 0.6, '1hit': 0.9}
    threshold = [center_threshold, retina_threshold, yolov4_threshold]
    start_time = time.time()
    for test_and_fn, ground_truth_image_dir, ground_truth_label_dir, centernet2_label_dir, retinanet_label_dir, yolov4_label_dir in \
        zip(test_and_fn_list, ground_truth_image_dir_list, ground_truth_label_dir_list, centernet2_label_dir_list, retinanet_label_dir_list, yolov4_label_dir_list):

        ensemble_label_dir = os.path.join(ensemble_result_dir, 'ensemble_result_{}'.format(model_type), test_and_fn)
        os_makedirs(ensemble_label_dir)

        if concurrent_ensemble:
            mp_pool = Pool() # (processes=1)
            for json_idx, json_file_name in enumerate(os.listdir(ground_truth_label_dir)):
                mp_pool.apply_async(ensemble, args=(json_idx, ground_truth_image_dir, centernet2_label_dir, retinanet_label_dir, yolov4_label_dir, \
                                                    ensemble_label_dir, json_file_name, threshold))
            mp_pool.close()
            mp_pool.join()
        else:
            for json_idx, json_file_name in enumerate(os.listdir(ground_truth_label_dir)):
                ensemble(json_idx, ground_truth_image_dir, centernet2_label_dir, retinanet_label_dir, yolov4_label_dir, \
                         ensemble_label_dir, json_file_name, threshold)
    logger.info("time duration = {}".format(time.time()-start_time))

if __name__ == '__main__':
    # # # # # # # # # # # # # #
    #   concatenate_jsons     #
    # # # # # # # # # # # # # #
    '''
    ensemble_dir = config['ensemble_dir']
    json_dir_1 = os.path.join(ensemble_dir, 'fp_31_0518/ensemble_result_3hit_0.1_2hit_0.6_1hit_0.9_2')
    json_dir_2 = os.path.join(ensemble_dir, 'fp_31_0518/labels')
    image_dir = os.path.join(ensemble_dir, 'fp_31_0518/images')
    concatenate_json_dir = os.path.join(ensemble_dir, 'fp_31_0518/ensemble_with_false_alarm_3hit_0.1_2hit_0.6_1hit_0.9_2')
    concatenate_jsons(json_dir_1, json_dir_2, image_dir, concatenate_json_dir)
    '''

    # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Get_ensemble_result_on_test_and_fn          #
    #       model_type = 'best_model' / 'old_model'     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # #
    get_ensemble_result_on_test_and_fn(model_type='best_model')
    get_ensemble_result_on_test_and_fn(model_type='old_model')
