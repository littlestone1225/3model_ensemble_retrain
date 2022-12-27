#!/usr/bin/env python3
import os, sys
from datetime import datetime
import cv2
from operator import add
import torch
import time
import logging
import math
import numpy as np
from multiprocessing import Process, Queue, Pipe
import SharedArray



aoi_dir_name = os.getenv('AOI_DIR_NAME')
assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"

current_dir = os.path.dirname(os.path.abspath(__file__))
idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
aoi_dir = current_dir[:idx]

sys.path.append(os.path.join(aoi_dir, "config"))
from config import read_config_yaml

sys.path.append(os.path.join(aoi_dir, "utils"))
from utils import os_makedirs, shutil_move
from logger import get_logger

sys.path.append(os.path.join(aoi_dir, "data_preprocess"))
from remove_black_border import remove_black_border
from crop_small_image import crop_sliding_window
from csv_json_conversion import csv_to_json, json_to_bbox

sys.path.append(os.path.join(aoi_dir, "validation"))
from nms import non_max_suppression_slow
from validation import unify_batch_predictor_output

sys.path.append(os.path.join(aoi_dir, "ensemble"))
from ensemble import ensemble

sys.path.append(os.path.join(aoi_dir, "YOLOv4"))
import darknet
from YOLOv4.darknet_inference import image_detection, batch_detection

sys.path.append(os.path.join(aoi_dir, "detectron2/projects/CenterNet2"))
from centernet.config import add_centernet_config

sys.path.append(os.path.join(aoi_dir, "detectron2"))
from detectron2.config import get_cfg
from detectron2.engine import BatchPredictor

# Get logger
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
logger = get_logger(name=__file__, console_handler_level=logging.INFO, file_handler_level=None)

# Read config_file
config_file = os.path.join(aoi_dir, 'config/config.yaml')
config = read_config_yaml(config_file)

num_to_category_dict = {0: 'bridge', 1: 'appearance_less', 2: 'excess_solder', 3: 'appearance'}
category_to_color_dict = {'bridge': [0, 0, 255], 'appearance_less': [255,191,0], 'excess_solder': [221,160,221], 'appearance': [0,165,255]}
default_color = [0, 255, 0] # in BGR

# A5000 (8315 MB / 24256 MB)
# centernet2_batch_size = 9 # 2723(MB)
# retinanet_batch_size = 11 # 2995(MB)
# yolov4_batch_size = 1     # 2597(MB)

# # A5000 (7145 MB / 24256 MB) / P4 (4099 MB / 7611 MB)
# centernet2_batch_size = 4 # A5000: 2309(MB) / P4: 1187(MB)
# retinanet_batch_size = 4  # A5000: 2239(MB) / P4: 1241(MB)
# yolov4_batch_size = 1     # A5000: 2597(MB) / P4: 1669(MB)

# P4 (7469 MB / 7611 MB)
centernet2_batch_size = 16 # 16 => P4: 1999(MB) / 32 => P4: 3067(MB)
retinanet_batch_size = 24  # P4: 2731(MB) ~ 3577(MB) why?
yolov4_batch_size = 1      # P4: 1669(MB)

keep_exists = False
use_centernet2 = True
use_retinanet = True
use_yolov4 = True

def dequeue_image_file(input_dict, image_file_queue, ensemble_parent_conn, models_parent_conn, gpu_num, p_id):
    image_dir = input_dict['image_dir']
    image_wo_border_dir = input_dict['image_wo_border_dir']

    if use_centernet2:
        centernet2_parent_conn = models_parent_conn["centernet2"]
    if use_retinanet:
        retinanet_parent_conn = models_parent_conn['retinanet']
    if use_yolov4:
        yolov4_parent_conn = models_parent_conn['yolov4']

    while 1:
        image_file_name = image_file_queue.get()
        if image_file_name == 'EOF':
            if use_centernet2:
                logger.debug("[dequeue_image_file] gpu_num = {}; centernet2_parent_conn[{}].send('EOF')".format(gpu_num, p_id))
                centernet2_parent_conn[gpu_num].send(['EOF', 'EOF', 'EOF'])
            if use_retinanet:
                logger.debug("[dequeue_image_file] gpu_num = {}; retinanet_parent_conn[{}].send('EOF')".format(gpu_num, p_id))
                retinanet_parent_conn[gpu_num].send(['EOF', 'EOF', 'EOF'])
            if use_yolov4:
                logger.debug("[dequeue_image_file] gpu_num = {}; yolov4_parent_conn[{}].send('EOF')".format(gpu_num, p_id))
                yolov4_parent_conn[gpu_num].send(['EOF', 'EOF', 'EOF'])
            logger.info("[dequeue_image_file] gpu_num = {}; break".format(gpu_num))
            break

        start_time = time.time()
        image_file_path = os.path.join(image_dir, image_file_name)
        image = cv2.imread(image_file_path)

        if image_wo_border_dir:
            # Save the without-border image to image_wo_border_dir
            xmin, ymin, xmax, ymax = remove_black_border(image, None)
            image_wo_border = image[ymin:ymax,xmin:xmax]
            image_wo_border_file_path = os.path.join(image_wo_border_dir, image_file_name)
            cv2.imwrite(image_wo_border_file_path, image_wo_border, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            crop_rect_list, crop_image_list = crop_sliding_window(image_wo_border)
            logger.info("[dequeue_image_file] gpu_num = {}; image_file_name = {}; sliding crop = {:>3d} {}; time = {:4.3f} s".\
                        format(gpu_num, image_file_name, len(crop_rect_list), image_wo_border.shape[:2], round(time.time()-start_time, 3)))
        else:
            crop_rect_list, crop_image_list = crop_sliding_window(image)
            logger.info("[dequeue_image_file] gpu_num = {}; image_file_name = {}; sliding crop = {:>3d} {}; time = {:4.3f} s".\
                        format(gpu_num, image_file_name, len(crop_rect_list), image.shape[:2], round(time.time()-start_time, 3)))

        crop_rect_arr = np.array(crop_rect_list, np.int32)
        crop_image_arr = np.array(crop_image_list, np.uint8)

        logger.debug("[dequeue_image_file] gpu_num = {};  ensemble_parent_conn[{}].recv()".format(gpu_num, p_id))
        ensemble_parent_conn[p_id].recv()

        start_time = time.time()
        try:
            crop_rect_shared_arr = SharedArray.create("shm://crop_rect_{}".format(p_id), crop_rect_arr.shape)
            crop_image_shared_arr = SharedArray.create("shm://crop_image_{}".format(p_id), crop_image_arr.shape)
        except:
            SharedArray.delete("shm://crop_rect_{}".format(p_id))
            SharedArray.delete("shm://crop_image_{}".format(p_id))
            crop_rect_shared_arr = SharedArray.create("shm://crop_rect_{}".format(p_id), crop_rect_arr.shape)
            crop_image_shared_arr = SharedArray.create("shm://crop_image_{}".format(p_id), crop_image_arr.shape)

        crop_rect_shared_arr[:] = crop_rect_arr[:]
        crop_image_shared_arr[:] = crop_image_arr[:]
        logger.info("[dequeue_image_file] gpu_num = {}; image_file_name = {}; save to shm = {:>3d}; time = {:4.3f} s".\
                    format(gpu_num, image_file_name, len(crop_rect_list), round(time.time()-start_time, 3)))

        if use_centernet2:
            logger.debug("[dequeue_image_file] gpu_num = {}; centernet2_parent_conn[{}].send()".format(gpu_num, p_id))
            if image_wo_border_dir:
                centernet2_parent_conn[p_id].send([image_wo_border_dir, image_file_name, image_wo_border.shape])
            else:
                centernet2_parent_conn[p_id].send([image_dir, image_file_name, image.shape])
        if use_retinanet:
            logger.debug("[dequeue_image_file] gpu_num = {}; retinanet_parent_conn[{}].send()".format(gpu_num, p_id))
            if image_wo_border_dir:
                retinanet_parent_conn[p_id].send([image_wo_border_dir, image_file_name, image_wo_border.shape])
            else:
                retinanet_parent_conn[p_id].send([image_dir, image_file_name, image.shape])
        if use_yolov4:
            logger.debug("[dequeue_image_file] gpu_num = {}; yolov4_parent_conn[{}].send()".format(gpu_num, p_id))
            if image_wo_border_dir:
                yolov4_parent_conn[p_id].send([image_wo_border_dir, image_file_name, image_wo_border.shape])
            else:
                yolov4_parent_conn[p_id].send([image_dir, image_file_name, image.shape])

def run_centernet2(input_dict, centernet2_child_conn, gpu_num, p_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_num)
    centernet2_label_dir = input_dict['centernet2_label_dir']

    # Initialize model
    yaml_file_path = config['centernet2_yaml_file_path']
    model_file_path = config['centernet2_best_model_file_path']

    cfg = get_cfg()
    cfg.defrost()
    add_centernet_config(cfg)
    cfg.merge_from_file(yaml_file_path)
    cfg.MODEL.WEIGHTS = model_file_path
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    cfg.freeze()

    # Initialize predictor
    predictor = BatchPredictor(cfg)

    while 1:
        logger.debug("[run_centernet2] gpu_num = {}; centernet2_child_conn[{}].recv()".format(gpu_num, p_id))
        image_wo_border_dir, image_file_name, image_shape = centernet2_child_conn[gpu_num].recv()

        if image_file_name == 'EOF':
            logger.debug("[run_centernet2] gpu_num = {}; centernet2_child_conn[{}].send('EOF')".format(gpu_num, p_id))
            centernet2_child_conn[p_id].send(['EOF', 'EOF'])
            logger.info("[run_centernet2] gpu_num = {}; break".format(gpu_num))
            break

        start_time = time.time()
        crop_rect_arr = SharedArray.attach("shm://crop_rect_{}".format(p_id))
        crop_image_arr = SharedArray.attach("shm://crop_image_{}".format(p_id))

        crop_rect_arr = crop_rect_arr.astype(np.int32)
        crop_image_arr = crop_image_arr.astype(np.uint8)

        # Do the batch inference on crop_image_list to get bbox_list
        outputs = list()
        total_crop_image = crop_image_arr.shape[0]
        logger.debug("[run_centernet2] gpu_num = {}; total_crop_image = {}".format(gpu_num, total_crop_image))
        if total_crop_image > centernet2_batch_size:
            for i in range(math.ceil(total_crop_image / centernet2_batch_size)):
                start = i*centernet2_batch_size
                end = (i+1)*centernet2_batch_size
                if end > total_crop_image:
                    end = total_crop_image
                # logger.debug("[run_centernet2] gpu_num = {}; start = {} ; end = {}".format(gpu_num, start, end))
                out = predictor(crop_image_arr[start:end])
                outputs.extend(out)
        else:
            outputs = predictor(crop_image_arr)

        assert len(outputs) == crop_rect_arr.shape[0], "len(outputs) != crop_rect_arr.shape[0]"
        bbox_list = unify_batch_predictor_output(image_file_name, image_shape, crop_rect_arr, outputs)
        bbox_list.sort(key = lambda bbox: bbox[-1], reverse=True)
        bbox_list = non_max_suppression_slow(bbox_list, 0.5)

        # Convert bbox_list to json for labelme visualization
        csv_to_json(bbox_list, image_wo_border_dir, centernet2_label_dir, coord_type="xmin_ymin_xmax_ymax", store_score=True)
        logger.info("[run_centernet2] gpu_num = {}; run_centernet2 time = {:4.3f} s".format(gpu_num, round(time.time()-start_time, 3)))

        json_file_name = os.path.splitext(image_file_name)[0] + '.json'
        logger.debug("[run_centernet2] gpu_num = {}; centernet2_child_conn[{}].send()".format(gpu_num, p_id))
        centernet2_child_conn[p_id].send([image_wo_border_dir, json_file_name])

def run_retinanet(input_dict, retinanet_child_conn, gpu_num, p_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_num)
    retinanet_label_dir = input_dict['retinanet_label_dir']

    # Initialize model
    yaml_file_path = config['retinanet_yaml_file_path']
    model_file_path = config['retinanet_best_model_file_path']

    cfg = get_cfg()
    cfg.defrost()
    cfg.merge_from_file(yaml_file_path)
    cfg.MODEL.WEIGHTS = model_file_path
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.1
    cfg.freeze()

    # Initialize predictor
    predictor = BatchPredictor(cfg)

    while 1:
        logger.debug("[run_retinanet] gpu_num = {}; retinanet_child_conn[{}].recv()".format(gpu_num, p_id))
        image_wo_border_dir, image_file_name, image_shape = retinanet_child_conn[gpu_num].recv()

        if image_file_name == 'EOF':
            logger.debug("[run_retinanet] gpu_num = {}; retinanet_child_conn[{}].send('EOF')".format(gpu_num, p_id))
            retinanet_child_conn[p_id].send(['EOF', 'EOF'])
            break

        start_time = time.time()
        crop_rect_arr = SharedArray.attach("shm://crop_rect_{}".format(p_id))
        crop_image_arr = SharedArray.attach("shm://crop_image_{}".format(p_id))

        crop_rect_arr = crop_rect_arr.astype(np.int32)
        crop_image_arr = crop_image_arr.astype(np.uint8)

        # Do the batch inference on crop_image_list to get bbox_list
        outputs = list()
        total_crop_image = crop_image_arr.shape[0]
        logger.debug("[run_retinanet] gpu_num = {}; total_crop_image = {}".format(gpu_num, total_crop_image))
        if total_crop_image > retinanet_batch_size:
            for i in range(math.ceil(total_crop_image / retinanet_batch_size)):
                start = i*retinanet_batch_size
                end = (i+1)*retinanet_batch_size
                if end > total_crop_image:
                    end = total_crop_image
                # logger.debug("gpu_num = {}; start = {} ; end = {}".format(gpu_num, start, end))
                out = predictor(crop_image_arr[start:end])
                outputs.extend(out)
        else:
            outputs = predictor(crop_image_arr)

        assert len(outputs) == crop_rect_arr.shape[0], "len(outputs) != crop_rect_arr.shape[0]"
        bbox_list = unify_batch_predictor_output(image_file_name, image_shape, crop_rect_arr, outputs)
        bbox_list.sort(key = lambda bbox: bbox[-1], reverse=True)
        bbox_list = non_max_suppression_slow(bbox_list, 0.5)

        # Convert bbox_list to json for labelme visualization
        csv_to_json(bbox_list, image_wo_border_dir, retinanet_label_dir, coord_type="xmin_ymin_xmax_ymax", store_score=True)
        logger.info("[run_retinanet] gpu_num = {}; run_retinanet time = {:4.3f} s".format(gpu_num, round(time.time()-start_time, 3)))

        json_file_name = os.path.splitext(image_file_name)[0] + '.json'
        logger.debug("[run_retinanet] gpu_num = {}; retinanet_child_conn[{}].send()".format(gpu_num, p_id))
        retinanet_child_conn[p_id].send([image_wo_border_dir, json_file_name])

def run_yolov4(input_dict, yolov4_child_conn, gpu_num, p_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_num)
    yolov4_label_dir = input_dict['yolov4_label_dir']

    score_threshold = 0.01
    edge_limit = 20 # pixels

    # load model
    network, class_names, class_colors = darknet.load_network( config['yolov4_cfg_file_path'], config['yolov4_data_file_path'], 
                                                               config['yolov4_best_model_file_path'], yolov4_batch_size )

    while 1:
        logger.debug("[run_yolov4] gpu_num = {}; yolov4_child_conn[{}].recv()".format(gpu_num, p_id))
        image_wo_border_dir, image_file_name, image_shape = yolov4_child_conn[gpu_num].recv()

        if image_file_name == 'EOF':
            logger.debug("[run_yolov4] gpu_num = {}; yolov4_child_conn[{}].send('EOF')".format(gpu_num, p_id))
            yolov4_child_conn[p_id].send(['EOF', 'EOF'])
            break

        start_time = time.time()
        crop_rect_arr = SharedArray.attach("shm://crop_rect_{}".format(p_id))
        crop_image_arr = SharedArray.attach("shm://crop_image_{}".format(p_id))

        crop_rect_arr = crop_rect_arr.astype(np.int32)
        crop_image_arr = crop_image_arr.astype(np.uint8)
        crop_image_arr = [cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB) for crop_img in crop_image_arr]


        if yolov4_batch_size == 1:
            bbox_list = image_detection(crop_image_arr, crop_rect_arr, image_file_name, network, class_names, class_colors, \
                                        score_threshold, edge_limit)
        else:
            bbox_list = batch_detection(crop_image_arr, crop_rect_arr, image_file_name, network, class_names, class_colors, \
                                        score_threshold, edge_limit, 0.5, 0.45, yolov4_batch_size)
        bbox_list = non_max_suppression_slow(bbox_list, 0.5)

        csv_to_json(bbox_list, image_wo_border_dir, yolov4_label_dir, coord_type="xmin_ymin_xmax_ymax", store_score=True)
        logger.info("[run_yolov4] gpu_num = {}; run_yolov4 time = {:4.3f} s".format(gpu_num, round(time.time()-start_time, 3)))

        json_file_name = os.path.splitext(image_file_name)[0] + '.json'
        logger.debug("[run_yolov4] gpu_num = {}; yolov4_child_conn[{}].send()".format(gpu_num, p_id))
        yolov4_child_conn[gpu_num].send([image_wo_border_dir, json_file_name])

def run_ensemble(input_dict, models_parent_conn, ensemble_child_conn, ensemble_time_parent_conn, gpu_num, p_id):
    centernet2_label_dir = input_dict['centernet2_label_dir']
    retinanet_label_dir = input_dict['retinanet_label_dir']
    yolov4_label_dir = input_dict['yolov4_label_dir']
    inference_result_label_dir = input_dict['inference_result_label_dir']
    inference_result_txt_dir = input_dict['inference_result_txt_dir']

    if use_centernet2:
        centernet2_parent_conn = models_parent_conn["centernet2"]
    if use_retinanet:
        retinanet_parent_conn = models_parent_conn['retinanet']
    if use_yolov4:
        yolov4_parent_conn = models_parent_conn['yolov4']

    center_threshold = {'3hit': 0.1, '2hit': 0.6, '1hit': 0.9}
    retina_threshold = {'3hit': 0.1, '2hit': 0.6, '1hit': 0.9}
    yolov4_threshold = {'3hit': 0.01, '2hit': 0.6, '1hit': 0.9}
    threshold = [center_threshold, retina_threshold, yolov4_threshold]
    json_idx = 1
    while 1:
        if use_centernet2:
            logger.debug("[run_ensemble] gpu_num = {}; centernet2_parent_conn[{}].recv()".format(gpu_num, p_id))
            image_wo_border_dir, json_file_name = centernet2_parent_conn[p_id].recv()
        if use_retinanet:
            logger.debug("[run_ensemble] gpu_num = {}; retinanet_parent_connn[{}].recv()".format(gpu_num, p_id))
            image_wo_border_dir, json_file_name = retinanet_parent_conn[p_id].recv()
        if use_yolov4:
            logger.debug("[run_ensemble] gpu_num = {}; yolov4_parent_conn[{}].recv()".format(gpu_num, p_id))
            image_wo_border_dir, json_file_name = yolov4_parent_conn[p_id].recv()

        if json_file_name == 'EOF':
            logger.info("[run_ensemble] gpu_num = {}; ensemble_time_parent_conn[{}].send('EOF')".format(gpu_num, p_id))
            ensemble_time_parent_conn[p_id].send('EOF')
            logger.info("[run_ensemble] gpu_num = {}; break".format(gpu_num))
            break

        logger.debug("[run_ensemble] gpu_num = {}; ensemble_child_conn[{}].send('Ready')".format(gpu_num, p_id))
        ensemble_child_conn[p_id].send("Ready")

        start_time = time.time()
        ensemble(json_idx, image_wo_border_dir, centernet2_label_dir, retinanet_label_dir, yolov4_label_dir, \
                 inference_result_label_dir, json_file_name, threshold, dashboard_txt_dir=inference_result_txt_dir)
        logger.info("[run_ensemble] gpu_num = {}; run_ensemble time = {:4.3f} s".format(gpu_num, round(time.time()-start_time, 3)))
        ensemble_time_parent_conn[p_id].send(json_file_name)

def draw_bbox(input_dict, json_file_name, result_image_dir):
    image_dir = input_dict['image_dir']
    image_wo_border_dir = input_dict['image_wo_border_dir']
    inference_result_label_dir = input_dict['inference_result_label_dir']
    inference_result_image_dir = input_dict['inference_result_image_dir']

    json_file_path = os.path.join(inference_result_label_dir, json_file_name)
    image_file_name = os.path.splitext(json_file_name)[0] + '.jpg'
    if image_wo_border_dir:
        image_file_path = os.path.join(image_wo_border_dir, image_file_name)
    else:
        image_file_path = os.path.join(image_dir, image_file_name)

    if os.path.isfile(json_file_path):
        bbox_list = json_to_bbox(json_file_path, store_score=True)
    else:
        bbox_list = list()
    image = cv2.imread(image_file_path)

    for bbox in bbox_list:
        xmin, ymin, xmax, ymax = bbox[2:6]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 4)
        '''
        score = bbox[6]
        center_x, center_y = (xmin + xmax)//2, (ymin + ymax)//2
        w, h = xmax-xmin, ymax-ymin
        overlay = image.copy()
        alpha = 0.4
        if score < 0.5:
            cv2.ellipse(overlay, (center_x, center_y), (w // 2 + 80, h // 2 + 80), 0, 0, 360, (25, 25, 255), -1)
        if score > 0.5 and score < 0.8:
            cv2.ellipse(overlay, (center_x, center_y), (w // 2 + 80, h // 2 + 80), 0, 0, 360, (25, 25, 255), -1)
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
        if score > 0.8:
            cv2.ellipse(overlay, (center_x, center_y), (w // 2 + 80, h // 2 + 80), 0, 0, 360, (25, 25, 255), -1)
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (255, 0, 255), -1)
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
        cv2.addWeighted(overlay, alpha, image, 1-alpha, 0, image)
        cv2.addWeighted(overlay, alpha, image, 1-alpha, 0, image)
        '''
    result_image_file_path = os.path.join(inference_result_image_dir, image_file_name)
    cv2.imwrite(result_image_file_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def measure_ensemble_time(input_dict, ensemble_time_child_conn, gpu_num, p_id):
    image_dir = input_dict['image_dir']
    image_backup_dir = input_dict['image_backup_dir']
    inference_result_image_dir = input_dict['inference_result_image_dir']

    start_time = time.time()
    while 1:
        json_file_name = ensemble_time_child_conn[p_id].recv()
        if json_file_name == 'EOF':
            logger.info("[measure_ensemble_time] gpu_num = {}; break".format(gpu_num))
            break
        end_time = time.time()
        logger.info("")
        logger.info("gpu_num = {}; ensemble time measure= {:4.3f} s".format(gpu_num, round(end_time-start_time, 3)))
        logger.info("")
        start_time = end_time

        draw_bbox(input_dict, json_file_name, inference_result_image_dir)

        if image_backup_dir:
            image_file_name = os.path.splitext(json_file_name)[0] + '.jpg'
            image_file_path = os.path.join(image_dir, image_file_name)
            shutil_move(image_file_path, image_backup_dir)

def inference(input_dict):
    gpu_list = config['gpu_list']
    gpu_nums = len(gpu_list)

    image_dir = input_dict['image_dir']
    centernet2_label_dir = input_dict['centernet2_label_dir']
    retinanet_label_dir = input_dict['retinanet_label_dir']
    yolov4_label_dir = input_dict['yolov4_label_dir']
    inference_result_image_dir = input_dict['inference_result_image_dir']
    inference_result_label_dir = input_dict['inference_result_label_dir']
    inference_result_txt_dir = input_dict['inference_result_txt_dir']
    image_wo_border_dir = input_dict['image_wo_border_dir']
    image_backup_dir = input_dict['image_backup_dir']
    inference_result_image_backup_dir = input_dict['inference_result_image_backup_dir']

    os_makedirs(image_dir, keep_exists=True)
    os_makedirs(centernet2_label_dir, keep_exists)
    os_makedirs(retinanet_label_dir, keep_exists)
    os_makedirs(yolov4_label_dir, keep_exists)
    os_makedirs(inference_result_image_dir, keep_exists)
    os_makedirs(inference_result_label_dir, keep_exists)
    os_makedirs(inference_result_txt_dir, keep_exists)

    if image_wo_border_dir:
        os_makedirs(image_wo_border_dir, keep_exists)
    if image_backup_dir:
        os_makedirs(image_backup_dir, keep_exists)
    if inference_result_image_backup_dir:
        os_makedirs(inference_result_image_backup_dir, keep_exists)

    image_file_queue = Queue(5)
    centernet2_parent_conn      = [None]*gpu_nums
    centernet2_child_conn       = [None]*gpu_nums
    retinanet_parent_conn       = [None]*gpu_nums
    retinanet_child_conn        = [None]*gpu_nums
    yolov4_parent_conn          = [None]*gpu_nums
    yolov4_child_conn           = [None]*gpu_nums
    ensemble_parent_conn        = [None]*gpu_nums
    ensemble_child_conn         = [None]*gpu_nums
    ensemble_time_parent_conn   = [None]*gpu_nums
    ensemble_time_child_conn    = [None]*gpu_nums

    p_dequeue_image_file        = [None]*gpu_nums
    p_centernet2                = [None]*gpu_nums
    p_retinanet                 = [None]*gpu_nums
    p_yolov4                    = [None]*gpu_nums
    p_ensemble                  = [None]*gpu_nums
    p_ensemble_time             = [None]*gpu_nums

    for p_id, gpu_num in enumerate(gpu_list):
        centernet2_parent_conn[p_id], centernet2_child_conn[p_id] = Pipe()
        retinanet_parent_conn[p_id], retinanet_child_conn[p_id] = Pipe()
        yolov4_parent_conn[p_id], yolov4_child_conn[p_id] = Pipe()
        ensemble_parent_conn[p_id], ensemble_child_conn[p_id] = Pipe()
        ensemble_time_parent_conn[p_id], ensemble_time_child_conn[p_id] = Pipe()

        models_parent_conn = {'centernet2': centernet2_parent_conn, 'retinanet': retinanet_parent_conn, 'yolov4': yolov4_parent_conn}

        p_dequeue_image_file[p_id] = Process(target=dequeue_image_file, name='p_dequeue_image_file[{}]'.format(p_id),
                                                args=(input_dict, image_file_queue, ensemble_parent_conn, models_parent_conn, gpu_num, p_id))
        p_dequeue_image_file[p_id].start()

        if use_centernet2:
            p_centernet2[p_id] = Process(target=run_centernet2, name='p_centernet2[{}]'.format(p_id),
                                            args=(input_dict, centernet2_child_conn, gpu_num, p_id))
            p_centernet2[p_id].start()

        if use_retinanet:
            p_retinanet[p_id] = Process(target=run_retinanet, name='p_retinanet[{}]'.format(p_id),
                                           args=(input_dict, retinanet_child_conn, gpu_num, p_id))
            p_retinanet[p_id].start()

        if use_yolov4:
            p_yolov4[p_id] = Process(target=run_yolov4, name='p_yolov4[{}]'.format(p_id),
                                        args=(input_dict, yolov4_child_conn, gpu_num, p_id))
            p_yolov4[p_id].start()

        p_ensemble[p_id] = Process(target=run_ensemble, name='p_ensemble[{}]'.format(p_id),
                                      args=(input_dict, models_parent_conn, ensemble_child_conn, ensemble_time_parent_conn, gpu_num, p_id))
        p_ensemble[p_id].start()

        p_ensemble_time[p_id] = Process(target=measure_ensemble_time, name='p_ensemble_time[{}]'.format(p_id),
                                           args=(input_dict, ensemble_time_child_conn, gpu_num, p_id))
        p_ensemble_time[p_id].start()

        logger.debug("ensemble_child_conn[{}].send('Ready')".format(p_id))
        ensemble_child_conn[p_id].send("Ready")

    for idx, image_file_name in enumerate(os.listdir(image_dir)):
        image_file_queue.put(image_file_name)
        print("idx = {} ; image_file_name = {}".format(idx+1, image_file_name))

    for gpu_num in gpu_list:
        image_file_queue.put("EOF")

    for gpu_num in gpu_list:
        p_dequeue_image_file[p_id].join()
        if use_centernet2:
            p_centernet2[p_id].join()
        if use_retinanet:
            p_retinanet[p_id].join()
        if use_yolov4:
            p_yolov4[p_id].join()
        p_ensemble[p_id].join()
        p_ensemble_time[p_id].join()


if __name__ == '__main__':
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
