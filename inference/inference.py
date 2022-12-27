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

sys.path.append(os.path.join(aoi_dir, "detectron2"))
from detectron2.config import get_cfg
from detectron2.engine import BatchPredictor
from detectron2.structures import Instances, Boxes
from projects.CenterNet2.centernet.config import add_centernet_config

sys.path.append(os.path.join(aoi_dir, "config"))
from config import read_config_yaml, write_config_yaml

sys.path.append(os.path.join(aoi_dir, "utils"))
from utils import os_makedirs, shutil_move
from logger import get_logger

sys.path.append(os.path.join(aoi_dir, "data_preprocess"))
from remove_black_border import remove_border
from crop_sliding_window import crop_sliding_window
from csv_json_conversion import csv_to_json, json_to_csv

sys.path.append(os.path.join(aoi_dir, "validation"))
from nms import non_max_suppression_slow

sys.path.append(os.path.join(aoi_dir, "ensemble"))
from ensemble import ensemble

sys.path.append(os.path.join(aoi_dir, "YOLOv4"))
import darknet
from darknet_inference import image_detection, batch_detection

# Read config_file
config_file = os.path.join(aoi_dir, 'config/config.yaml')
config = read_config_yaml(config_file)

num_to_category_dict = {0: 'bridge', 1: 'appearance_less', 2: 'excess_solder', 3: 'appearance'}
category_to_color_dict = {'bridge': [0, 0, 255], 'appearance_less': [255,191,0], 'excess_solder': [221,160,221], 'appearance': [0,165,255]}
default_color = [0, 255, 0] # in BGR

centernet2_batch_size = 18
retinanet_batch_size = 16
yolov4_batch_size = 1

keep_exists = True
use_centernet2 = True
use_retinanet = True
use_yolov4 = True

def unify_batch_predictor_output(image_file_name, image_shape, crop_rect_list, outputs):
    image_h, image_w, image_c = image_shape

    # bbox_list = [[file_name, error_type, xmin, ymin, xmax, ymax, score], ...]
    bbox_list = list()

    pred_boxes_list = list()
    scores_list = list()
    pred_classes_list = list()
    inst = Instances((image_h, image_w))
    for crop_rect, output in zip(crop_rect_list, outputs):
        crop_xmin, crop_ymin, crop_xmax, crop_ymax = crop_rect
        output_cpu = output["instances"].to("cpu")
        pred_boxes = output_cpu.pred_boxes
        scores = output_cpu.scores
        pred_classes = output_cpu.pred_classes.tolist()

        for pred_box, score, pred_class in zip(pred_boxes, scores, pred_classes):
            # crop image coordinate
            bbox_crop_coord = [int(pred) for pred in pred_box.tolist()]
            # original image coordinate
            bbox_global_coord = list(map(add, bbox_crop_coord, [crop_xmin, crop_ymin, crop_xmin, crop_ymin]))
            pred_boxes_list.append(bbox_global_coord)

            score = round(score.item(), 4)
            scores_list.append(score)

            pred_classes_list.append(pred_class)

            error_type = num_to_category_dict[pred_class]
            bbox = [image_file_name, error_type]
            bbox.extend(bbox_global_coord)
            bbox.append(score)
            bbox_list.append(bbox)

    inst.pred_boxes = Boxes(torch.tensor(pred_boxes_list))
    inst.scores = torch.tensor(scores_list)
    inst.pred_classes = torch.tensor(pred_classes_list)
    inst_dict = {'instances': inst}
    return bbox_list, inst_dict

def check_jpg_integrity(image_file_path):
    # https://stackoverflow.com/questions/46802866/how-to-detect-if-the-jpg-jpeg-image-file-is-corruptedincomplete
    for i in range(11):
        with open(image_file_path, 'rb') as f:
            # start_of_image_marker = f.read()[0:2] # b'\xff\xd8'
            end_of_image_marker = f.read()[-2:] # b'\xff\xd9'
            logger.debug("[check_jpg_integrity] image_file_name = {} ; EOI = {}".format
                        (os.path.basename(image_file_path), end_of_image_marker))
            if end_of_image_marker==b'\xff\xd9':
                return True
        if i==10:
            logger.debug("[check_jpg_integrity] fail")
            return False
        else:
            time.sleep(0.5)

def enqueue_image_file(image_file_queue):
    image_dir = config['image_dir']
    inference_result_image_dir = config['inference_result_image_dir']
    inference_result_image_backup_dir = config['inference_result_image_backup_dir']

    image_list_old = list()
    image_list_new = list()
    trigger = True

    while 1:
        image_list_new = os.listdir(image_dir)

        for image_file_name in list(image_list_old):
            if image_file_name in image_list_new:
                image_list_new.remove(image_file_name)
            else:
                image_list_old.remove(image_file_name)

        for image_file_name in image_list_new:
            image_file_path = os.path.join(image_dir, image_file_name)
            if check_jpg_integrity(image_file_path):
                image_file_queue.put(image_file_name)
                image_list_old.append(image_file_name)

        # Move image from inference_result_image_dir to inference_result_image_backup_dir at backup_time
        current_time = datetime.now()
        current_time_hm = current_time.strftime("%H:%M")

        backup_time = config['backup_time']
        backup_time = ':'.join(backup_time.split('_'))
        if current_time_hm == backup_time:
            if trigger:
                for image_file in os.listdir(inference_result_image_dir):
                    inference_result_image_path = os.path.join(inference_result_image_dir, image_file)
                    shutil_move(inference_result_image_path, inference_result_image_backup_dir)
                trigger = False
        else:
            trigger = True

        time.sleep(1)

def dequeue_image_file(image_file_queue, gpu_num):
    image_dir = config['image_dir']
    image_wo_border_dir = config['image_wo_border_dir']

    while 1:
        image_file_name = image_file_queue.get()
        start_time = time.time()
        image_file_path = os.path.join(image_dir, image_file_name)
        image = cv2.imread(image_file_path)
        logger.info("[dequeue_image_file] gpu_num = {}; image_file_name = {}".format(gpu_num, image_file_name))
        xmin, ymin, xmax, ymax = remove_border(image, None)

        if [xmin, ymin, xmax, ymax] == [0, 0, 0, 0]:
            logger.info("[dequeue_image_file] gpu_num = {}; [xmin, ymin, xmax, ymax] == [0, 0, 0, 0]".format(gpu_num))
            shutil_move(image_file_path, inference_result_image_dir)
            continue

        # Save the without-border image to image_wo_border_dir
        image_wo_border = image[ymin:ymax,xmin:xmax]
        image_wo_border_file_path = os.path.join(image_wo_border_dir, image_file_name)
        cv2.imwrite(image_wo_border_file_path, image_wo_border, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        crop_rect_list, crop_image_list = crop_sliding_window(image_wo_border)
        crop_rect_arr = np.array(crop_rect_list, np.int32)
        crop_image_arr = np.array(crop_image_list, np.uint8)
        logger.info("[dequeue_image_file] gpu_num = {}; sliding crop = {:>3d} {}; time = {:4.3f} s".\
                    format(gpu_num, len(crop_rect_list), image_wo_border.shape[:2], round(time.time()-start_time, 3)))

        logger.debug("[dequeue_image_file] gpu_num = {};  ensemble_parent_conn[{}].recv()".format(gpu_num, gpu_num))
        ensemble_parent_conn[gpu_num].recv()

        start_time = time.time()
        try:
            crop_rect_shared_arr = SharedArray.create("shm://crop_rect_{}".format(gpu_num), crop_rect_arr.shape)
            crop_image_shared_arr = SharedArray.create("shm://crop_image_{}".format(gpu_num), crop_image_arr.shape)
        except:
            SharedArray.delete("shm://crop_rect_{}".format(gpu_num))
            SharedArray.delete("shm://crop_image_{}".format(gpu_num))
            crop_rect_shared_arr = SharedArray.create("shm://crop_rect_{}".format(gpu_num), crop_rect_arr.shape)
            crop_image_shared_arr = SharedArray.create("shm://crop_image_{}".format(gpu_num), crop_image_arr.shape)

        crop_rect_shared_arr[:] = crop_rect_arr[:]
        crop_image_shared_arr[:] = crop_image_arr[:]
        logger.info("[dequeue_image_file] gpu_num = {}; save to shm = {:>3d}; time = {:4.3f} s".\
                    format(gpu_num, len(crop_rect_list), round(time.time()-start_time, 3)))

        if use_centernet2:
            logger.debug("[dequeue_image_file] gpu_num = {}; centernet2_parent_conn[{}].send()".format(gpu_num, gpu_num))
            centernet2_parent_conn[gpu_num].send([image_wo_border_dir, image_file_name, image.shape])
        if use_retinanet:
            logger.debug("[dequeue_image_file] gpu_num = {}; retinanet_parent_conn[{}].send()".format(gpu_num, gpu_num))
            retinanet_parent_conn[gpu_num].send([image_wo_border_dir, image_file_name, image.shape])
        if use_yolov4:
            logger.debug("[dequeue_image_file] gpu_num = {}; yolov4_parent_conn[{}].send()".format(gpu_num, gpu_num))
            yolov4_parent_conn[gpu_num].send([image_wo_border_dir, image_file_name, image.shape])

def run_centernet2(centernet2_child_conn, gpu_num):
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_num)

    # Initialize model
    yaml_file_path = config['centernet2_yaml_file_path']
    model_file_path = config['centernet2_model_file_path']

    cfg = get_cfg()
    add_centernet_config(cfg)
    cfg.merge_from_file(yaml_file_path)
    cfg.MODEL.WEIGHTS = model_file_path
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.1

    # Initialize predictor
    predictor = BatchPredictor(cfg)

    while 1:
        logger.debug("[run_centernet2] gpu_num = {}; centernet2_child_conn[{}].recv()".format(gpu_num, gpu_num))
        image_wo_border_dir, image_file_name, image_shape = centernet2_child_conn[gpu_num].recv()
        start_time = time.time()
        crop_rect_arr = SharedArray.attach("shm://crop_rect_{}".format(gpu_num))
        crop_image_arr = SharedArray.attach("shm://crop_image_{}".format(gpu_num))

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
        bbox_list, outputs_all = unify_batch_predictor_output(image_file_name, image_shape, crop_rect_arr, outputs)
        bbox_list.sort(key = lambda bbox: bbox[-1], reverse=True)
        bbox_list = non_max_suppression_slow(bbox_list, 0.5)

        # Convert bbox_list to json for labelme visualization
        csv_to_json(bbox_list, image_wo_border_dir, centernet2_label_dir, coord_type="xmin_ymin_xmax_ymax", store_score=True)
        logger.info("[run_centernet2] gpu_num = {}; run_centernet2 time = {:4.3f} s".format(gpu_num, round(time.time()-start_time, 3)))

        json_file_name = os.path.splitext(image_file_name)[0] + '.json'
        logger.debug("[run_centernet2] gpu_num = {}; centernet2_child_conn[{}].send()".format(gpu_num, gpu_num))
        centernet2_child_conn[gpu_num].send([image_wo_border_dir, json_file_name])

def run_retinanet(retinanet_child_conn, gpu_num):
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_num)

    # Initialize model
    yaml_file_path = config['retinanet_yaml_file_path']
    model_file_path = config['retinanet_model_file_path']

    cfg = get_cfg()
    cfg.merge_from_file(yaml_file_path)
    cfg.MODEL.WEIGHTS = model_file_path
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.1

    # Initialize predictor
    predictor = BatchPredictor(cfg)

    while 1:
        logger.debug("[run_retinanet] gpu_num = {}; retinanet_child_conn[{}].recv()".format(gpu_num, gpu_num))
        image_wo_border_dir, image_file_name, image_shape = retinanet_child_conn[gpu_num].recv()
        start_time = time.time()
        crop_rect_arr = SharedArray.attach("shm://crop_rect_{}".format(gpu_num))
        crop_image_arr = SharedArray.attach("shm://crop_image_{}".format(gpu_num))

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
        bbox_list, outputs_all = unify_batch_predictor_output(image_file_name, image_shape, crop_rect_arr, outputs)
        bbox_list.sort(key = lambda bbox: bbox[-1], reverse=True)
        bbox_list = non_max_suppression_slow(bbox_list, 0.5)

        # Convert bbox_list to json for labelme visualization
        csv_to_json(bbox_list, image_wo_border_dir, retinanet_label_dir, coord_type="xmin_ymin_xmax_ymax", store_score=True)
        logger.info("[run_retinanet] gpu_num = {}; run_retinanet time = {:4.3f} s".format(gpu_num, round(time.time()-start_time, 3)))

        json_file_name = os.path.splitext(image_file_name)[0] + '.json'
        logger.debug("[run_retinanet] gpu_num = {}; retinanet_child_conn[{}].send()".format(gpu_num, gpu_num))
        retinanet_child_conn[gpu_num].send([image_wo_border_dir, json_file_name])

def run_yolov4(yolov4_child_conn, gpu_num):
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_num)

    score_threshold = 0.01
    nms_flag = True
    iou_threshold = 0.75
    edge_limit = 20 # pixels

    # load model
    network, class_names, class_colors = darknet.load_network( config['yolov4_cfg_file_path'], config['yolov4_data_file_path'], 
                                                               config['yolov4_model_file_path'], yolov4_batch_size )

    while 1:
        logger.debug("[run_yolov4] gpu_num = {}; yolov4_child_conn[{}].recv()".format(gpu_num, gpu_num))
        image_wo_border_dir, image_file_name, image_shape = yolov4_child_conn[gpu_num].recv()
        start_time = time.time()
        crop_rect_arr = SharedArray.attach("shm://crop_rect_{}".format(gpu_num))
        crop_image_arr = SharedArray.attach("shm://crop_image_{}".format(gpu_num))

        crop_rect_arr = crop_rect_arr.astype(np.int32)
        crop_image_arr = crop_image_arr.astype(np.uint8)
        crop_image_arr = [cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB) for crop_img in crop_image_arr]

        if yolov4_batch_size == 1:
            bbox_list = image_detection(crop_image_arr, crop_rect_arr, image_file_name, network, class_names, class_colors, \
                                        score_threshold, edge_limit)
        else:
            bbox_list = batch_detection(crop_image_arr, crop_rect_arr, image_file_name, network, class_names, class_colors, \
                                        score_threshold, edge_limit, 0.5, 0.45, yolov4_batch_size)
        
        bbox_list.sort(key = lambda bbox: bbox[-1], reverse=True)
        bbox_list = non_max_suppression_slow(bbox_list, 0.5)

        csv_to_json(bbox_list, image_wo_border_dir, yolov4_label_dir, coord_type="xmin_ymin_xmax_ymax", store_score=True)
        logger.info("[run_yolov4] gpu_num = {}; run_yolov4 time = {:4.3f} s".format(gpu_num, round(time.time()-start_time, 3)))

        json_file_name = os.path.splitext(image_file_name)[0] + '.json'
        logger.debug("[run_yolov4] gpu_num = {}; yolov4_child_conn[{}].send()".format(gpu_num, gpu_num))
        yolov4_child_conn[gpu_num].send([image_wo_border_dir, json_file_name])

def run_ensemble(gpu_num):
    center_threshold = {'3hit': 0.1, '2hit': 0.6, '1hit': 0.9}
    retina_threshold = {'3hit': 0.1, '2hit': 0.6, '1hit': 0.9}
    yolov4_threshold = {'3hit': 0.01, '2hit': 0.6, '1hit': 0.9}
    threshold = [center_threshold, retina_threshold, yolov4_threshold]
    json_idx = 1
    while 1:
        if use_centernet2:
            logger.debug("[run_ensemble] gpu_num = {}; centernet2_parent_conn[{}].recv()".format(gpu_num, gpu_num))
            image_wo_border_dir, json_file_name = centernet2_parent_conn[gpu_num].recv()
        if use_retinanet:
            logger.debug("[run_ensemble] gpu_num = {}; retinanet_parent_connn[{}].recv()".format(gpu_num, gpu_num))
            image_wo_border_dir, json_file_name = retinanet_parent_conn[gpu_num].recv()
        if use_yolov4:
            logger.debug("[run_ensemble] gpu_num = {}; yolov4_parent_conn[{}].recv()".format(gpu_num, gpu_num))
            image_wo_border_dir, json_file_name = yolov4_parent_conn[gpu_num].recv()

        logger.debug("[run_ensemble] gpu_num = {}; ensemble_child_conn[{}].send('Ready')".format(gpu_num, gpu_num))
        ensemble_child_conn[gpu_num].send("Ready")

        start_time = time.time()
        ensemble(json_idx, image_wo_border_dir, centernet2_label_dir, retinanet_label_dir, yolov4_label_dir, \
                 inference_result_label_dir, json_file_name, threshold, dashboard_txt_dir=inference_result_txt_dir)
        logger.info("[run_ensemble] gpu_num = {}; run_ensemble time = {:4.3f} s".format(gpu_num, round(time.time()-start_time, 3)))
        ensemble_time_parent_conn[gpu_num].send(json_file_name)

def draw_bbox(json_file_name, result_image_dir):
    image_wo_border_dir = config['image_wo_border_dir']
    json_file_path = os.path.join(inference_result_label_dir, json_file_name)
    image_file_name = os.path.splitext(json_file_name)[0] + '.jpg'
    image_file_path = os.path.join(image_wo_border_dir, image_file_name)

    if os.path.isfile(json_file_path):
        bbox_list = json_to_csv(json_file_path, store_score=True)
    else:
        bbox_list = list()
    image = cv2.imread(image_file_path)

    for bbox in bbox_list:
        xmin, ymin, xmax, ymax = bbox[2:6]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 8)
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

def measure_ensemble_time(gpu_num):
    image_dir = config['image_dir']
    image_backup_dir = config['image_backup_dir']

    start_time = time.time()
    while 1:
        json_file_name = ensemble_time_child_conn[gpu_num].recv()
        end_time = time.time()
        logger.info("")
        logger.info("gpu_num = {}; ensemble time measure= {:4.3f} s".format(gpu_num, round(end_time-start_time, 3)))
        logger.info("")
        start_time = end_time

        draw_bbox(json_file_name, inference_result_image_dir)

        image_file_name = os.path.splitext(json_file_name)[0] + '.jpg'
        image_file_path = os.path.join(image_dir, image_file_name)
        shutil_move(image_file_path, image_backup_dir)

if __name__ == '__main__':
    try:
        gpu_nums = config['gpu_nums']

        image_dir = config['image_dir']
        image_backup_dir = config['image_backup_dir']
        image_wo_border_dir = config['image_wo_border_dir']

        centernet2_label_dir = config['centernet2_label_dir']
        retinanet_label_dir = config['retinanet_label_dir']
        yolov4_label_dir = config['yolov4_label_dir']
        inference_result_image_dir = config['inference_result_image_dir']
        inference_result_image_backup_dir = config['inference_result_image_backup_dir']
        inference_result_label_dir = config['inference_result_label_dir']
        inference_result_txt_dir = config['inference_result_txt_dir']

        os_makedirs(image_dir, keep_exists)
        os_makedirs(image_backup_dir, keep_exists)
        os_makedirs(image_wo_border_dir, keep_exists)
        os_makedirs(centernet2_label_dir, keep_exists)
        os_makedirs(retinanet_label_dir, keep_exists)
        os_makedirs(yolov4_label_dir, keep_exists)
        os_makedirs(inference_result_image_dir, keep_exists)
        os_makedirs(inference_result_image_backup_dir, keep_exists)
        os_makedirs(inference_result_label_dir, keep_exists)
        os_makedirs(inference_result_txt_dir, keep_exists)

        # Get logger
        # logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
        logger = get_logger(name=__file__, console_handler_level=logging.INFO, \
                            file_handler_level=logging.INFO, file_name=config['inference_log_file_path'])

        image_file_queue = Queue(5)
        p_enqueue_image_file = Process(target=enqueue_image_file, name='p_enqueue_image_file', args=(image_file_queue,))
        p_enqueue_image_file.start()

        centernet2_parent_conn = [None]*gpu_nums
        centernet2_child_conn = [None]*gpu_nums
        retinanet_parent_conn = [None]*gpu_nums
        retinanet_child_conn = [None]*gpu_nums
        yolov4_parent_conn = [None]*gpu_nums
        yolov4_child_conn = [None]*gpu_nums
        ensemble_parent_conn = [None]*gpu_nums
        ensemble_child_conn = [None]*gpu_nums
        ensemble_time_parent_conn = [None]*gpu_nums
        ensemble_time_child_conn = [None]*gpu_nums

        p_dequeue_image_file = [None]*gpu_nums
        p_centernet2 = [None]*gpu_nums
        p_retinanet = [None]*gpu_nums
        p_yolov4 = [None]*gpu_nums
        p_ensemble = [None]*gpu_nums
        p_ensemble_time = [None]*gpu_nums

        for gpu_num in range(gpu_nums):
            centernet2_parent_conn[gpu_num], centernet2_child_conn[gpu_num] = Pipe()
            retinanet_parent_conn[gpu_num], retinanet_child_conn[gpu_num] = Pipe()
            yolov4_parent_conn[gpu_num], yolov4_child_conn[gpu_num] = Pipe()
            ensemble_parent_conn[gpu_num], ensemble_child_conn[gpu_num] = Pipe()
            ensemble_time_parent_conn[gpu_num], ensemble_time_child_conn[gpu_num] = Pipe()

            p_dequeue_image_file[gpu_num] = Process(target=dequeue_image_file, name='p_dequeue_image_file[{}]'.format(gpu_num), args=(image_file_queue, gpu_num))
            p_dequeue_image_file[gpu_num].start()

            if use_centernet2:
                p_centernet2[gpu_num] = Process(target=run_centernet2, name='p_centernet2[{}]'.format(gpu_num), args=(centernet2_child_conn, gpu_num))
                p_centernet2[gpu_num].start()

            if use_retinanet:
                p_retinanet[gpu_num] = Process(target=run_retinanet, name='p_retinanet[{}]'.format(gpu_num), args=(retinanet_child_conn, gpu_num))
                p_retinanet[gpu_num].start()

            if use_yolov4:
                p_yolov4[gpu_num] = Process(target=run_yolov4, name='p_yolov4[{}]'.format(gpu_num), args=(yolov4_child_conn, gpu_num))
                p_yolov4[gpu_num].start()

            p_ensemble[gpu_num] = Process(target=run_ensemble, name='p_ensemble[{}]'.format(gpu_num), args=(gpu_num, ))
            p_ensemble[gpu_num].start()

            p_ensemble_time[gpu_num] = Process(target=measure_ensemble_time, name='p_ensemble_time[{}]'.format(gpu_num), args=(gpu_num, ))
            p_ensemble_time[gpu_num].start()

            logger.debug("ensemble_child_conn[{}].send('Ready')".format(gpu_num))
            ensemble_child_conn[gpu_num].send("Ready")

        while 1:
            if not p_enqueue_image_file.is_alive():
                logger.warning("p_enqueue_image_file restart")
                p_enqueue_image_file = Process(target=enqueue_image_file, name='p_enqueue_image_file', args=(image_file_queue,))
                p_enqueue_image_file.start()

            for gpu_num in range(gpu_nums):
                if not p_dequeue_image_file[gpu_num].is_alive():
                    logger.warning("p_dequeue_image_file[{}] restart".format(gpu_num))
                    p_dequeue_image_file[gpu_num] = Process(target=dequeue_image_file, name='p_dequeue_image_file[{}]'.format(gpu_num), args=(image_file_queue, gpu_num))
                    p_dequeue_image_file[gpu_num].start()

                if use_centernet2:
                    if not p_centernet2[gpu_num].is_alive():
                        logger.warning("p_centernet2[{}] restart".format(gpu_num))
                        p_centernet2[gpu_num] = Process(target=run_centernet2, name='p_centernet2[{}]'.format(gpu_num), args=(centernet2_child_conn, gpu_num))
                        p_centernet2[gpu_num].start()
                if use_retinanet:
                    if not p_retinanet[gpu_num].is_alive():
                        logger.warning("p_retinanet[{}] restart".format(gpu_num))
                        p_retinanet[gpu_num] = Process(target=run_retinanet, name='p_retinanet[{}]'.format(gpu_num), args=(retinanet_child_conn, gpu_num))
                        p_retinanet[gpu_num].start()
                if use_yolov4:
                    if not p_yolov4[gpu_num].is_alive():
                        logger.warning("p_yolov4[{}] restart".format(gpu_num))
                        p_yolov4[gpu_num] = Process(target=run_yolov4, name='p_yolov4[{}]'.format(gpu_num), args=(yolov4_child_conn, gpu_num))
                        p_yolov4[gpu_num].start()

                if not p_ensemble[gpu_num].is_alive():
                    logger.warning("p_ensemble[{}] restart".format(gpu_num))
                    p_ensemble[gpu_num] = Process(target=run_ensemble, name='p_ensemble[{}]'.format(gpu_num), args=(gpu_num, ))
                    p_ensemble[gpu_num].start()

                if not p_ensemble_time[gpu_num].is_alive():
                    logger.warning("p_ensemble_time[{}] restart".format(gpu_num))
                    p_ensemble_time[gpu_num] = Process(target=measure_ensemble_time, name='p_ensemble_time[{}]'.format(gpu_num), args=(gpu_num, ))
                    p_ensemble_time[gpu_num].start()

            time.sleep(10)
        '''
        p_enqueue_image_file.join()
        for gpu_num in range(gpu_nums):
            p_dequeue_image_file[gpu_num].join()
            if use_centernet2:
                p_centernet2[gpu_num].join()
            if use_retinanet:
                p_retinanet[gpu_num].join()
            if use_yolov4:
                p_yolov4[gpu_num].join()
            p_ensemble[gpu_num].join()
            p_ensemble_time[gpu_num].join()
        '''
    except KeyboardInterrupt:
        for shm in SharedArray.list():
            shm_name = shm.name.decode("utf-8")
            if "crop_rect" in shm_name or "crop_image" in shm_name:
                SharedArray.delete("shm://{}".format(shm_name))
                # SharedArray.delete("shm://crop_rect_{}".format(gpu_num))
                # SharedArray.delete("shm://crop_image_{}".format(gpu_num))

        p_enqueue_image_file.terminate()
        for gpu_num in range(gpu_nums):
            p_dequeue_image_file[gpu_num].terminate()
            if use_centernet2:
                p_centernet2[gpu_num].terminate()
            if use_retinanet:
                p_retinanet[gpu_num].terminate()
            if use_yolov4:
                p_yolov4[gpu_num].terminate()
            p_ensemble[gpu_num].terminate()
            p_ensemble_time[gpu_num].terminate()
