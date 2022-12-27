#!/usr/bin/env python3
import os, sys
import cv2
from operator import add
import torch
import time
import logging
import math


from detectron2.config import get_cfg
from detectron2.engine import BatchPredictor
from detectron2.structures import Instances, Boxes


aoi_dir_name = os.getenv('AOI_DIR_NAME')
assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"

current_dir = os.path.dirname(os.path.abspath(__file__))
idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
aoi_dir = current_dir[:idx]

from nms import non_max_suppression_slow,non_max_suppression_fast

sys.path.append(os.path.join(aoi_dir, "config"))
from config import read_config_yaml

sys.path.append(os.path.join(aoi_dir, "utils"))
from utils import os_makedirs
from logger import get_logger

sys.path.append(os.path.join(aoi_dir, "data_preprocess"))
from crop_small_image import crop_sliding_window
from csv_json_conversion import csv_to_json, json_to_bbox

sys.path.append(os.path.join(aoi_dir, "detectron2/projects/CenterNet2"))
from centernet.config import add_centernet_config

sys.path.append(os.path.join(aoi_dir, "YOLOv4"))
import darknet
from darknet_inference import image_detection, batch_detection, \
                              Score_threshold, Edge_limit,Batch_size



# Read config_file
config_file = os.path.join(aoi_dir, 'config/config.yaml')
config = read_config_yaml(config_file)

# Get logger
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
logger = get_logger(name=__file__, console_handler_level=logging.INFO, file_handler_level=None)

num_to_category_dict = {0: 'bridge', 1: 'appearance_less', 2: 'excess_solder', 3: 'appearance'}
category_to_color_dict = {'bridge': [0, 0, 255], 'appearance_less': [255,191,0], 'excess_solder': [221,160,221], 'appearance': [0,165,255]}
default_color = [0, 255, 0] # in BGR

total_image = 60
YOLOv4_batch_size = 1

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

    return bbox_list

def is_model_name_with_iter(model_file_name):
    model_file_id, ext = os.path.splitext(model_file_name)
    if 'model_' in model_file_id and ext == '.pth':
        iter_str =  model_file_id.split('_')[1]
        if iter_str.isdigit():
            return True
        else:
            return False
    elif 'yolov4-pcb_' in model_file_id and ext == '.weights':
        return True
    else:
        return False

def sort_model_name_by_iter(model_file_name):
    model_file_id, ext = os.path.splitext(model_file_name)
    iter_str =  model_file_id.split('_')[1]
    return iter_str

def validation(input_dict):
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(1)
    model_type = input_dict['model_type']
    data_dir = input_dict['data_dir']
    image_dir = input_dict['image_dir']
    model_version = input_dict['model_version']
    model_file_dir = input_dict['model_file_dir']
    model_file_name = input_dict['model_file_name']
    model_iter_list = input_dict['model_iter_list']
    save_inference_image = input_dict['save_inference_image']
    print(model_type)
    if model_file_name==None:
        # Sort the model name by iteration number
        model_file_list = os.listdir(model_file_dir) 
        model_file_list = list(filter(is_model_name_with_iter, model_file_list))
        model_file_list.sort(key = sort_model_name_by_iter)
    else:
        model_file_list = [model_file_name]

    #model_file_list = ["model_0000111.pth","model_0000011.pth"]
    # Iterate through model files
    for model_file_name in model_file_list:
        model_file_id, ext = os.path.splitext(model_file_name)

        if model_iter_list==None:
            logger.info("model_file_name = {}".format(model_file_name))
        else:
            if os.path.splitext(model_file_name)[0] in model_iter_list:
                logger.info("model_file_name = {}".format(model_file_name))
            else:
                continue

        model_file_path = os.path.join(model_file_dir, model_file_name)

        # Create inference result directory to store inference images and labels
        if save_inference_image:
            inference_result_image_dir = os.path.join(data_dir, '{}_inference_result'.format(model_version), model_file_id, 'images')
            os_makedirs(inference_result_image_dir)
        inference_result_label_dir = os.path.join(data_dir, '{}_inference_result'.format(model_version), model_file_id, 'labels')
        os_makedirs(inference_result_label_dir)

        # Initialize model
        if model_type == 'YOLOv4':
            network, class_names, class_colors = darknet.load_network(config['yolov4_cfg_file_path'], 
                                                                      config['yolov4_data_file_path'], 
                                                                      model_file_path, 
                                                                      YOLOv4_batch_size
                                                                     )
        elif model_type in ['RetinaNet', 'CenterNet2']:
            cfg = get_cfg()
            cfg.defrost()
            if model_type == 'RetinaNet':
                yaml_file_path = config['retinanet_yaml_file_path']
                cfg.merge_from_file(yaml_file_path)
                cfg.MODEL.WEIGHTS = model_file_path
                cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.1
            elif model_type == 'CenterNet2':
                add_centernet_config(cfg)
                yaml_file_path = config['centernet2_yaml_file_path']
                cfg.merge_from_file(yaml_file_path)
                cfg.MODEL.WEIGHTS = model_file_path
                cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.1
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1

            cfg.freeze()

            # Initialize predictor
            predictor = BatchPredictor(cfg)
        else:
            assert False, "model_type = {} is not supported".format(model_type)

        # Iterate through validation image
        global_start_time = time.time()
        min_time_duration = 0.0
        max_time_duration = 0.0
        for idx, image_file_name in enumerate(os.listdir(image_dir)):
            image_file_path = os.path.join(image_dir, image_file_name)
            image_org = cv2.imread(image_file_path)

            if model_type == 'YOLOv4':   
                image_org = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)


            # crop big pcb to small img for inference
            crop_rect_list, crop_image_list = crop_sliding_window(image_org)

            # Do the batch inference on crop_image_list to get bbox_list
            start_time = time.time()
            if model_type == 'YOLOv4':                
                if YOLOv4_batch_size == 1 :
                    bbox_list = image_detection(crop_image_list, crop_rect_list, image_file_name, network, class_names, class_colors, \
                                                Score_threshold, Edge_limit)
                else : 
                    bbox_list = batch_detection(crop_image_list, crop_rect_list, image_file_name, network, class_names, class_colors, \
                                                Score_threshold, Edge_limit, .5, .45, YOLOv4_batch_size)
                
                bbox_list.sort(key = lambda bbox: bbox[-1], reverse=True)
                bbox_list = non_max_suppression_slow(bbox_list, 0.5)

            else: # model_type in ['RetinaNet', 'CenterNet2']
                outputs = list()
                total_crop_image = len(crop_image_list)
                if total_crop_image > total_image:
                    for i in range(math.ceil(total_crop_image / total_image)):
                        start = i*total_image
                        end = (i+1)*total_image
                        if end > total_crop_image:
                            end = total_crop_image
                        logger.debug("start = {} ; end = {}".format(start, end))
                        out = predictor(crop_image_list[start:end])
                        outputs.extend(out)
                else:
                    outputs = predictor(crop_image_list)

                assert len(outputs) == len(crop_image_list), "len(outputs) != len(crop_image_list)"
                bbox_list = unify_batch_predictor_output(image_file_name, image_org.shape, crop_rect_list, outputs)
                logger.debug("brfore nms bbox = {}".format(len(bbox_list)))
                bbox_list.sort(key = lambda bbox: bbox[-1], reverse=True)
                bbox_list = non_max_suppression_slow(bbox_list, 0.5)
                logger.debug("after nms bbox = {}".format(len(bbox_list)))

            # Convert bbox_list to json for labelme visualization
            csv_to_json(bbox_list, image_dir, inference_result_label_dir, coord_type="xmin_ymin_xmax_ymax", store_score=True)

            if save_inference_image:
                # Save original image with inference result
                image_bbox = image_org.copy()
                for bbox in bbox_list:
                    file_name, error_type, xmin, ymin, xmax, ymax, score = bbox
                    color = category_to_color_dict.get(error_type, default_color)
                    cv2.rectangle(image_bbox, (xmin, ymin), (xmax, ymax), color, 6)
                    cv2.putText(image_bbox, str(score), (xmin, ymin-10), cv2.FONT_HERSHEY_TRIPLEX, 1, color, 2, cv2.LINE_8)
                image_bbox_file_path = os.path.join(inference_result_image_dir, image_file_name)
                image_bbox = cv2.resize(image_bbox, None, fx=0.5, fy=0.5)
                cv2.imwrite(image_bbox_file_path, image_bbox, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            time_duration = round(time.time()-start_time, 3)
            if idx == 0:
                min_time_duration = time_duration
                max_time_duration = time_duration
            else:
                if time_duration < min_time_duration:
                    min_time_duration = time_duration
                elif time_duration > max_time_duration:
                    max_time_duration = time_duration
            logger.info("{:>4d}, crop image number = {:>3d}, time = {:4.3f} s, ({:4.3f}, {:4.3f})" \
                        .format(idx, len(crop_rect_list), time_duration, min_time_duration, max_time_duration))
            
        if model_type =='YOLOv4':
            darknet.free_network_ptr(network) # Important: free darknet from gpu 

        logger.info("Total time = {} s".format(round(time.time()-global_start_time)))


def validate_old_model_on_test_and_fn(model_type):
    test_data_dir = config['test_data_dir']
    retrain_data_val_dir = config['retrain_data_val_dir']

    if model_type == 'RetinaNet':
        old_model_file_path = config['retinanet_old_model_file_path'] # /home/aoi/AOI_PCB_Retrain_detectron2/models/old/RetinaNet/v6_4_15_model_0035999.pth
    elif model_type == 'CenterNet2':
        old_model_file_path = config['centernet2_old_model_file_path']
    elif model_type == 'YOLOv4':
        old_model_file_path = config['yolov4_old_model_file_path']
    else:
        assert False, "model_type = {} is not supported".format(model_type)

    old_model_file_dir = os.path.dirname(old_model_file_path) # /home/aoi/AOI_PCB_Retrain_detectron2/models/old/RetinaNet
    old_model_file_name = os.path.basename(old_model_file_path) # v6_4_15_model_0035999.pth

    test_data_dict = {
        'model_type': model_type,
        'data_dir': test_data_dir,
        'image_dir': os.path.join(test_data_dir, 'images_wo_border'),
        'model_version': '{}_old'.format(model_type),
        'model_file_dir': old_model_file_dir,
        'model_file_name': old_model_file_name,
        'model_iter_list': None,
        'save_inference_image': False
    }
    validation(test_data_dict)

    retrain_data_val_dict = {
        'model_type': model_type,
        'data_dir': retrain_data_val_dir,
        'image_dir': os.path.join(retrain_data_val_dir, 'images_random_crop'),
        'model_version': '{}_old'.format(model_type),
        'model_file_dir': old_model_file_dir,
        'model_file_name': old_model_file_name,
        'model_iter_list': None,
        'save_inference_image': False
    }
    validation(retrain_data_val_dict)

def validate_new_models_on_val(model_type, model_iter_list=None):
    val_data_dir = config['val_data_dir']

    if model_type == 'RetinaNet':
        model_output_dir = config['retinanet_model_output_dir']
        model_output_version = config['retinanet_model_output_version']
    elif model_type == 'CenterNet2':
        model_output_dir = config['centernet2_model_output_dir']
        model_output_version = config['centernet2_model_output_version']
    elif model_type == 'YOLOv4':
        model_output_dir = config['yolov4_model_output_dir']
        model_output_version = config['yolov4_model_output_version']
    else:
        assert False, "model_type = {} is not supported".format(model_type)

    val_data_dict = {
        'model_type': model_type,
        'data_dir': val_data_dir,
        'image_dir': os.path.join(val_data_dir, 'images_wo_border'),
        'model_version': model_output_version,
        'model_file_dir': model_output_dir,
        'model_file_name': None,
        'model_iter_list': model_iter_list,
        'save_inference_image': False
    }
    validation(val_data_dict)

def validate_new_models_on_test_and_fn(model_type, model_iter_list):
    test_data_dir = config['test_data_dir']
    retrain_data_val_dir = config['retrain_data_val_dir']

    if model_type == 'RetinaNet':
        model_output_dir = config['retinanet_model_output_dir']
        model_output_version = config['retinanet_model_output_version']
    elif model_type == 'CenterNet2':
        model_output_dir = config['centernet2_model_output_dir']
        model_output_version = config['centernet2_model_output_version']
    elif model_type == 'YOLOv4':
        model_output_dir = config['yolov4_model_output_dir']
        model_output_version = config['yolov4_model_output_version']
    else:
        assert False, "model_type = {} is not supported".format(model_type)

    test_data_dict = {
        'model_type': model_type,
        'data_dir': test_data_dir,
        'image_dir': os.path.join(test_data_dir, 'images_wo_border'),
        'model_version': model_output_version,
        'model_file_dir': model_output_dir,
        'model_file_name': None,
        'model_iter_list': model_iter_list,
        'save_inference_image': False
    }
    validation(test_data_dict)

    retrain_data_val_dict = {
        'model_type': model_type,
        'data_dir': retrain_data_val_dir,
        'image_dir': os.path.join(retrain_data_val_dir, 'images_random_crop'),
        'model_version': model_output_version,
        'model_file_dir': model_output_dir,
        'model_file_name': None,
        'model_iter_list': model_iter_list,
        'save_inference_image': False
    }
    validation(retrain_data_val_dict)

if __name__ == '__main__':
    # model_iter_list = ['model_0014999']
    model_iter_list = list(range(200, 40001, 200))
    min_iter = 1000
    max_iter = 30000
    model_iter_list = list(filter(lambda iter: iter>=min_iter and iter<=max_iter, model_iter_list))
    model_iter_list = ['model_{:07d}'.format(int(model_iter)-1) for model_iter in model_iter_list]
    # '''
    validate_new_models_on_test_and_fn(model_type='RetinaNet', model_iter_list=model_iter_list)
    # validate_new_models_on_val()

