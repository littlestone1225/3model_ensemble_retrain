#!/usr/bin/env python3
import os, sys
import time
import logging
import math

aoi_dir_name = os.getenv('AOI_DIR_NAME')
assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"

current_dir = os.path.dirname(os.path.abspath(__file__))
idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
aoi_dir = current_dir[:idx]

sys.path.append(os.path.join(aoi_dir, "validation"))
from validation import validate_new_models_on_val, validate_new_models_on_test_and_fn, \
                       validate_old_model_on_test_and_fn
from evaluation import evaluate_new_models_on_val, evaluate_new_models_on_test_and_fn_by_fp_rate, \
                       evaluate_old_model_on_test_and_fn_by_fp_rate, get_best_model

sys.path.append(os.path.join(aoi_dir, "config"))
from config import read_config_yaml, write_config_yaml

sys.path.append(os.path.join(aoi_dir, "utils"))
from utils import os_makedirs
from logger import get_logger

# Read config_file
config_file = os.path.join(aoi_dir, 'config/config.yaml')
config = read_config_yaml(config_file)

# Get logger
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
logger = get_logger(name=__file__, console_handler_level=logging.DEBUG, file_handler_level=None)

if __name__ == '__main__':
    run_validation = False
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   1. Validate new models on validation set                                    #
    #   2. Evaluate new models on validation set by fixed fp_rate_threshold = 0.4   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    inference_path = os.path.join(config['val_data_dir'], '{}_inference_result'.format(config['yolov4_model_output_version']))

    #if run_validation or not os.path.exists(inference_path):
    #    validate_new_models_on_val(model_type='YOLOv4')

    val_best_model_iter_list = evaluate_new_models_on_val(model_type='YOLOv4', fp_rate=0.6, return_type='best_models')

    for best_model_iter in val_best_model_iter_list:
        logger.info(best_model_iter)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   1. Validate new models on test set and retrain val set                                    #
    #   2. Evaluate new models on test set and retrain val set by fixed fp_rate_threshold = 0.6   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    test_inference_path = os.path.join(config['test_data_dir'], '{}_inference_result'.format(config['yolov4_model_output_version']))
    retrain_inference_path = os.path.join(config['retrain_data_val_dir'], '{}_inference_result'.format(config['yolov4_model_output_version']))

    if run_validation or (not os.path.exists(retrain_inference_path) or not os.path.exists(test_inference_path)):
        validate_new_models_on_test_and_fn(model_type='YOLOv4', model_iter_list=val_best_model_iter_list)

    test_best_model_iter_list = evaluate_new_models_on_test_and_fn_by_fp_rate(model_type='YOLOv4', model_iter_list=val_best_model_iter_list, \
                                                                              fp_rate=0.6, return_type='best_models')

    for best_model_iter in test_best_model_iter_list:
        logger.info(best_model_iter)

    
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   1. Evaluate new models on test set and retrain val set by fixed fp_rate_threshold = 0.6   #
    #   2. Get best model from new models                                                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    new_eval_result_list = evaluate_new_models_on_test_and_fn_by_fp_rate(model_type='YOLOv4', model_iter_list=test_best_model_iter_list, \
                                                                         fp_rate=0.6, return_type='eval_result')
    best_model_iter = get_best_model(new_eval_result_list)
    best_eval_result = None
    for eval_result in new_eval_result_list:
        if eval_result['model_iter'] == best_model_iter:
            best_eval_result = eval_result
            break
    logger.info("best_model_iter = {}".format(best_model_iter))
    

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   1. Validate old model on test set and retrain val set                                     #
    #   2. Evaluate old model on test set and retrain val set by fixed fp_rate_threshold = 0.6    #
    #   3. Get best model from between old model and best new model                               #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    test_old_inference_path = os.path.join(config['test_data_dir'], 'YOLOv4_old_inference_result')
    retrain_olg_inference_path = os.path.join(config['retrain_data_val_dir'], 'YOLOv4_old_inference_result')

    if run_validation or (not os.path.exists(test_old_inference_path) or not os.path.exists(retrain_olg_inference_path)):
        validate_old_model_on_test_and_fn(model_type='YOLOv4')
    eval_result_list = evaluate_old_model_on_test_and_fn_by_fp_rate(model_type='YOLOv4', fp_rate=0.6, return_type='eval_result')
    eval_result_list.append(best_eval_result)
    best_model_iter = get_best_model(eval_result_list)
    logger.info("best_model_iter = {}".format(best_model_iter))


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   1. Update yolov4_best_model_file_path in config.yaml     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    test_data_dir = config['test_data_dir']
    retrain_data_val_dir = config['retrain_data_val_dir']

    yolov4_old_model_file_path = config['yolov4_old_model_file_path']
    yolov4_old_model_file_id = os.path.splitext(os.path.basename(yolov4_old_model_file_path))[0]
    yolov4_old_model_label_dir = [os.path.join(test_data_dir, 'YOLOv4_old_inference_result', yolov4_old_model_file_id, 'labels'),
                                     os.path.join(retrain_data_val_dir, 'YOLOv4_old_inference_result', yolov4_old_model_file_id, 'labels')]

    if best_model_iter == best_eval_result['model_iter']:
        yolov4_model_output_version = config['yolov4_model_output_version']
        yolov4_best_model_file_path = os.path.join(config['yolov4_model_output_dir'], "{}.weights".format(best_model_iter))
        yolov4_best_model_label_dir = [os.path.join(test_data_dir, '{}_inference_result'.format(yolov4_model_output_version), \
                                          best_model_iter, 'labels'),
                                          os.path.join(retrain_data_val_dir, '{}_inference_result'.format(yolov4_model_output_version), \
                                          best_model_iter, 'labels')]
    else:
        yolov4_best_model_file_path = yolov4_old_model_file_path
        yolov4_best_model_label_dir = yolov4_old_model_label_dir


    subdict = { 'yolov4_old_model_label_dir': yolov4_old_model_label_dir,
                'yolov4_best_model_file_path': yolov4_best_model_file_path,
                'yolov4_best_model_label_dir': yolov4_best_model_label_dir
              }
    write_config_yaml(config_file, subdict)
