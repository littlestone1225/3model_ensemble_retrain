#!/usr/bin/env python3
import os, sys
from collections import OrderedDict
import logging
import zipfile

aoi_dir_name = os.getenv('AOI_DIR_NAME')
assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"

current_dir = os.path.dirname(os.path.abspath(__file__))
idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
aoi_dir = current_dir[:idx]

from afs import models
from evaluation import evaluation_models_by_fp_rate, evaluation_ensemble_result

sys.path.append(os.path.join(aoi_dir, "config"))
from config import read_config_yaml

sys.path.append(os.path.join(aoi_dir, "utils"))
from utils import os_makedirs, shutil_copyfile, shutil_copyfile_to_dir
from logger import get_logger

sys.path.append(os.path.join(aoi_dir, "ensemble"))
from ensemble import get_ensemble_result_on_test_and_fn

# Read config_file
config_file = os.path.join(aoi_dir, 'config/config.yaml')
config = read_config_yaml(config_file)

# Get logger
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
logger = get_logger(name=__file__, console_handler_level=logging.INFO, file_handler_level=None)

def copy_best_model_to_new_model_file_dir(src_file_path, dst_file_dir):
    os_makedirs(dst_file_dir)
    src_file_name = os.path.basename(src_file_path)
    src_file_id, ext = os.path.splitext(src_file_name)
    src_file_id_list = src_file_id.split('_')
    if len(src_file_id_list) == 4 and src_file_id_list[0].isalpha() and \
       src_file_id_list[1].isnumeric() and src_file_id_list[2].isnumeric() and src_file_id_list[3].isnumeric():
        dst_file_path = os.path.join(dst_file_dir, src_file_name)
    else:
        dst_file_name_1 = os.path.basename(dst_file_dir).lower()
        dst_file_name_2 = os.path.basename(os.path.dirname(dst_file_dir)).lower()
        dst_file_name = '_'.join([dst_file_name_1, dst_file_name_2]) + ext
        dst_file_path = os.path.join(dst_file_dir, dst_file_name)

    shutil_copyfile(src_file_path, dst_file_path)


if __name__ == '__main__':
    # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Get_ensemble_result_on_test_and_fn          #
    #       model_type = 'best_model' / 'old_model'     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # #
    get_ensemble_result_on_test_and_fn(model_type='best_model')
    get_ensemble_result_on_test_and_fn(model_type='old_model')


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Copy best models to models/yyyy_mm_dd folder        #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    copy_best_model_to_new_model_file_dir(src_file_path=config['centernet2_best_model_file_path'], \
                                          dst_file_dir=config['centernet2_new_model_file_dir'])
    copy_best_model_to_new_model_file_dir(src_file_path=config['retinanet_best_model_file_path'], \
                                          dst_file_dir=config['retinanet_new_model_file_dir'])
    copy_best_model_to_new_model_file_dir(src_file_path=config['yolov4_best_model_file_path'], \
                                          dst_file_dir=config['yolov4_new_model_file_dir'])


    # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Create best model result directory      #
    # # # # # # # # # # # # # # # # # # # # # # # # #
    best_model_dir = os.path.dirname(config['centernet2_new_model_file_dir'])
    best_model_result_dir = os.path.join(best_model_dir, 'result')
    os_makedirs(best_model_result_dir)


    # # # # # # # # # # # # # # # # # # # # # # #
    #       Evaluation individual result        #
    # # # # # # # # # # # # # # # # # # # # # # #
    # Evaluate best models
    label_dir_list = [  config['centernet2_best_model_label_dir'], \
                        config['retinanet_best_model_label_dir'], \
                        config['yolov4_best_model_label_dir'] ]
    new_model_file_dir_list = [ config['centernet2_new_model_file_dir'], \
                                config['retinanet_new_model_file_dir'], \
                                config['yolov4_new_model_file_dir'] ]
    refine_num_list = [11, 11, 101]
    log_file_path, score_threshold_list = evaluation_models_by_fp_rate(label_dir_list, new_model_file_dir_list, refine_num_list)
    dst_file_path = os.path.join(best_model_result_dir, 'best_models_fp_0.6.txt')
    shutil_copyfile(log_file_path, dst_file_path)

    # Evaluate old models
    label_dir_list = [  config['centernet2_old_model_label_dir'], \
                        config['retinanet_old_model_label_dir'], \
                        config['yolov4_old_model_label_dir'] ]
    new_model_file_dir_list = [ config['centernet2_new_model_file_dir'], \
                                config['retinanet_new_model_file_dir'], \
                                config['yolov4_new_model_file_dir'] ]
    refine_num_list = [11, 11, 101]
    log_file_path, score_threshold_list = evaluation_models_by_fp_rate(label_dir_list, new_model_file_dir_list, refine_num_list)
    dst_file_path = os.path.join(best_model_result_dir, 'old_models_fp_0.6.txt')
    shutil_copyfile(log_file_path, dst_file_path)


    # # # # # # # # # # # # # # # # # # # # #
    #       Evaluation ensemble result      #
    # # # # # # # # # # # # # # # # # # # # #
    old_ensemble_result, old_ensemble_result_yaml_file_path, old_ensemble_result_txt_file_path = \
        evaluation_ensemble_result(ensemble_result_dirname = 'ensemble_result_old_model', \
                                   score_threshold = 0.01, test_and_fn_list = ['test', 'fn'])

    best_ensemble_result, best_ensemble_result_yaml_file_path, best_ensemble_result_txt_file_path = \
        evaluation_ensemble_result(ensemble_result_dirname = 'ensemble_result_best_model', \
                                   score_threshold = 0.01, test_and_fn_list = ['test', 'fn'])

    shutil_copyfile_to_dir(old_ensemble_result_yaml_file_path, best_model_result_dir)
    shutil_copyfile_to_dir(old_ensemble_result_txt_file_path, best_model_result_dir)
    shutil_copyfile_to_dir(best_ensemble_result_yaml_file_path, best_model_result_dir)
    shutil_copyfile_to_dir(best_ensemble_result_txt_file_path, best_model_result_dir)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Zip best models and corresponding matrices      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    zip_file_path = best_model_dir + '.zip'
    zf = zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(best_model_dir):
        for file_name in files:
            src_file_path = os.path.join(root, file_name)
            dst_file_path = src_file_path.replace(best_model_dir + '/', '')
            zf.write(src_file_path, dst_file_path)
    zf.close()

    # # # # # # # # # # # # # # # # # # # # # # #
    #       Upload to aifs model repository     #
    # # # # # # # # # # # # # # # # # # # # # # #
    if config['production'] == 'retrain_aifs':
        eval_result = OrderedDict()
        anomaly_type_list = ['bridge', 'empty', 'appearance_less', 'appearance_hole', 'excess_solder', 'appearance']
        confusion_matrix_list = ['critical_recall', 'normal_recall', 'fp_rate']
        for key in anomaly_type_list + confusion_matrix_list:
            eval_result[key] = list()

        for anomaly_type in anomaly_type_list:
            eval_result[anomaly_type].append(str(old_ensemble_result['test_and_fn']['label_fn_dict'][anomaly_type]))
            eval_result[anomaly_type].append(str(best_ensemble_result['test_and_fn']['label_fn_dict'][anomaly_type]))
            eval_result[anomaly_type].append(str(best_ensemble_result['test_and_fn']['label_dict'][anomaly_type]))

        for confusion_matrix in confusion_matrix_list:
            eval_result[confusion_matrix].append(str(old_ensemble_result['test_and_fn'][confusion_matrix]))
            eval_result[confusion_matrix].append(str(best_ensemble_result['test_and_fn'][confusion_matrix]))

        # User-define evaluation result. Type:dict
        extra_evaluation = {}

        # User-define Tags. Type:dict
        tags = {
            'old/new/gt (bridge)': '/'.join(eval_result['bridge']),
            'old/new/gt (empty)': '/'.join(eval_result['empty']),
            'old/new/gt (excess_solder)': '/'.join(eval_result['excess_solder']),
            'old/new/gt (appearance_hole)': '/'.join(eval_result['appearance_hole']),
            'old/new/gt (appearance)': '/'.join(eval_result['appearance']),
            'old/new/gt (appearance_less)': '/'.join(eval_result['appearance_less']),
            'old/new (critical_recall)': '/'.join(eval_result['critical_recall']),
            'old/new (normal_recall)': '/'.join(eval_result['normal_recall']),
            'old/new (fp_rate)': '/'.join(eval_result['fp_rate']),
        }

        # Model object
        afs_models = models()

        afs_models.upload_model(
            model_path=zip_file_path,
            model_repository_name='pcb_retrain.zip',
            # extra_evaluation=extra_evaluation,
            tags=tags
        )
