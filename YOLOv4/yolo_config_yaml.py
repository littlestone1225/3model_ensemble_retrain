#!/usr/bin/env python3
import os,sys 

aoi_dir_name = os.getenv('AOI_DIR_NAME')
assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"

current_dir = os.path.dirname(os.path.abspath(__file__))
idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
aoi_dir = current_dir[:idx]

sys.path.append(os.path.join(aoi_dir, "config"))
from config import read_config_yaml, write_config_yaml

if __name__ == "__main__":

    # read global yaml
    global_config_file = os.path.join(aoi_dir, 'config/config.yaml')
    global_config_dict = read_config_yaml(global_config_file)
    
    # Write config_file
    config_file = os.path.join(aoi_dir, 'YOLOv4/yolov4.yaml')
    config_dict = read_config_yaml(config_file)
    
    # set path
    config_dict['TRAIN_crop_image_path'] = os.path.join(global_config_dict['pcb_data_dir'], 'train')
    config_dict['TRAIN_crop_label_file'] = os.path.join(global_config_dict['pcb_data_dir'], 'annotations/train.json')
    config_dict['VALID_crop_image_path'] = os.path.join(global_config_dict['pcb_data_dir'], 'val')
    config_dict['VALID_crop_label_file'] = os.path.join(global_config_dict['pcb_data_dir'], 'annotations/val.json')
    config_dict['VAL_image_path']       = os.path.join(global_config_dict['val_data_dir'])
    config_dict['TEST_image_path']       = os.path.join(global_config_dict['test_data_dir'])
    config_dict['RETRAIN_crop_image_path'] = os.path.join(global_config_dict['retrain_data_val_dir'])
    
    config_dict['YOLO_darknet_path'] = os.path.join(aoi_dir, 'darknet')
    config_dict['YOLO_config_path'] = os.path.join(current_dir, 'cfg')
    config_dict['YOLO_weight_path'] = os.path.join(current_dir, 'weights/{}'.format(global_config_dict['yolov4_model_output_version']))
    config_dict['YOLO_dataset_path']= os.path.join(current_dir, 'yolov4_dataset')
    
    config_dict['before_retrain_path']= os.path.join(current_dir, 'result/{}/before_retrain'.format(global_config_dict['yolov4_model_output_version']))
    config_dict['after_retrain_path']= os.path.join(current_dir, 'result/{}/after_retrain'.format(global_config_dict['yolov4_model_output_version']))
    config_dict['select_weight_path']= os.path.join(current_dir, 'result/{}/after_retrain/select_weights'.format(global_config_dict['yolov4_model_output_version']))
    config_dict['final_weight_path']= os.path.join(current_dir, 'result/{}/after_retrain/final_weights'.format(global_config_dict['yolov4_model_output_version']))
    
    # set file
    config_dict['yolov4_old_model_file_path'] = global_config_dict['yolov4_old_model_file_path']

    
    config_dict['test_FN_csv'] = os.path.join(current_dir, 'cfg/GT_FN.csv')
    config_dict['test_GT_csv'] = os.path.join(current_dir, 'cfg/GT.csv')
    config_dict['valid_GT_100_csv'] = os.path.join(current_dir, 'cfg/GT_100.csv')
    
    config_dict['yolo_result_100_csv'] = "yolo_result_100.csv"
    config_dict['yolo_result_csv'] = "yolo_result.csv"
    config_dict['yolo_result_FN_csv'] = "yolo_result_FN.csv"
    
    config_dict['valid_result_csv'] = "inference_for_each_model.csv"
    config_dict['test_result_csv'] = "test_inference_for_each_model_FPrate.csv"

    # set config
    config_dict['window_size']         = global_config_dict['crop_w']
    config_dict['margin']              = global_config_dict['margin']
    config_dict['inference_batch_size']= 4


    print(dict(config_dict))
    write_config_yaml(config_file, config_dict)