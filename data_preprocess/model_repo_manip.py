#!/usr/bin/python3
import os, sys
import logging
import zipfile
from afs import models

aoi_dir_name = os.getenv('AOI_DIR_NAME')
assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"

current_dir = os.path.dirname(os.path.abspath(__file__))
idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
aoi_dir = current_dir[:idx]

sys.path.append(os.path.join(aoi_dir, "config"))
from config import read_config_yaml, write_config_yaml_with_key_value

sys.path.append(os.path.join(aoi_dir, "utils"))
from utils import os_makedirs, shutil_copyfile
from logger import get_logger

# Read config_file
config_file = os.path.join(aoi_dir, 'config/config.yaml')
config = read_config_yaml(config_file)

# Get logger
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
logger = get_logger(name=__file__, console_handler_level=logging.DEBUG, file_handler_level=None)


# def upload_to_model_repo(repo_name='pcb_retrain'):


def take_floating_ymd(elem):
    days_list = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    file_id, ext = os.path.splitext(elem)
    year, month, day = [int(num) for num in file_id.split('_')]
    days = sum(days_list[0:month]) + day
    key = year + days / sum(days_list)
    return key

def get_latest_model_from_pcb_dataset(pcb_dataset_model_dir):
    zip_file_list = [zipfile for zipfile in os.listdir(pcb_dataset_model_dir) if zipfile.endswith('.zip')]
    zip_file_list.sort(key=take_floating_ymd, reverse=True)
    if len(zip_file_list) > 0:
        zip_file_name = zip_file_list[0]
        zip_file_path = os.path.join(pcb_dataset_model_dir, zip_file_name)
        return zip_file_path
    return None

def download_from_model_repo(repo_name='pcb_retrain'):
    old_model_dir = os.path.join(aoi_dir, 'models/old')
    write_config_yaml_with_key_value(config_file, 'old_model_dir', old_model_dir)
    os_makedirs(old_model_dir)

    pcb_dataset_dir = config['pcb_dataset_dir']
    pcb_dataset_model_dir = os.path.join(pcb_dataset_dir, 'models') 

    if config['production']=='retrain_aifs':
        try:
            afs_models = models()
            info = afs_models.get_latest_model_info(model_repository_name=repo_name)
            zip_file_path = old_model_dir + '.zip'
            afs_models.download_model(
                save_path = zip_file_path, 
                model_repository_name = repo_name, 
                last_one = True)
        except Exception as e:
            logger.warning(e)
            zip_file_path = get_latest_model_from_pcb_dataset(pcb_dataset_model_dir)
    else:
        zip_file_path = get_latest_model_from_pcb_dataset(pcb_dataset_model_dir)
    
    assert zip_file_path!=None, "zip_file_path = {}".format(zip_file_path)

    with zipfile.ZipFile(zip_file_path, 'r') as zf:
        zf.extractall(old_model_dir)

if __name__ == "__main__":
    download_from_model_repo()