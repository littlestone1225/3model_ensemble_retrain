#!/usr/bin/python3
import os, sys
import logging
from afs2datasource import DBManager, constant

aoi_dir_name = os.getenv('AOI_DIR_NAME')
assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"

current_dir = os.path.dirname(os.path.abspath(__file__))
idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
aoi_dir = current_dir[:idx]

sys.path.append(os.path.join(aoi_dir, "config"))
from config import read_config_yaml, write_config_yaml_with_key_value

sys.path.append(os.path.join(aoi_dir, "utils"))
from utils import os_makedirs
from logger import get_logger

# Read config_file
config_file = os.path.join(aoi_dir, 'config/config.yaml')
config = read_config_yaml(config_file)

# Get logger
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
logger = get_logger(name=__file__, console_handler_level=logging.DEBUG, file_handler_level=None)


def upload_to_s3(bucket_name='pcb-dataset'):
    train_data_dir = config['train_data_dir']
    test_data_dir = config['test_data_dir']
    retrain_data_dir = config['retrain_data_dir']
    retrain_data_org_dir = config['retrain_data_org_dir']
    pcb_dataset_dir = config['pcb_dataset_dir']

    train_data_image_dir = os.path.join(train_data_dir, 'images')
    train_data_label_dir = os.path.join(train_data_dir, 'labels')
    test_data_image_dir = os.path.join(test_data_dir, 'images')
    test_data_label_dir = os.path.join(test_data_dir, 'labels')
    models_dir = os.path.join(pcb_dataset_dir, 'models')



    local_and_root_dir_list = [ [train_data_image_dir, os.path.dirname(train_data_dir) + '/'],
                                [train_data_label_dir, os.path.dirname(train_data_dir) + '/'],
                                [test_data_image_dir , os.path.dirname(test_data_dir) + '/'],
                                [test_data_label_dir , os.path.dirname(test_data_dir) + '/'],
                                [retrain_data_org_dir, os.path.dirname(retrain_data_dir) + '/'],
                                [models_dir, os.path.dirname(models_dir) + '/']
                              ]

    if config['production']=='retrain_aifs':
        db = DBManager()
    else:
        db = DBManager(
            db_type=constant.DB_TYPE['S3'],
            endpoint="https://200-9090.aifs.ym.wise-paas.com",
            access_key="mDUtLL4OCkM4CjieIxlxozvTkpp8r06L",
            secret_key="Q3CwL8dYmX5i1jSbnSQrU29bHtRPr0BT",
            is_verify=False,
            buckets=[{
                'bucket': bucket_name,
                'blobs': {
                    'files': [],
                    'folders': []
                }
            }]
        )

    try:
        db.connect()
        for local_and_root_dir in local_and_root_dir_list:
            local_dir = local_and_root_dir[0]
            root_dir = local_and_root_dir[1]

            for root, dirs, files in os.walk(local_dir):
                for local_file in files:
                    local_file_path = os.path.join(root, local_file)
                    s3_file_path = local_file_path.replace(root_dir, '')
                    logger.info('Uploading from {} to {}'.format(local_file_path, s3_file_path))
                    db.insert(table_name=bucket_name, source=local_file_path, destination=s3_file_path)
    except Exception as e:
        logger.error(e)


def download_from_s3(bucket_name='pcb-dataset'):
    if config['production']=='retrain_aifs':
        db = DBManager()
    else:
        db = DBManager(
            db_type=constant.DB_TYPE['S3'],
            endpoint="https://200-9090.aifs.ym.wise-paas.com",
            access_key="mDUtLL4OCkM4CjieIxlxozvTkpp8r06L",
            secret_key="Q3CwL8dYmX5i1jSbnSQrU29bHtRPr0BT",
            is_verify=False,
            buckets=[{
                'bucket': bucket_name,
                'blobs': {
                    'files': [],
                    'folders': ['train_data', 'test_data', 'retrain_data', 'models']
                }
            }]
        )

    try:
        db.connect()
        bucket_name = db.execute_query()[0]
        logger.info('bucket_name = {}'.format(bucket_name))
        return bucket_name
    except Exception as e:
        logger.error(e)
        return None

if __name__ == "__main__":
    bucket_name = download_from_s3(bucket_name='pcb-dataset')
    if bucket_name:
        write_config_yaml_with_key_value(config_file, 'pcb_dataset_dir', os.path.join(os.getcwd(), bucket_name))
    else:
        write_config_yaml_with_key_value(config_file, 'pcb_dataset_dir', None)
        exit(1)
