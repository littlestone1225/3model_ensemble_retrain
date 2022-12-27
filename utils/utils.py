#!/usr/bin/env python3
import os, sys, shutil
import logging
from logger import get_logger

# Get logger
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
logger = get_logger(name=__file__, console_handler_level=logging.INFO, file_handler_level=None)

def os_makedirs(dst_dir, keep_exists=False):
    if keep_exists:
        if not os.path.isdir(dst_dir):
            logger.info("makedirs {}".format(dst_dir))
            os.makedirs(dst_dir)
    else:
        if os.path.isdir(dst_dir):
            logger.info("Remove {}".format(dst_dir))
            shutil.rmtree(dst_dir)
        logger.info("makedirs {}".format(dst_dir))
        os.makedirs(dst_dir)

def os_remove(dst_file_path):
    if os.path.isfile(dst_file_path):
        logger.info("Remove {}".format(dst_file_path))
        os.remove(dst_file_path)

def shutil_rmtree(dst_dir):
    if os.path.isdir(dst_dir):
        logger.info("Remove {}".format(dst_dir))
        shutil.rmtree(dst_dir)

def shutil_copytree(src_dir, dst_dir):
    logger.info("Copy from {} to {}".format(src_dir, dst_dir))
    shutil.copytree(src_dir, dst_dir)

def shutil_copyfile(src_file_path, dst_file_path):
    logger.info("Copy from {} to {}".format(src_file_path, dst_file_path))
    shutil.copyfile(src_file_path, dst_file_path)

def shutil_copyfile_to_dir(src_file_path, dst_dir):
    src_file_name = os.path.basename(src_file_path)
    dst_file_path = os.path.join(dst_dir, src_file_name)
    logger.info("Copy from {} to {}".format(src_file_path, dst_file_path))
    shutil.copyfile(src_file_path, dst_file_path)

def shutil_move(src_file_path, dst_dir):
    src_file_name = os.path.basename(src_file_path)
    dst_file_path = os.path.join(dst_dir, src_file_name)
    if (os.path.exists(dst_file_path)):
        logger.info("Remove {}".format(dst_file_path))
        os.remove(dst_file_path)
    logger.info("Move from {} to {}".format(src_file_path, dst_dir))
    shutil.move(src_file_path, dst_dir)