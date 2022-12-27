#!/usr/bin/env python3
import sys
import logging
# https://titangene.github.io/article/python-logging.html
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
# FORMAT = '%(filename)s:%(lineno)d:%(levelname)s: %(message)s' # default = %(levelname)s:%(name)s:%(message)s
# logging.basicConfig(level=logging.INFO, format=FORMAT)
# logging.debug('debug message')
# logging.info('info message')
# logging.warning('warning message')
# logging.error('error message')
# logging.critical('critical message')

def get_logger(name=None, console_handler_level=logging.DEBUG, file_handler_level=None, file_name='logger.log'):
    default = 'root'
    if name:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(default)

    logger.setLevel(logging.DEBUG)

    if console_handler_level:
        console_formatter = logging.Formatter('%(filename)s:%(lineno)d:%(levelname)s: %(message)s')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(console_handler_level)
        logger.addHandler(console_handler)

    if file_handler_level:
        file_formatter = logging.Formatter('%(message)s')
        file_handler = logging.FileHandler(file_name, mode='w')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(file_handler_level)
        logger.addHandler(file_handler)
    # logger.propagate = False
    return logger