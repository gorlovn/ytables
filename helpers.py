#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from logging.handlers import RotatingFileHandler

CWD = os.getcwd()
LOG_PATH = os.path.join(CWD, 'log')
LOG_FORMATTER = logging.Formatter('%(asctime)s %(levelname)s %(message)s')  # include timestamp


def setup_logger(logger_name, log_file, log_path=LOG_PATH, level=logging.INFO, console_out=False):
    """
    Function setup as many loggers as you want
    https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings
    """

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    l_file = os.path.join(log_path, log_file)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    handler = RotatingFileHandler(l_file, maxBytes=100000, backupCount=5)
    handler.setFormatter(LOG_FORMATTER)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    if console_out:
        console = logging.StreamHandler()
        console.setLevel(level)
        logger.addHandler(console)

    return logger
