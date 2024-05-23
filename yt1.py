#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract table from pdf using YOLO LLM model
https://huggingface.co/keremberke/yolov8m-table-extraction

https://iamrajatroy.medium.com/document-intelligence-series-part-1-table-detection-with-yolo-1fa0a198fd7

Этап первый:
1. Преобразовать страницу pdf в изображение
2. С помощью LLM модели YOLO определить где на изображении находится таблица и обрезать изображение.
3. Выполнить распознавание текста на странице с использованием pytesseract
   и сохранить промежуточный результат в xlsx файл.

"""
import os
import time
import logging

import torch
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

from helpers import setup_logger

if __name__ == "__main__":
    log = setup_logger('', 'yt1.log', console_out=True)
else:
    log = logging.getLogger(__name__)

CWD = os.getcwd()
DATA_PATH = os.path.join(CWD, 'data')
IMAGES_PATH = os.path.join(CWD, 'images')
EXT_DF_FILE_NAME = 'ext_df.xlsx'

if not os.path.isdir(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)


def convert_pdf_to_image(file_name, first_page=None, last_page=None,
                         data_path=DATA_PATH, images_path=IMAGES_PATH):
    from pdf2image import convert_from_path

    _file_path = os.path.join(data_path, file_name)
    if not os.path.isfile(_file_path):
        log.error(f"Не найден файл {_file_path}")
        return None

    log.info(f"Преобразование файла {_file_path} в изображения")
    log.info(f"Страницы: {first_page} - {last_page}")
    log.info(f"Папка с изображениями: {images_path}")

    _start = time.time()
    _images = convert_from_path(_file_path, output_folder=images_path,
                                first_page=first_page, last_page=last_page,
                                fmt='jpg')
    _ni = len(_images) if type(_images) is list else None
    log.info(f"Получено изображений: {_ni}")

    _dur = (time.time() - _start) * 1000
    log.info(f"Время обработки: {_dur:.2f} ms")

    return _images


def clear_gpu_memory():
    """
    # https://stackoverflow.com/questions/59129812/how-to-avoid-cuda-out-of-memory-in-pytorch
    :return:
    """
    torch.cuda.empty_cache()
    gc.collect()
    # del variables


def wait_until_enough_gpu_memory(min_memory_available, max_retries=10, sleep_time=5):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    for _ in range(max_retries):
        info = nvmlDeviceGetMemoryInfo(handle)
        if info.free >= min_memory_available:
            break
        print(f"Waiting for {min_memory_available} bytes of free GPU memory. Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
    else:
        raise RuntimeError(f"Failed to acquire {min_memory_available} bytes of free GPU memory after {max_retries} retries.")


def extract_tables(image, _ext_df_file_name=EXT_DF_FILE_NAME,  _data_path=DATA_PATH):
    import numpy as np
    import pandas as pd
    from ultralyticsplus import YOLO, render_result
    from PIL import Image
    import pytesseract
    from pytesseract import Output

    min_memory_available = 20 * 1024 * 1024  # 20MB
    clear_gpu_memory()
    wait_until_enough_gpu_memory(min_memory_available)

    # load model
    model = YOLO('keremberke/yolov8m-table-extraction')

    # set model parameters
    model.overrides['conf'] = 0.25  # NMS confidence threshold
    model.overrides['iou'] = 0.45  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # maximum number of detections per image

    # perform inference
    results = model.predict(image)

    # observe results
    print(results[0].boxes)
    render = render_result(model=model, image=image, result=results[0])
    # render.show()

    x1, y1, x2, y2, _, _ = tuple(int(item) for item in results[0].boxes.cpu().data.numpy()[0])
    img = np.array(image)
    # cropping
    cropped_image = img[y1:y2, x1:x2]
    cropped_image = Image.fromarray(cropped_image)

    log.info("--- Start OCR process ...")
    _start = time.time()
    ext_df = pytesseract.image_to_data(cropped_image, output_type=Output.DATAFRAME, config="--psm 6 --oem 3",
                                       lang="chi_sim")
    log.info("OCR finished")
    _dur = (time.time() - _start) * 1000
    log.info(f"Время обработки: {_dur:.2f} ms")

    # Запишем промежуточный результат
    _ext_df_file_path = os.path.join(_data_path, _ext_df_file_name)
    ext_df.to_excel(_ext_df_file_path)
    log.info(f"Записали файл промежуточных результатов: {_ext_df_file_path}")

    return ext_df


if __name__ == "__main__":
    """
    1. имя pdf файла
    2. начальная страница
    3. конечная страница
    """
    import sys

    PDF_FILE = '新疆统计年鉴2018（O）.pdf'
    FIRST_PAGE = 58
    LAST_PAGE = 59

    pdf_file, p1, p2 = PDF_FILE, FIRST_PAGE, LAST_PAGE

    n_args = len(sys.argv)
    if n_args > 1:
        pdf_file = sys.argv[1]
        if n_args > 2:
            p1 = int(sys.argv[2])
            if n_args > 3:
                p2 = int(sys.argv[3])

    images = convert_pdf_to_image(pdf_file, p1, p2)

    rr = extract_tables(images[0])

    sys.exit(0)
