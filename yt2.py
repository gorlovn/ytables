#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Этап второй:
 Из полученной на первом этапе промежуточной таблицы
 сформировать результат
"""
import os
import time
import logging

import torch
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

from helpers import setup_logger

if __name__ == "__main__":
    log = setup_logger('', 'yt2.log', console_out=True)
else:
    log = logging.getLogger(__name__)

CWD = os.getcwd()
DATA_PATH = os.path.join(CWD, 'data')
YAML_FILE = "yt2.yaml"
EXT_DF_FILE_NAME = 'ext_df.xlsx'
RESULT_DF_FILE_NAME = 'result_df.xlsx'


def ext_df_2_result_df(_ext_df_file, _result_df_file, _data_path=DATA_PATH,
                       _line_start=None, _line_end=None, _columns=None):
    import pandas

    _ext_df_path = os.path.join(_data_path, _ext_df_file)
    if not os.path.isfile(_ext_df_path):
        log.error(f"Не найден файл {_ext_df_path}")
        return None

    ext_df = pandas.read_excel(_ext_df_path)

    _c_names, _c_pos = [], []
    for _key, _val in _columns.items():
        _c_names.append(_key)
        _c_pos.append(_val)

    _c_data = []  # данные со столбцами
    _nc = len(_columns)
    _row_data = [None] * _nc
    _current_line = None

    _first_line = 0 if _line_start is None else _line_start

    for _, _row in ext_df.iterrows():

        if _row.line_num < _first_line:
            continue

        if _line_end is not None and _row.line_num > _line_end:
            break

        if _row.line_num != _current_line:
            if _current_line is not None:
                _c_data.append(_row_data)
                _row_data = [None] * _nc
            _current_line = _row.line_num
            _current_idx = None
            _current_cell = ""

        _left_pos = _row.left
        # определяем индекс столбца
        _idx = None
        _i = 0
        for _pos in _c_pos:
            if _left_pos > _pos:
                _idx = _i
            if _left_pos < _pos:
                break

            _i += 1

        if _idx is not None:
            if _current_idx is None:
                _current_idx = _idx
            if _current_idx == _idx:
                if len(_current_cell) > 0:
                    _current_cell += ' '
                _current_cell += _row.text
            else:
                _row_data[_current_idx] = _current_cell
                _current_idx = _idx
                _current_cell = _row.text

        if _idx is not None:
            _row_data[_idx] = _current_cell

    _c_data.append(_row_data)
    result_df = pandas.DataFrame(_c_data, columns=_c_names)
    _result_df_path = os.path.join(_data_path, _result_df_file)
    result_df.to_excel(_result_df_path)

    log.info(f"Сохранили результат в файл {_result_df_path}")

    return result_df


if __name__ == "__main__":
    """
    1. Имя файла с параметрами
    """
    import sys
    import yaml

    yaml_file = YAML_FILE
    n_args = len(sys.argv)
    if n_args > 1:
        yaml_file = sys.argv[1]

    yaml_path = os.path.join(DATA_PATH, yaml_file)
    if not os.path.isfile(yaml_path):
        log.error(f"Не найден конфигурационный файл {yaml_path}")
        sys.exit(1)

    with open(yaml_path, "r") as f:
        args = yaml.safe_load(f)

    ext_df_file = args.get('ext_df_file', EXT_DF_FILE_NAME)
    result_df_file = args.get('result_df_file', RESULT_DF_FILE_NAME)
    line_start = args.get('line_start')
    line_end = args.get('line_end')
    columns = args.get('columns')
    if type(columns) is not dict:
        log.error(f"Столбцы (columns) не определены в файле {yaml_path}")
        sys.exit(1)

    rr = ext_df_2_result_df(ext_df_file, result_df_file,
                            _line_start=line_start,
                            _line_end=line_end,
                            _columns=columns)

    sys.exit(0)
