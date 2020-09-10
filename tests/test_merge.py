#!/usr/bin/env python

import os
import numpy as np

from OPTIMAS.merge_npy import merge_npy
from OPTIMAS.utils.files_handling import images_list, read_image_size

def test_merge_npy():
    input_data_folder = 'tests/test_data/2020_09_03'
    experiment = 'experiment_1'
    output_path = f'{input_data_folder}/{experiment}'

    merge_npy(input_data_folder, output_path, experiment)

    merged_npy_path = f"{output_path}/raw_data.npy"
    merged_npy = np.load(merged_npy_path)

    json_file_path = f"{input_data_folder}/{experiment}/{experiment}_info.json"
    image_x, image_y = read_image_size(json_file_path)

    os.remove(merged_npy_path)

    assert merged_npy.shape[0] == len(os.listdir(
                                f"{input_data_folder}/{experiment}/raw_data"))
    assert merged_npy.shape[1] == image_x
    assert merged_npy.shape[2] == image_y
