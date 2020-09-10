#!/usr/bin/env python

import os
import numpy as np
from tqdm import tqdm
from OPTIMAS.utils.files_handling import images_list, read_image_size


def merge_npy(input_data_folder, experiment):
    """
    merge all the individual npy files into one bigger file for faster I/O
    """
    path_input_npy_folder = f"{input_data_folder}/{experiment}/raw_data"
    output_data_folder = f'{input_data_folder}/{experiment}'
    json_file_path = f"{input_data_folder}/{experiment}/{experiment}_info.json"
    output = []
    files = images_list(path_input_npy_folder, 'npy')
    size_img = read_image_size(json_file_path)
    for file in tqdm(files):
        image = np.load(f"{path_input_npy_folder}/{file}").reshape(size_img[0], size_img[1])
        output.append(image)
    merged_npy_file_path = f"{output_data_folder}/raw_data.npy"
    np.save(merged_npy_file_path, np.array(output))


if __name__ == "__main__":

    input_data_folder = '/mnt/home_nas/jeremy/Recherches/Postdoc/Projects/Memory/Computational_Principles_of_Memory/optopatch/data/2020_03_02'
    experiment = 'experiment_132'
    output_path = f'{input_data_folder}/{experiment}'

    merge_npy(input_data_folder, output_path, experiment)
