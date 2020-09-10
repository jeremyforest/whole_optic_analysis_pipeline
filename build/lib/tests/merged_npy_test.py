import pytest
import numpy as np
from utils.files_handling import images_list
from merge_npy import merge_npy



def check_merged_npy():

    # TODO find a way to make this dynamic or use a within package example folder
    input_data_folder = '/mnt/home_nas/jeremy/Recherches/Postdoc/Projects/Memory/Computational_Principles_of_Memory/optopatch/data/2020_03_02'
    experiment = 'experiment_132'
    output_data_folder = f'{input_data_folder}/{experiment}'
    files = images_list(path_input_npy_folder)
    merge_npy(input_data_folder, output_data_folder, experiment)
    json_file_path = f"{input_data_folder}/{experiment}/{experiment}_info.json"
    data = np.load(json_file_path)

    assert data.shape[0] == len(files)
