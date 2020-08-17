

import os
import numpy as np
from tqdm import tqdm

def images_list(folder):
    # folder = path_input
    files_len = len(os.listdir(folder))
    image_list = []
    for i in range(files_len):
        image_name = 'image'+ str(i) + '.npy'
        image_list.append(image_name)
    return image_list


def merge_npy(path_input_npy_folder, output_data_folder):
    files = images_list(path_input_npy_folder)
    output = []
    for file in tqdm(files):
        # file=files[0]
        size_img = (128,128)
        image = np.load(path_input_npy_folder + '/' + file).reshape(size_img[0], size_img[1])
        output.append(image)
    output = np.array(output)
    print(output.shape)
    np.save(f'{output_path}raw_data.npy', output)

if __name__ == "__main__":

    experiment = 'experiment_132'
    input_data_folder = '/mnt/home_nas/jeremy/Recherches/Postdoc/Projects/Memory/Computational_Principles_of_Memory/optopatch/data/2020_03_02'
    path_input_npy_folder = input_data_folder + '/{}/raw_data'.format(experiment)
    output_path = input_data_folder + '/{}/'.format(experiment)

    merge_npy(path_input_npy_folder, output_path)
