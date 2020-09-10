#!/usr/bin/env python

import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from OPTIMAS.utils.files_handling import read_image_size


def png_conversion(input_data_folder, experiment):
    '''
    takes the images in npy form and convert them into png format for
    visualization purposes.
    inpupt: path of the folder that contains all the experiments (it is the
            folder with the date of the experiments)
    output: save the images in input_data_folder/experiment/images/
    '''
    files = os.listdir(f'{input_data_folder}/{experiment}/raw_data/')
    for file in tqdm(files):
        #file=files[0]
        if file.endswith('.npy'):
            try:
                img_array = np.load(f'{input_data_folder}/{experiment}/raw_data/{file}')
                json_file_path = f"{input_data_folder}/{experiment}/{experiment}_info.json"
                image_x, image_y = read_image_size(json_file_path)
                img_array = img_array.reshape(image_x, image_y)
                output_path = f'{input_data_folder}/{experiment}/images/'
                img_name = output_path+file.replace('.npy', '.png')
                plt.imsave(img_name, img_array, cmap='gray')
            except:
                print(f'cannot convert{str(file)}')


if __name__ == "__main__":

    experiment = 'experiment_132'
    path_input = f'/mnt/home_nas/jeremy/Recherches/Postdoc/Projects/Memory/Computational_Principles_of_Memory/optopatch/data/2020_03_02'

    try:
        os.mkdir(f'{path_input}/{experiment}/images/')
    except FileExistsError:
        pass

    png_conversion(input_data_folder = path_input,
                    output_data_folder = path_output)
