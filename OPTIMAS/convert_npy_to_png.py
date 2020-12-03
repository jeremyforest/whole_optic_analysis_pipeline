#!/usr/bin/env python

import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

from OPTIMAS.utils.files_handling import read_image_size


def png_conversion(input_data_folder, experiment):
    '''
    takes the images in npy format and convert them into png format for
    visualization purposes. One npy file is outputed as one png image.
    inpupt: path of the folder that contains all the experiments (it is the
            folder with the date of the experiments)
    output: save the images in input_data_folder/experiment/images/
    '''
    files = os.listdir(f'{input_data_folder}/{experiment}/raw_data/')
    for file in tqdm(files):
        #file=files[100]
        if file.endswith('.npy'):
            try:
                img_array = np.load(f'{input_data_folder}/{experiment}/raw_data/{file}')
                json_file_path = f"{input_data_folder}/{experiment}/{experiment}_info.json"
                image_x, image_y = read_image_size(json_file_path)
                #####################################################
                ##### TO CHANGE IN UPDATED PIPELINE VERSION #########
                # img_array = img_array.reshape(image_x, image_y)
                img_array = img_array.reshape(image_y, image_x)
                ####################################################
                output_path = f'{input_data_folder}/{experiment}/images/'
                img_name = output_path+file.replace('.npy', '.png')
                plt.imsave(img_name, img_array, cmap='gray')
            except:
                print(f'cannot convert{str(file)}')

def png_conversion_from_one_npy(input_data_folder, experiment):
    input_file = np.load(f'{input_data_folder}/{experiment}/denoised_data.npy')
    # input_file = np.load(f'{input_data_folder}/{experiment}/comparison_data.npy')
    input_file.shape
    json_file_path = f"{input_data_folder}/{experiment}/{experiment}_info.json"
    image_x, image_y = read_image_size(json_file_path)
    output_path = f'{input_data_folder}/{experiment}/images_denoised'
    if os.path.exists(output_path):
        pass
    else:
        os.mkdir(output_path)
    for image in tqdm(range(input_file.shape[-1])):
        img_name = f'{output_path}/image_denoised{image}.png'
        img_array = input_file[:,:,image]
        plt.imsave(img_name, img_array, cmap='gray')




if __name__ == "__main__":

    # experiment = 'experiment_merged_3_19_manual'
    # input_data_folder = f'/home/jeremy/Downloads/2020_03_06'

    experiment = 'experiment_347'
    input_data_folder = f'/home/jeremy/Desktop/2020_11_23'

    # experiment = 'experiment_71'
    # input_data_folder = f'/media/jeremy/Seagate Portable Drive/data/2020_11_05'


    try:
        os.mkdir(f'{input_data_folder}/{experiment}/images/')
    except FileExistsError:
        pass

    png_conversion(input_data_folder = input_data_folder,
                    experiment = experiment)

    # png_conversion_from_one_npy(input_data_folder = input_data_folder,
    #                             experiment = experiment)
