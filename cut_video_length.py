import os
import numpy as np
from tqdm import tqdm

from animate import animate

def cut_length(path_input_npy, path_output, image_start, image_end):
    image_start = 602
    image_end = 1370
    ## files imports
    Ytemp = []
    files = os.listdir(path_input_npy)
    for file in tqdm(files):
        if file.endswith('.npy'):
                img_array= np.load(path_input_npy+"/"+file)
                Ytemp.append([img_array])
    Y = Ytemp[image_start:image_end]
    return Y


if __name__ == "__main__":

    experiment = 'experiment_132'

    input_data_folder = '/home/jeremy/Documents/Postdoc/Projects/Memory/Computational_Principles_of_Memory/optopatch/data/2020_03_02'
    path_input_npy = input_data_folder + '/{}/raw_data'.format(experiment)
    path_output_data = input_data_folder + '/{}/'.format(experiment)
    path_output_images_denoised = input_data_folder + '/{}/denoised_images'.format(experiment)

    cut_data = cut_length(path_input_npy, path_output, 602, 1370)
