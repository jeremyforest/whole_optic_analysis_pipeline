#!/usr/bin/env python


import numpy as np
import cv2
import scipy
import os
from tqdm import tqdm

# input_data_folder = '/media/jeremy/Data/Data_Jeremy/20_08_2019/experiment_36/images'
# path_output_video = '/media/jeremy/Data/Data_Jeremy/20_08_2019/experiment_36/experiment_36.avi'


def images_list(folder):
    files_len = len(os.listdir(folder))
    image_list = []
    for i in range(files_len):
        image_name = 'image'+ str(i) + '.png'
        image_list.append(image_name)
    return image_list

def animate(input_data_folder, path_output_video):
    filenames = images_list(input_data_folder)  ## needs this for as long as the name of the files are that way. Need to think about changing them with leading 0 or something
    img_array = []
    try:
        for filename in tqdm(filenames):
            # filename = "image1.png"
            img = cv2.imread(input_data_folder+'/'+filename)
            h, v, z = img.shape
            size = (h,v)
            img_array.append(img)
        out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'I420'), 30, size)

        for i in tqdm(range(len(img_array))):
            out.write(img_array[i])
        out.release()

    except :
        print('no video file to create')

if __name__ == "__main__":

    experiment = 'experiment_31'

    # input_data_folder = '/media/jeremy/Data/Data_Jeremy/20_08_2019/{}/images'.format(experiment)
    # path_output_video = '/media/jeremy/Data/Data_Jeremy/20_08_2019/{}/{}.avi'. format(experiment, experiment)

    input_data_folder = '/media/jeremy/Data/Data_Jeremy/2019_10_12/{}/denoised_images'.format(experiment)
    path_output_video = '/media/jeremy/Data/Data_Jeremy/2019_10_12/{}/{}_denoised.avi'.format(experiment, experiment)

    # input_data_folder = '/media/jeremy/Data/Data_Jeremy/20_08_2019/{}/comparison_images'.format(experiment)
    # path_output_video = '/media/jeremy/Data/Data_Jeremy/20_08_2019/{}/{}_comparison.avi'. format(experiment, experiment)

    animate(input_data_folder, path_output_video)
