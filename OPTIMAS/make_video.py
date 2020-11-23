#!/usr/bin/env python

import numpy as np
import cv2
import os
from tqdm import tqdm

from OPTIMAS.utils.files_handling import images_list, read_image_size

def make_video(input_data_folder, experiment, data_type, start_frame=0, end_frame=int(-1)):
    if data_type == 'raw':
        input_data_images = f'{input_data_folder}/{experiment}/images/'
        path_output_video = f'{input_data_folder}/{experiment}/{experiment}_raw.avi'
        filenames = images_list(input_data_images, 'png', 'raw')  ## needs this for as long as the name of the files are that way. Need to think about changing them with leading 0 or something
    elif data_type == 'denoised':
        input_data_images = f'{input_data_folder}/{experiment}/images_denoised/'
        path_output_video = f'{input_data_folder}/{experiment}/{experiment}_denoised.avi'
        filenames = images_list(input_data_images, 'png', 'denoised')
    if start_frame != 0 | end_frame != -1:
        filenames = filenames[start_frame:end_frame]
    json_file_path = f"{input_data_folder}/{experiment}/{experiment}_info.json"
    image_x, image_y = read_image_size(json_file_path)
    out = cv2.VideoWriter(path_output_video,
                          cv2.VideoWriter_fourcc(*'XVID'),
                          10,
                          #(image_x, image_y)
                          (image_y, image_x))
    try:
        # frame = 1
        for filename in tqdm(filenames):
            # filename = "image1.png"
            img = cv2.imread(f'{input_data_images}{filename}')
            # cv2.putText(img,f'frame {frame}',(int(image_x/10),int(image_y/5)),
                        # cv2.FONT_HERSHEY_SIMPLEX, .1,(255,255,255),
                        # 1)
            # frame += 1
            out.write(img)
        out.release()
    except :
        print('cannot create video file')

if __name__ == "__main__":

    # experiment = 'experiment_132'
    # input_data_folder = f'/mnt/home_nas/jeremy/Recherches/Postdoc/Projects/Memory/Computational_Principles_of_Memory/optopatch/data/2020_03_02'

    experiment = 'experiment_2'
    input_data_folder = f'/home/jeremy/Desktop/2020_11_20'

    # input_data_folder = '/home/jeremy/Documents/Postdoc/Projects/Memory/Computational_Principles_of_Memory/optopatch/data/2020_03_02/{}/denoised_images'.format(experiment)
    # path_output_video = '/home/jeremy/Documents/Postdoc/Projects/Memory/Computational_Principles_of_Memory/optopatch/data/2020_03_02/{}/{}_denoised.avi'.format(experiment, experiment)

    # make_video(input_data_folder, experiment, data_type = 'raw')
    make_video(input_data_folder, experiment, data_type = 'denoised')
