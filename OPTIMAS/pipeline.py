#!/usr/bin/env python

import os
import argparse

## own scripts
from OPTIMAS.merge_npy import merge_npy
from OPTIMAS.convert_npy_to_png import png_conversion
from OPTIMAS.make_video import make_video
#import OPTIMAS.trefide_pipeline as trefide


#########################
##### USAGE EXAMPLE #####
"""
python OPTIMAS/pipeline.py \
--input_data_folder /mnt/home_nas/jeremy/Recherches/Postdoc/Projects/Memory/Computational_Principles_of_Memory/optopatch/data/2020_03_02 \
--merge_npy \
--convert_npy_to_png \
--make_video
"""
#########################



## command line arguments
parser = argparse.ArgumentParser(description="analysis to run")
parser.add_argument("--input_data_folder",
                    help="input folder where the data to analyze is stored. It \
                          is the master folder with the date of the experiment.",
                    required=True)
parser.add_argument("--merge_npy",
                    action='store_true',
                    help="merge all the individual npy files into one bigger \
                          file for faster I/O",
                    required=True)
parser.add_argument("--convert_npy_to_png",
                    action="store_true" ,
                    help="this is to run the script png_conversion - convert \
                          npy files to images.")
parser.add_argument("--make_video",
                    action="store_true" ,
                    help="this is to run the script animate - convert \
                          images to video.")
parser.add_argument("--trefide_pipeline",
                    action="store_true",
                    help="this is to run the script trefide_pipeline - denoise \
                          demix and compress the data.")
args = parser.parse_args()

if args.input_data_folder:
    input_data_folder = args.input_data_folder
    print("working on data in :" + str(input_data_folder))

for experiment in next(os.walk(input_data_folder))[1]:
    print(f'\n -----------------{experiment}-----------------')

    if os.path.exists(f'{input_data_folder}/{experiment}/raw_data'):

        if args.merge_npy:
            if os.path.exists(f'{input_data_folder}/{experiment}/raw_data.npy'):
                print('merged file already exists')
            else:
                try:
                    print('generating merged npy files')
                    merge_npy(input_data_folder, experiment)
                except:
                    print('could not genete merged npy files')
                    
        if args.convert_npy_to_png:
            path_output_images = f'{input_data_folder}/{experiment}/images'
            if os.path.exists(path_output_images):
                print(f"image folder for {experiment} already exists")
                try:
                    if len(next(os.walk(f'{input_data_folder}/{experiment}/raw_data'))[2]) == len(next(os.walk(f'{input_data_folder}/{experiment}/images'))[2]):
                        print(f'image files for {experiment} already exist')
                    else:
                        print('image list is incomplete, need to re-run conversion')
                        png_conversion(input_data_folder, experiment)
                except: ## this is to catch the stopIteration error of the walk generator
                    pass
            else:
                print ("creating output image folder")
                os.mkdir(f'{input_data_folder}/{experiment}/images')
                print('converting raw npy files to images')
                png_conversion(input_data_folder, experiment)
        else:
             print("{} not converting npy files to png".format(experiment))

        if args.make_video:
            path_output_video = f'{input_data_folder}/{experiment}/{experiment}_raw.avi'
            if os.path.isfile(path_output_video):
                print('existing video file')
            else:
                print('creating video file')
                make_video(input_data_folder,
                           experiment,
                           'raw')
        else:
            print('not creating video file')

        if args.trefide_pipeline:
            try:
                path_output_processed_data = path_output_video
                video_name_denoised = path_output_processed_data+'{}_denoised.avi'.format(experiment)
                path_output_images_denoised = input_data_folder + '/{}/denoised_images/'.format(experiment)
                if os.path.exists(path_output_images_denoised):
                    print ("denoised image folder for {} already exists".format(experiment))
                    try:
                        if len(next(os.walk(input_data_folder + '/{}/raw_data'.format(experiment)))[2]) == len(next(os.walk(input_data_folder + '/{}/denoised_images'.format(experiment)))[2]):
                            print('denoised image files for {} already exist'.format(experiment))
                        else:
                            print('denoised image list is incomplete, need to re-run conversion')
                            try:
                                trefide.denoiser(path_input_npy, path_output_processed_data, path_output_images_denoised, 32, 32)
                            except:
                                trefide.denoiser(path_input_npy, path_output_processed_data, path_output_images_denoised, 8, 8)
                            animate.animate(path_output_images_denoised, video_name_denoised)
                    except:
                        pass
                else:
                    print ("creating folder {}".format(path_output_images_denoised))
                    os.mkdir(input_data_folder + '/{}/denoised_images'.format(experiment))
                    print('processing images through trefide pipeline')
                    trefide.denoiser(path_input_npy, path_output_processed_data, path_output_images_denoised, 8, 8)
                    animate.animate(path_output_images_denoised, video_name_denoised)
            except:
                print('cannot use trefide pipeline')
    else:
        print(('no raw folder in {}').format(experiment))

print("processing done")
