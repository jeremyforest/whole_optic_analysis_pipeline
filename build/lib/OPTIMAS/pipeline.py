#!/usr/bin/env python

import os
import argparse
import pdb

## own scripts
import convert_npy_to_png
import make_video
import trefide_pipeline as trefide

## command line arguments
parser = argparse.ArgumentParser(description="analysis to run")
parser.add_argument("--input_data_folder", help="input folder where the data to analyze is stored. It is the master folder with the date of the experiment.")
parser.add_argument("--merge_npy", help="merge all the individual npy files into one bigger file for faster I/O")
parser.add_argument("--convert_npy_to_png", action="store_true" , help="this is to run the script png_conversion - convert npy files to images.")
parser.add_argument("--make_video", action="store_true" , help="this is to run the script animate - convert images to video.")
parser.add_argument("--trefide_pipeline", action="store_true" , help="this is to run the script trefide_pipeline - denoise, demix and compress the data.")
args = parser.parse_args()

if args.input_data_folder:
    # input_data_folder = '/home/jeremy/Documents/Postdoc/Projects/Memory/Computational_Principles_of_Memory/optopatch/data/2020_03_02'
    input_data_folder = args.input_data_folder
    print("working on data in :" + str(input_data_folder))

# experiment= 'experiment_3'
for experiment in next(os.walk(input_data_folder))[1]:
    print('\n {}'.format(experiment))
    if os.path.exists(input_data_folder + '/{}/raw_data'.format(experiment)):

        if args.convert_npy_to_png:
            path_input_npy = input_data_folder + '/{}/raw_data'.format(experiment)
            path_output_images = input_data_folder + '/{}/images/'.format(experiment)
            if os.path.exists(path_output_images):
                print ("image folder for {} already exists".format(experiment))
                try:
                    if len(next(os.walk(input_data_folder + '/{}/raw_data'.format(experiment)))[2]) == len(next(os.walk(input_data_folder + '/{}/images'.format(experiment)))[2]):
                        print('image files for {} already exist'.format(experiment))
                    else:
                        print('image list is incomplete, need to re-run conversion')
                        png.png_conversion(path_input_npy, path_output_images)
                except: ## this is to catch the stopIteration error of the walk generator
                    pass
            else:
                print ("creating folder {}".format(path_output_images))
                os.mkdir(input_data_folder + '/{}/images'.format(experiment))
                print('converting raw npy files to images')
                png.png_conversion(path_input_npy, path_output_images)
        else:
             print("{} not converting npy files to png".format(experiment))

        if args.make_video:
            path_input_images = input_data_folder + '/{}/images'.format(experiment)
            path_output_video =  input_data_folder + '/{}/'.format(experiment)
            video_name = path_output_video+'{}.avi'.format(experiment)
            if os.path.isfile(video_name):
                print('existing video file')
            else:
                print('creating video file')
                animate.animate(path_input_images, video_name)
        else:
            print('not creating video file')

        if args.trefide_pipeline:
            try:
                # import pdb; pdb.set_trace()
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
