#!/usr/bin/env python

import os
import argparse

## own scripts
from OPTIMAS.merge_npy import merge_npy
from OPTIMAS.convert_npy_to_png import png_conversion, png_conversion_from_one_npy
from OPTIMAS.make_video import make_video
#import OPTIMAS.trefide_pipeline2 as trefide
import OPTIMAS.time_serie_pixel_analysis as analysis
import OPTIMAS.interactive_plotting as interactive


#########################
##### USAGE EXAMPLE #####
"""
python OPTIMAS/pipeline.py \
--input_data_folder /mnt/home_nas/jeremy/Recherches/Postdoc/Projects/Memory/Computational_Principles_of_Memory/optopatch/data/2020_03_02 \
--merge_npy \
--convert_npy_to_png \
--make_video \
--roi_analysis \
--trefide_pipeline \
--overwrite
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
parser.add_argument("--roi_analysis",
                    action='store_true',
                    help='this is to run the script time_serie_pixel_analysis \
                          plot time_series of defined ROIs')
parser.add_argument("--trefide_pipeline",
                    action="store_true",
                    help="this is to run the script trefide_pipeline - denoise \
                          demix and compress the data.")
parser.add_argument('--overwrite',
                    action = 'store_true',
                    help = 'true if want to overwrite the data')
args = parser.parse_args()

if args.input_data_folder:
    input_data_folder = args.input_data_folder
    print("working on data in :" + str(input_data_folder))

for experiment in next(os.walk(input_data_folder))[1]:
    print(f'\n ================ preprocessing {experiment} ================')

    if os.path.exists(f'{input_data_folder}/{experiment}/raw_data'):

        if args.merge_npy:
            print(f'\n == step 1: merging file ==')
            if os.path.exists(f'{input_data_folder}/{experiment}/raw_data.npy'):
                print('merged file already exists')
                if args.overwrite :
                    print('overwriting')
                    merge_npy(input_data_folder, experiment)
            else:
                try:
                    print('generating merged npy files')
                    merge_npy(input_data_folder, experiment)
                except:
                    print('could not genete merged npy files')

        if args.convert_npy_to_png:
            print(f'\n == step 2: convert to png ==')
            path_output_images = f'{input_data_folder}/{experiment}/images'
            if os.path.exists(path_output_images):
                print(f"image folder for {experiment} already exists")
                try:
                    if len(next(os.walk(f'{input_data_folder}/{experiment}/raw_data'))[2]) == len(next(os.walk(f'{input_data_folder}/{experiment}/images'))[2]):
                        print(f'image files for {experiment} already exist')
                        if args.overwrite:
                            print('overwriting files')
                            png_conversion(input_data_folder, experiment)
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
            try:
                print(f'\n == step 3: making video ==')
                path_output_video = f'{input_data_folder}/{experiment}/{experiment}_raw.avi'
                if os.path.isfile(path_output_video):
                    print('existing video file')
                    if args.overwrite:
                        make_video(input_data_folder,
                                   experiment,
                                   data_type = 'raw')
                else:
                    print('creating video file')
                    make_video(input_data_folder,
                               experiment,
                               data_type = 'raw')
            except:
                print('could not creating video file')
        else:
            print('not creating video file')

        if args.roi_analysis:
            print(f'\n == step 4: performing analysis ==')
            try:
                print('static plots')
                analysis.time_serie(input_data_folder, experiment,
                                    data_type='raw', timing=False,
                                    draw_laser_timings=False,
                                    draw_dlp_timings=False,
                                    time_start=0, time_stop=int(-1))
                print('interactive plots')
                interactive.interactive_plotting(input_data_folder, experiment)
            except:
                print('error while analysing time series of this experiment')

        if args.trefide_pipeline:
            print(f'\n == step 5: denoising data ==')
            try:
                if os.path.isfile(f'{input_data_folder}/{experiment}/denoised_data.npy'):
                    print("denoised data file already exists")
                else:
                    print('denoising data')
                    trefide.denoising_pipeline(input_data_folder, experiment)
                    print('generating images from numpy files for denoised data')
                    png_conversion_from_one_npy(input_data_folder, experiment)
                    print('generating video file for denoised data')
                    make_video(input_data_folder,
                               experiment,
                               data_type='denoised',
                               start_frame=0,
                               end_frame=int(-1))
            except:
                print('error while processing trefide pipeline')

    print(f"================  preprocessing done {experiment}  ================")
