import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import json
import pdb
import argparse

from OPTIMAS.utils.files_handling import read_image_size

"""
how to use:
python merging_experiments.py \
--main_folder_path /media/jeremy/Data/local/Data_manip/2020_02_05/ \
--experiments 1 \
--experiments 10 \
--use_dlp_timing
"""

parser = argparse.ArgumentParser(description="options")
parser.add_argument("--main_folder_path", required=True, action="store",
                    help="main folder absolute path, somehting like /media/jeremy/Data/local/Data_manip/2020_02_05")
parser.add_argument("--experiments", required=True, action="append", type=int,
                    help="values of experiments to go over")
parser.add_argument("--use_dlp_timing", default=False, action="store_true",
                    help="if True, will use dlp activation as index for merging")
parser.add_argument("--use_laser_timing", default=False, action="store_true",
                    help="if True, will use laser activation as index for merging")
parser.add_argument("--manual_timings", action="append", type=int,
                    help="manually give the timings that will be used as index for merging")
parser.add_argument("--estimate_connections", default=False, action="store_true",
                    help="used to average images to check for network connections")
args = parser.parse_args()

merging = True

path = args.main_folder_path
# path = '/home/jeremy/Downloads/2020_11_05/'

## Expriments to merge
experiments = range(args.experiments[0], args.experiments[1])
# experiments = np.arange(71,90,1)

## if use_dlp_timing == false then need to manualy input
## the frames to synch on for each experiment
use_dlp_timing = args.use_dlp_timing
# use_dlp_timing = True
use_laser_timing = args.use_laser_timing
# use_laser_timing = False

manual_timings_laser = []
manual_timings_dlp = []

if args.estimate_connections:
    manual_timings_laser.append(0)
    manual_timings_dlp.append(0)

## expe 10-14, laser timing
if args.manual_timings != None:
    for timing in args.manual_timings:
        manual_timings_laser.append(timing)

def load_json_timing_data(path_output, experiment):
    """
    This function allows to load the json file where the timings of the laser, dlp, camera
    and ephys are stored.
    Input: path of the json file and the experiment folder number as input.
    Output: timings, each in their own variable
    """
    with open(f'{path_output}experiment_{experiment}_timings.json') as file:
        timings_data = dict(json.load(file))
    timings_dlp_on = timings_data['dlp']['on']
    timings_dlp_off = timings_data['dlp']['off']
    timings_laser_on = timings_data['laser']['on']
    timings_laser_off = timings_data['laser']['off']
    timings_camera_images = []  ## timings of the images as per the dcam api
    for images_timings in timings_data['camera']:
        for image_timing in images_timings:
            timings_camera_images.append(image_timing)

    timings_camera_images_bis = timings_data['camera_bis'] ## timing of the first frame as per manual clock
    print(f'timing difference between first image metadata and manual log is \
            {(timings_camera_images[0] - timings_camera_images_bis[0]) * 1000}')   ## ms difference

    return timings_dlp_on, timings_dlp_off, timings_camera_images, timings_laser_on, timings_laser_off, timings_camera_images_bis

import pdb; pdb.set_trace()
if merging:
    file_list_length = []
    frame_dlp_on = []
    frame_laser_on = []


    for experiment in experiments:
        # experiment = 4
        print(f'experiment - {experiment}')
        path_input = f"{path}experiment_{experiment}/raw_data/"
        path_json_file = f"{path}experiment_{experiment}/"

        if use_laser_timing:
            path_output = f"{path}experiment_merged_{experiments[0]}_{experiments[-1]}_laser/raw_data/"
        elif use_dlp_timing:
            path_output = f"{path}experiment_merged_{experiments[0]}_{experiments[-1]}_dlp/raw_data/"
        else:
            path_output = f"{path}experiment_merged_{experiments[0]}_{experiments[-1]}_manual/raw_data/"

        if os.path.exists(path_output):
            pass
        else:
            os.mkdir(path_output[:-9])
            os.mkdir(path_output)

        file_list = os.listdir(path_input)
        file_list_length.append(len(file_list))


        ### TIMING DATA ###
        timings_dlp_on, timings_dlp_off, \
        timings_camera_images, \
        timings_laser_on, timings_laser_off, \
        timings_camera_images_bis = load_json_timing_data(path_json_file, experiment)

        # to put both dlp and camera timings in the same format --> putting camera images in ns
        timings_camera_images = [timings_camera_images[i]*10**9 for i in range(len(timings_camera_images))]
        print(f'frame number: {len(timings_camera_images)}')

        takeClosest = lambda num,collection:min(collection,key=lambda x:abs(x-num))

        # putting dlp and laser timings back into camera time reference
        if use_dlp_timing:
            _timings_dlp_on = timings_camera_images[0] + (timings_camera_images[0] - timings_camera_images_bis[0]) + (timings_dlp_on[0] - timings_camera_images_bis[1])/1000
            _timings_dlp_off = timings_camera_images[0] + (timings_camera_images[0] - timings_camera_images_bis[0]) + (timings_dlp_off[0] - timings_camera_images_bis[1])/1000
        if use_laser_timing:
            _timings_laser_on = timings_camera_images[0] + (timings_camera_images[0] - timings_camera_images_bis[0]) + (timings_laser_on[0] - timings_camera_images_bis[1])/1000
            _timings_laser_off = timings_camera_images[0] + (timings_camera_images[0] - timings_camera_images_bis[0]) + (timings_laser_off[0] - timings_camera_images_bis[1])/1000

            # t = datetime.datetime.fromtimestamp(timings_camera_images[0])
            # (t-datetime.datetime(1970,1,1)).total_seconds()

        timings_dlp_on = []
        timings_dlp_off = []
        timings_laser_on = []
        timings_laser_off = []

        if use_dlp_timing:
            timings_dlp_on.append(_timings_dlp_on)
            timings_dlp_off.append(_timings_dlp_off)
        if use_laser_timing:
            timings_laser_on.append(_timings_laser_on)
            timings_laser_off.append(_timings_laser_off)

        if (use_dlp_timing and use_laser_timing == False):
            value_to_center_on = takeClosest(timings_dlp_on[0],timings_camera_images)
            frame_dlp_on.append(timings_camera_images.index(value_to_center_on))
            print(f'frame to center on: {timings_camera_images.index(value_to_center_on)}')
        elif (use_dlp_timing == False and use_laser_timing):
            value_to_center_on = takeClosest(timings_laser_on[0],timings_camera_images)
            frame_laser_on.append(timings_camera_images.index(value_to_center_on))
            print(f'frame to center on: {timings_camera_images.index(value_to_center_on)}')
        elif (use_dlp_timing and use_laser_timing):
            value_to_center_on_dlp = takeClosest(timings_dlp_on[0],timings_camera_images)
            value_to_center_on_laser = takeClosest(timings_laser_on[0],timings_camera_images)
            frame_dlp_on.append(timings_camera_images.index(value_to_center_on_dlp))
            frame_laser_on.append(timings_camera_images.index(value_to_center_on_laser))

    print(file_list_length)
    min_file_list_length = np.min(file_list_length)


    if (use_dlp_timing and use_laser_timing == False):
        frame_dlp_on = np.array(frame_dlp_on)
        shift_forward = np.max(frame_dlp_on) - frame_dlp_on ## shift forward will allow to keep data instead of truncate it when moving backward
        new_min_file_list_length = np.min(np.array(file_list_length) + shift_forward)
    elif (use_dlp_timing == False and use_laser_timing):
        frame_laser_on = np.array(frame_laser_on)
        shift_forward = np.max(frame_laser_on) - frame_laser_on ## shift forward will allow to keep data instead of truncate it when moving backward
        new_min_file_list_length = np.min(np.array(file_list_length) + shift_forward)
    elif (use_dlp_timing and use_laser_timing):
        frame_dlp_on = np.array(frame_dlp_on)
        frame_laser_on = np.array(frame_laser_on)
        difference_between_laser_and_dlp = frame_dlp_on - frame_laser_on
        print(difference_between_laser_and_dlp)
    else:
        print("using manual timings")
        frame_laser_on = np.array(manual_timings_laser)
        frame_dlp_on = manual_timings_dlp
        shift_forward = np.max(frame_laser_on) - frame_laser_on ## shift forward will allow to keep data instead of truncate it when plotting with 0 as dlp stimulation
        new_min_file_list_length = np.min(np.array(file_list_length) + shift_forward)


    for frame in range(new_min_file_list_length):
        # frame = 582
        for experiment, exp_nb in zip(experiments, range(len(experiments))):
            path_json_file = f"{path}experiment_{experiment}/"
            print(f"working on experiment {experiment} - {frame} / {new_min_file_list_length}")
            # experiment = 71
            path_input = f"{path}experiment_{experiment}/raw_data/"
            if (frame == 0 and experiment == experiments[exp_nb]):
                img_size = read_image_size(f'{path_json_file}experiment_{experiment}_info.json')
                _averaged_image = np.zeros((img_size[0]*img_size[1])) ## temporary image that will be used as storage buffer to average over experiments
            if frame <= shift_forward[experiment-experiments[exp_nb]]:
                img_array = np.zeros((img_size[0]*img_size[1]))  ## black image if no image
                print('padding with empty frame at the beginning')
            elif frame >= file_list_length[exp_nb]:
                img_array = np.zeros((img_size[0]*img_size[1]))  ## black image if no image
                print('padding with empty frame at the end')
            else:
                img_array = np.load(f"{path_input}image{frame-shift_forward[experiment-experiments[exp_nb]]}.npy") ## else load the actual image
                print(f"loaded image {frame-shift_forward[experiment-experiments[exp_nb]]}" )
                print("true image")

            averaged_image = _averaged_image + img_array/len(experiments)
            _averaged_image = averaged_image

            if experiment == experiments[-1]:
                np.save(f"{path_output}image{frame}", averaged_image)
                print ('image saved')

        _averaged_image = np.zeros((img_size[0]*img_size[1])) ## reinitialize image used as buffer
        print ('----')
    print(f"{path_output} done")
