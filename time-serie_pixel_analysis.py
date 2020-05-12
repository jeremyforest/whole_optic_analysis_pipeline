import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import os
import pickle
import json
from roipoly.roipoly import RoiPoly, MultiRoi
import argparse
import datetime

"""
how to use with no timings:
python time-serie_pixel_analysis.py \
--main_folder_path /media/jeremy/Data/local/Data_manip/2020_02_05/ \
--experiments 1 \
--experiments 10 \
--time 300 \
--time 900

with timings:
python time-serie_pixel_analysis.py \
--main_folder_path /media/jeremy/Data/local/Data_manip/2020_02_05/ \
--experiments 1 \
--experiments 10 \
--timings \
--draw_laser_timings \
--draw_dlp_timings \
--time 300 \
--time 900
"""

parser = argparse.ArgumentParser(description="options")
parser.add_argument("--main_folder_path", required=True, action="store" ,
                        help="main folder absolute path, something like /media/jeremy/Data/local/Data_manip/2020_02_05")
parser.add_argument("--experiments", required=True, action="append", type=int,
                        help="values of experiments to go over")
parser.add_argument("--denoised", default=False, action="store_true",
                        help="if True, will use denoised data")
parser.add_argument("--timings", default=False, action="store_true",
                        help="if True, will use time as index; otherwise uses frame")
parser.add_argument("--draw_laser_timings", default=False, action="store_true",
                        help="if set to true: will draw laser timings on plot")
parser.add_argument("--draw_dlp_timings", default=False, action="store_true",
                        help="if set to true: will draw laser timings on plot")
parser.add_argument("--time", action="append", type=int,
                        help="start and end frame of window used for plotting data")
args = parser.parse_args()




main_folder_path = args.main_folder_path
# main_folder_path = '/media/jeremy/Data/local/Data_manip/2020_03_02/'
experiments = range(args.experiments[0], args.experiments[1]+1)
# experiment = 1
# experiment = 'merged_121_150_dlp'
denoised_images = args.denoised
# denoised_images = False
timing = args.timings
# timing = True
draw_laser_timings = args.draw_laser_timings
# draw_laser_timings = True
draw_dlp_timings = args.draw_dlp_timings
# draw_dlp_timings = True
if args.time != None:
    time_init = args.time[0]
if args.time != None:
    time_end = args.time[1]


fps = 400

def images_list(folder):
    """
    Generate an ordered list of images that will be used for subsequent analysis
    based on the number of files in the desired folder.
    Necessary because of the numbering of the files which makes the extracting based
    on name un-ordered.
    Input: folder where are stored the images
    Output: image list ordered from 0 to last.
    """
    # folder = path_input
    files_len = len(os.listdir(folder))
    image_list = []
    if args.time == None:
        for i in range(files_len):
            image_name = 'image'+ str(i) + '.png'
            image_list.append(image_name)
    else:
        for i in range(time_init,time_end):
            image_name = 'image'+ str(i) + '.png'
            image_list.append(image_name)
    return image_list

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


print('processing experiments')
for experiment in tqdm(experiments):
    data_export = [] # placeholder for saving data at the end

    ### PATHS ###
    print(f"processing experiment - {experiment}")
    path_input = f"{main_folder_path}experiment_{experiment}/images/"
    path_input_denoised = f"{main_folder_path}experiment_{experiment}/denoised_images/"
    path_output = f"{main_folder_path}experiment_{experiment}/"
    roi_file_path = path_output + 'roi_masks.txt'

    ### use raw or denoised data ###
    if denoised_images:
        images = images_list(path_input_denoised)
    else:
        images = images_list(path_input)
        # images = image_list

    #############
    ### ROIS ####
    #############
    if os.path.isfile(roi_file_path):
        print('ROI file exists')
        with open(roi_file_path, "rb") as file:
            rois = pickle.load(file)
    else:
        print('ROI file doesnt exists')

        if denoised_images:  ## denoised image size is not the same as the raw image
            w,h = cv2.imread(path_input_denoised+images[0],cv2.IMREAD_GRAYSCALE).shape
        else:
            w,h = cv2.imread(path_input+images[0],cv2.IMREAD_GRAYSCALE).shape

        ### ref image for defining rois, need to think about what it can be. The best would be a DIC image ?
        ### use an automatic segmentation algorithm if possible with a DIC image ?
        image = cv2.imread(path_output + 'neurons.png'.format(experiment), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (w,h))
        # Show the image
        fig = plt.figure()
        plt.imshow(image, interpolation='none', cmap='gray')
        plt.title("Click on the button to add a new ROI")
        # Draw multiple ROI
        multiroi_named = MultiRoi(roi_names=['Background', 'ROI 1', 'ROI 2', 'ROI 3', 'ROI 4', 'ROI 5',
                                                'ROI 6', 'ROI 7', 'ROI 8', 'ROI 9', 'ROI 10', 'ROI 11',
                                                'ROI 12', 'ROI 13', 'ROI 14', 'ROI 15', 'ROI 16', 'ROI 17'])
        # Draw all ROIs
        plt.imshow(image, interpolation='none', cmap='gray')
        rois = []
        for name, roi in multiroi_named.rois.items():
            roi.display_roi()
            # roi.display_mean(image)
            mask = roi.get_mask(image)
            rois.append([name, mask])
        plt.legend()
        plt.savefig(path_output+'rois.png')
        plt.close()
        ## writing rois to disk
        with open(roi_file_path, "wb") as file:
            pickle.dump(rois, file)


    rois_signal = []
    ## not the most optimized, would be better to log every roi in each image than load every image multiple times
    for roi in rois:
        tmp_time_serie_roi = []
        for image in tqdm(images):
            # image = 'image0.npy'
            if denoised_images:
                img = cv2.imread(path_input_denoised+image,cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(path_input+image,cv2.IMREAD_GRAYSCALE)
            mask = roi[1]
            roi_average = np.mean(img[mask])
            tmp_time_serie_roi.append(roi_average)
        rois_signal.append(tmp_time_serie_roi)
    print ('generating data plots')



    ### TIMING DATA ###
    if timing:
        timings_dlp_on, timings_dlp_off, timings_camera_images, \
        timings_laser_on, timings_laser_off, \
        timings_camera_images_bis = load_json_timing_data(path_output, experiment)

        # timings_camera_images_bis.append(660) ## timings perf_counter equivalent to unix timestamp

        if len(timings_dlp_on)>1:
            print('more than 1 timing from DLP ON')
            timings_dlp_on = [timings_dlp_on[0]]

        ## for diagnostic plot for the times of the camera dcam api metadata
        # plt.plot(np.array(timings_camera_images), range(0,len(timings_camera_images)))

        ## use the timings metadata of the dcap api  ## for now broken, replaced with manual incrementation
        #timings_camera_images_new = timings_camera_images[time_init : time_end] ## back to this when solved problem of metadata from the dcam api
        timings_camera_images_new = []
        timings_camera_images_new.append(timings_camera_images[0])
        for nb_of_times in range(1,len(timings_camera_images)):
            fps = fps ## to get for each expe
            timings_camera_images_new.append(timings_camera_images[0] + (1/fps * nb_of_times))
        ## diagnostic
        # plt.plot(np.array(timings_camera_images_new), range(0,len(timings_camera_images_new)))

        timings_camera_images = timings_camera_images_new
        print(f'number of camera images: {len(timings_camera_images)}')
        print(f'number of points in the each roi signal: {len(rois_signal[0])}')
        assert len(images) == len(timings_camera_images), 'not the same number of images and images timing metadata'  ## will not work when working on subset of the data

        ## to put both dlp, laser and camera timings in the same format
        ## putting dlp and laser time refs back into camera ref
        #timings_camera_images = [timings_camera_images[i]*10**9 for i in range(len(timings_camera_images))]  ##for dcam api meta
        _timings_dlp_on = timings_camera_images[0] + (timings_camera_images[0] - timings_camera_images_bis[0]) + (timings_dlp_on[0] - timings_camera_images_bis[1])/1000
        _timings_dlp_off = timings_camera_images[0] + (timings_camera_images[0] - timings_camera_images_bis[0]) + (timings_dlp_off[0] - timings_camera_images_bis[1])/1000
        _timings_laser_on = timings_camera_images[0] + (timings_camera_images[0] - timings_camera_images_bis[0]) + (timings_laser_on[0] - timings_camera_images_bis[1])/1000
        _timings_laser_off = timings_camera_images[0] + (timings_camera_images[0] - timings_camera_images_bis[0]) + (timings_laser_off[0] - timings_camera_images_bis[1])/1000

        timings_dlp_on = []
        timings_dlp_off = []
        timings_laser_on = []
        timings_laser_off = []

        timings_dlp_on.append(_timings_dlp_on)
        timings_dlp_off.append(_timings_dlp_off)
        timings_laser_on.append(_timings_laser_on)
        timings_laser_off.append(_timings_laser_off)

        ### if different length between timings and images
        # cropped_rois_signal = []
        # for roi_signal in rois_signal:
        #     cropped_rois_signal.append(roi_signal[0:len(timings_camera_images)])
        # len(cropped_rois_signal[0])
        # rois_signal = cropped_rois_signal

        time_sorted_rois_signal = []
        x_axis_sorted_values = []
        for i in range(len(rois_signal)):
            data = np.vstack((timings_camera_images, rois_signal[i]))
            data.shape
            time_sorted_rois_signal.append(data[1][data[0,:].argsort()])
            x_axis_sorted_values = np.array(data[0][data[0,:].argsort()])
        x_axis = np.array([(x_axis_sorted_values[frame] - x_axis_sorted_values[0]) for frame in range(len(x_axis_sorted_values))])

        ## diagnostic plot: time between 2 images
        times_between_two_images = []
        for frame in range(len(x_axis)-1):
            times_between_two_images.append((x_axis[frame+1] - x_axis[frame]))
        times_between_two_images.append(times_between_two_images[-1])
        nb_images = np.arange(0,len(data[1]), 1)
        plt.plot(nb_images, np.array(times_between_two_images))

        rois_signal = time_sorted_rois_signal

    ## for baseline calculation:
    if timing:
        ## find laser_on index on x_axis
        takeClosest = lambda num,collection:min(collection,key=lambda x:abs(x-num))
        closest_index_laser_on_on_x = takeClosest(timings_laser_on[0]/10**9, x_axis)
        index_laser_on_for_baseline_calc = np.where(x_axis == closest_index_laser_on_on_x)
        ## baseline starting and ending
        baseline_starting_frame = index_laser_on_for_baseline_calc[0][0] + 5  ## need to be timed on the frames after laser activation I think
        baseline_frame_number = 20  ## also need to be adjusted to be up to the frame-1 of dlp activation (the more data, the better the baseline)
    else :
        baseline_starting_frame = 10  ## need to be timed on the frames after laser activation I think
        baseline_frame_number = 10  ## also need to be adjusted to be up to the frame-1 of dlp activation (the more data, the better the baseline)




    ### GRAPHS ###
    ## calculation of F(t)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    if timing == False:
        x_axis = np.arange(0,len(images), 1)
    for i in range(len(rois_signal)):
        plt.plot(x_axis, np.array(rois_signal[i]), color = colors[i], label = rois[i][0], alpha=0.7)
        if timing:
            for i in range(len(timings_dlp_on)):
                if draw_dlp_timings:
                    plt.axvspan(timings_dlp_on[i] - x_axis_sorted_values[0],timings_dlp_off[i] - x_axis_sorted_values[0], color='blue', alpha=0.05)
                if draw_laser_timings:
                    plt.axvspan(timings_laser_on[i] - x_axis_sorted_values[0],timings_laser_off[i] - x_axis_sorted_values[0], color='red', alpha=0.05)
    plt.legend()
    plt.title('Pixel value evolution with frames')
    plt.ylabel('Value')
    if args.time == None:
        plt.savefig(path_output+'pixel_time_serie_whole_data.svg')
        plt.savefig(path_output+'pixel_time_serie_whole_data.png')
    else:
        plt.savefig(f'{path_output}pixel_time_serie_whole_data_{args.time[0]}_{args.time[1]}.svg')
        plt.savefig(f'{path_output}pixel_time_serie_whole_data_{args.time[0]}_{args.time[1]}.png')
    plt.close()

    ## calculation of F(t) - background(t)
    for i in np.arange(1, len(rois_signal), 1):
        plt.plot(x_axis, np.array(rois_signal[0])-np.array(rois_signal[i]), color = colors[i], label = rois[i][0], alpha=0.7)
        if timing:
            for i in range(len(timings_dlp_on)):
                if draw_dlp_timings:
                    plt.axvspan(timings_dlp_on[i] - x_axis_sorted_values[0],timings_dlp_off[i] - x_axis_sorted_values[0], color='blue', alpha=0.05)
                if draw_laser_timings:
                    plt.axvspan(timings_laser_on[i] - x_axis_sorted_values[0],timings_laser_off[i] - x_axis_sorted_values[0], color='red', alpha=0.05)
    plt.title('Fluorescence with substracted backg fluorescence (per frame)')
    plt.ylabel('Value')
    plt.legend()
    if args.time == None:
        plt.savefig(f'{path_output}pixel_time_serie_with_backg_substraction_whole_data.svg')
        plt.savefig(f'{path_output}pixel_time_serie_with_backg_substraction_whole_data.png')
    else:
        plt.savefig(f'{path_output}pixel_time_serie_with_backg_substraction_{args.time[0]}_{args.time[1]}.svg')
        plt.savefig(f'{path_output}pixel_time_serie_with_backg_substraction_{args.time[0]}_{args.time[1]}.png')
    plt.close()

    ## calculation of percent delta F/F0
    times = []
    baseline_background = np.mean(np.array(rois_signal[0][baseline_starting_frame:baseline_starting_frame+baseline_frame_number])) ## temporal average
    dF_over_F0_background = ((np.array(rois_signal[0]) - baseline_background) / baseline_background)
    percent_dF_over_F0_background = dF_over_F0_background*100
    # plt.plot(x_axis, percent_dF_over_F0_background, color= 'b', label = rois[0][0], alpha=0.7)
    if timing:
        for i in range(len(timings_dlp_on)):
                if draw_dlp_timings:
                    plt.axvspan(timings_dlp_on[i] - x_axis_sorted_values[0],timings_dlp_off[i] - x_axis_sorted_values[0], color='blue', alpha=0.05)
                if draw_laser_timings:
                    plt.axvspan(timings_laser_on[i] - x_axis_sorted_values[0],timings_laser_off[i] - x_axis_sorted_values[0], color='red', alpha=0.05)
    for i in np.arange(1, len(rois_signal), 1):
        _times = []
        baseline_soma = np.mean(np.array(rois_signal[i][baseline_starting_frame:baseline_starting_frame + baseline_frame_number]))
        dF_over_F0_soma = ((np.array(rois_signal[i]) - baseline_soma) / baseline_soma) - dF_over_F0_background
        percent_dF_over_F0_soma = dF_over_F0_soma * 100
        # plt.ylim([-5,35])
        plt.plot(x_axis, percent_dF_over_F0_soma, color = colors[i], label = rois[i][0], alpha=0.7)
        data_export.append(percent_dF_over_F0_soma.tolist())
    if timing:
        for i in range(len(timings_dlp_on)):
            if draw_dlp_timings:
                dlp_on_value_on_x = timings_dlp_on[i] - x_axis_sorted_values[0]
                dlp_off_value_on_x = timings_dlp_off[i] - x_axis_sorted_values[0]
                plt.axvspan(dlp_on_value_on_x, dlp_off_value_on_x, color='blue', alpha=0.05)
            if draw_laser_timings:
                laser_on_value_on_x = timings_laser_on[i] - x_axis_sorted_values[0]
                laser_off_value_on_x = timings_laser_off[i] - x_axis_sorted_values[0]
                plt.axvspan(laser_on_value_on_x, laser_off_value_on_x, color='red', alpha=0.05)
            _times = dlp_on_value_on_x, dlp_off_value_on_x , laser_on_value_on_x, laser_off_value_on_x
            times.append(_times)
    plt.ylabel(r'$\%$ $\Delta$ F/F0')
    plt.legend()
    if args.time == None:
        plt.savefig(f'{path_output}delta_F_over_F0__whole_data.svg')
        plt.savefig(f'{path_output}delta_F_over_F0__whole_data.png')
    else:
        plt.savefig(f'{path_output}delta_F_over_F0_{args.time[0]}_{args.time[1]}.svg')
        plt.savefig(f'{path_output}delta_F_over_F0_{args.time[0]}_{args.time[1]}.png')
    # saving data
    data_export.append(x_axis.tolist())
    data_export.append(times)
    data_export = np.array(data_export)
    ## data has format [[ROI1], [ROI2] ..., [ROIn], [X_axis], [[timings_dlp_on(ROI1), timings_dlp_off(ROI1), timings_laser_on(ROI1), timings_laser_off(ROI1)],[...]]
    np.save(f'{path_output}dF_over_F0_backcorrect.npy', data_export)
    plt.close()

    ## ephys-type graph for percent delta F/F0
    # from matplotlib_scalebar.scalebar import ScaleBar
    fig = plt.figure(frameon=False)
    fig, axs = plt.subplots(len(rois_signal), 1)
    fig.subplots_adjust(hspace=0)
    baseline_roi_background = np.mean(np.array(rois_signal[0][baseline_starting_frame:baseline_starting_frame + baseline_frame_number]))
    # axs[0].set_ylim([-5,150])
    axs[0].plot(x_axis, percent_dF_over_F0_background, color = 'b', label = rois[0][0], alpha=0.7)
    axs[0].set_axis_off()
    if timing:
        for j in range(len(timings_dlp_on)):
            if draw_dlp_timings:
                axs[0].axvspan(timings_dlp_on[j] - x_axis_sorted_values[0],timings_dlp_off[j] - x_axis_sorted_values[0], color='blue', alpha=0.05)
            if draw_laser_timings:
                axs[0].axvspan(timings_laser_on[j] - x_axis_sorted_values[0],timings_laser_off[j] - x_axis_sorted_values[0], color='red', alpha=0.05)
    # equivalent_10ms = 0.1
    # scalebar = ScaleBar(0.1, 'ms', frameon=False, location='lower left', length_fraction = equivalent_10ms)
    # plt.gca().add_artist(scalebar)
    # axs[0].annotate('', xy=(0, 0),  xycoords='axes fraction', xytext=(0, .2), textcoords='axes fraction',
    #                     ha='center', va='center', arrowprops=dict(arrowstyle="-", color='black'))
    # axs[0].annotate('', xy=(0, 0),  xycoords='axes fraction', xytext=(longueur_5ms, 0), textcoords='axes fraction',
    #                     ha='center', va='center', arrowprops=dict(arrowstyle="-", color='black'))
    for i in np.arange(1, len(rois_signal), 1):
        dF_over_F0_roi = ((np.array(rois_signal[i]) - baseline_roi_background) / baseline_roi_background) - dF_over_F0_background
        percent_dF_over_F0_roi = dF_over_F0_roi * 100
        axs[i].set_ylim([-5,150])
        axs[i].plot(x_axis, percent_dF_over_F0_roi, color = colors[i], label = rois[i][0], alpha=0.7)
        axs[i].set_axis_off()
        if timing:
            for j in range(len(timings_dlp_on)):
                if draw_dlp_timings:
                    axs[i].axvspan(timings_dlp_on[j] - x_axis_sorted_values[0],timings_dlp_off[j] - x_axis_sorted_values[0], color='blue', alpha=0.05)
                if draw_laser_timings:
                    axs[i].axvspan(timings_laser_on[j] - x_axis_sorted_values[0],timings_laser_off[j] - x_axis_sorted_values[0], color='red', alpha=0.05)
    if args.time == None:
        plt.savefig(f'{path_output}delta_F_over_F0_ephys_style_whole_data.svg')
        plt.savefig(f'{path_output}delta_F_over_F0_ephys_style_whole_data.png')
    else:
        plt.savefig(f'{path_output}delta_F_over_F0_ephys_style_{args.time[0]}_{args.time[1]}.svg')
        plt.savefig(f'{path_output}delta_F_over_F0_ephys_style_{args.time[0]}_{args.time[1]}.png')
    plt.close()
