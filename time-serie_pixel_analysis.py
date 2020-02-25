import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import os
import pickle
import json
from roipoly.roipoly import RoiPoly, MultiRoi

# experiments = [10,11,12,13,14]
experiments = range(1,31)
# experiments = ['merged_37_39']
# experiment = 15
denoised_images = False
draw_laser_timings = True
draw_dlp_timings = False
timing = True
time_init = 0
time_end = 1400
baseline_starting_frame = 10
baseline_frame_number = 10

def images_list(folder):
    files_len = len(os.listdir(folder))
    image_list = []
    # for i in range(time_init,time_end):
    for i in range(files_len):
        image_name = 'image'+ str(i) + '.png'
        image_list.append(image_name)
    return image_list

def load_json_timing_data(path_output, experiment):
    with open(f'{path_output}experiment_{experiment}_timings.json') as file:
        timings_data = dict(json.load(file))
    timings_dlp_on = timings_data['dlp']['on']
    timings_dlp_off= timings_data['dlp']['off']
    timings_laser_on = timings_data['laser']['on']
    timings_laser_off = timings_data['laser']['off']
    timings_camera_images = []
    for images_timings in timings_data['camera']:
        for image_timing in images_timings:
            timings_camera_images.append(image_timing)
    return timings_dlp_on, timings_dlp_off, timings_camera_images, timings_laser_on, timings_laser_off


print('processing experiments')
for experiment in tqdm(experiments):
    # experiment = 'merged_10_60'
    ### PATHS ###
    print(f"processing experiment - {experiment}")
    # experiment = 15
    path_input = "/media/jeremy/Data/local/Data_manip/2020_02_18/experiment_{}/images/".format(experiment)
    path_input_denoised = "/media/jeremy/Data/local/Data_manip/2020_02_18/experiment_{}/denoised_images/".format(experiment)
    path_output = "/media/jeremy/Data/local/Data_manip/2020_02_18/experiment_{}/".format(experiment)
    roi_file_path = path_output + 'roi_masks.txt'

    if denoised_images:
        images = images_list(path_input_denoised)
    else:
        images = images_list(path_input)



    ### ROIS ####
    if os.path.isfile(roi_file_path):
        print('ROI file exists')
        with open(roi_file_path, "rb") as file:
            rois = pickle.load(file)

    else:
        print('ROI file doesnt exists')

        if denoised_images:
            w,h = cv2.imread(path_input_denoised+images[0],cv2.IMREAD_GRAYSCALE).shape
        else:
            w,h = cv2.imread(path_input+images[0],cv2.IMREAD_GRAYSCALE).shape

        ### ref image for defining rois, need to think about what it can be. The best would be a DIC image.
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
        plt.savefig(path_output+'rois.svg')
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
        timings_dlp_on, timings_dlp_off, \
        timings_camera_images, \
        timings_laser_on, timings_laser_off = load_json_timing_data(path_output, experiment)
        if len(timings_dlp_on)>1:
            print('more than 1 timing from DLP ON')
            timings_dlp_on = [timings_dlp_on[0]] ## temporary fix because dlp on is registered 2 times instead of 1.

        ## for diagnostic
        # plt.plot(np.array(timings_camera_images), range(0,len(timings_camera_images)))

        timings_camera_images_new = timings_camera_images[time_init : time_end]
        timings_camera_images = timings_camera_images_new
        print(f'number of camera images: {len(timings_camera_images)}')
        print(f'number of points in the each roi signal: {len(rois_signal[0])}')
        assert len(images) == len(timings_camera_images), 'not the same number of images and images timing metadata'  ## will not work when working on subset of the data

        # to put both dlp, laser and camera timings in the same format: putting camera images in ns
        timings_camera_images = [timings_camera_images[i]*10**9 for i in range(len(timings_camera_images))]



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
    plt.savefig(path_output+'pixel_time_serie.svg')
    plt.savefig(path_output+'pixel_time_serie.png')
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
    plt.savefig(path_output+'pixel_time_serie_with_backg_substraction.svg')
    plt.savefig(path_output+'pixel_time_serie_with_backg_substraction.png')
    plt.close()

    ## calculation of percent delta F/F0
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
        baseline_soma = np.mean(np.array(rois_signal[i][baseline_starting_frame:baseline_starting_frame + baseline_frame_number]))
        dF_over_F0_soma = ((np.array(rois_signal[i]) - baseline_soma) / baseline_soma) - dF_over_F0_background
        percent_dF_over_F0_soma = dF_over_F0_soma * 100
        plt.plot(x_axis, percent_dF_over_F0_soma, color = colors[i], label = rois[i][0], alpha=0.7)
        if timing:
            for i in range(len(timings_dlp_on)):
                if draw_dlp_timings:
                    plt.axvspan(timings_dlp_on[i] - x_axis_sorted_values[0],timings_dlp_off[i] - x_axis_sorted_values[0], color='blue', alpha=0.05)
                if draw_laser_timings:
                    plt.axvspan(timings_laser_on[i] - x_axis_sorted_values[0],timings_laser_off[i] - x_axis_sorted_values[0], color='red', alpha=0.05)
    plt.ylabel(r'$\%$ $\Delta$ F/F0')
    plt.legend()
    plt.savefig(path_output+'delta F over F0.svg')
    plt.savefig(path_output+'delta F over F0.png')
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
    plt.savefig(path_output+'delta F over F0 ephys style.svg',bbox_inches='tight', pad_inches=0)
    plt.savefig(path_output+'delta F over F0 ephys style.png',bbox_inches='tight', pad_inches=0)
    plt.close()
