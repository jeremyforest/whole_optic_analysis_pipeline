import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import os
import pickle

from roipoly.roipoly import RoiPoly, MultiRoi

experiments = [1,3,4,5]
# experiments = [10,11,12,13,14,15,16,18,19,20,21]
# experiment = 15


def images_list(folder):
    files_len = len(os.listdir(folder))
    image_list = []
    # for i in range(300, 350):
    for i in range(files_len):
        image_name = 'image'+ str(i) + '.png'
        image_list.append(image_name)
    return image_list


print('processing experiments')
for experiment in tqdm(experiments):
    print(f"processing experiment {experiment}")
    # experiment = 1
    path_input = "/media/jeremy/Data/Data_Jeremy/2019_10_29/experiment_{}/images/".format(experiment)
    path_input_denoised = "/media/jeremy/Data/Data_Jeremy/2019_10_29/experiment_{}/denoised_images/".format(experiment)
    path_output = "/media/jeremy/Data/Data_Jeremy/2019_10_29/experiment_{}/".format(experiment)
    roi_file_path = path_output+'roi_masks.txt'

    # images = images_list(path_input)
    images = images_list(path_input_denoised)

    if os.path.isfile(roi_file_path):
        print('ROI file exists')
        with open(roi_file_path, "rb") as file:
            rois = pickle.load(file)

    else:
        print('ROI file doesnt exists')

        # w,h = cv2.imread(path_input+images[0],cv2.IMREAD_GRAYSCALE).shape
        w,h = cv2.imread(path_input_denoised+images[0],cv2.IMREAD_GRAYSCALE).shape

        ### ref image for defining rois, need to think about what it can be. The best would be a DIC image.
        image = cv2.imread(path_output + 'neurons.png'.format(experiment), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (w,h))
        # Show the image
        fig = plt.figure()
        plt.imshow(image, interpolation='none', cmap='gray')
        plt.title("Click on the button to add a new ROI")
        # Draw multiple RO
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
            img = cv2.imread(path_input_denoised+image,cv2.IMREAD_GRAYSCALE)
            mask = roi[1]
            roi_average = np.mean(img[mask])
            tmp_time_serie_roi.append(roi_average)
        rois_signal.append(tmp_time_serie_roi)

    print ('generating data plots')

    ## calculation of F(t)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    x = np.arange(0,len(images), 1)
    for i in range(len(rois_signal)):
        plt.plot(x, np.array(rois_signal[i]), color = colors[i], label = rois[i][0], alpha=0.7)
        # for on in range(len(laser_activation)):
        #     plt.axvspan(laser_activation[on][0],laser_activation[on][1], color='red', alpha=0.2)
    plt.legend()
    plt.title('Pixel value evolution with frames')
    plt.ylabel('Value')
    plt.savefig(path_output+'pixel_time_serie.svg')
    plt.savefig(path_output+'pixel_time_serie.png')
    plt.close()

    ## calculation of F(t) - background(t)
    for i in np.arange(1, len(rois_signal), 1):
        plt.plot(x, np.array(rois_signal[0])-np.array(rois_signal[i]), color = colors[i], label = rois[i][0], alpha=0.7)
    plt.title('Fluorescence with substracted backg fluorescence (per frame)')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(path_output+'pixel_time_serie_with_backg_substraction.svg')
    plt.savefig(path_output+'pixel_time_serie_with_backg_substraction.png')
    plt.close()

    ## calculation of percent delta F/F0
    baseline_background = np.mean(np.array(rois_signal[0][0:20])) ## temporal average
    dF_over_F0_background = ((np.array(rois_signal[0]) - baseline_background) / baseline_background)
    percent_dF_over_F0_background = dF_over_F0_background*100
    plt.plot(x, percent_dF_over_F0_background, color= 'b', label = rois[0][0], alpha=0.7)
    for i in np.arange(1, len(rois_signal), 1):
        baseline_soma = np.mean(np.array(rois_signal[i][0:20]))
        dF_over_F0_soma = ((np.array(rois_signal[i]) - baseline_soma) / baseline_soma) - dF_over_F0_background
        percent_dF_over_F0_soma = dF_over_F0_soma * 100
        plt.plot(x, percent_dF_over_F0_soma, color = colors[i], label = rois[i][0], alpha=0.7)
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
    baseline_roi_background = np.mean(np.array(rois_signal[0][0:20]))
    axs[0].set_ylim([-10,200])
    axs[0].plot(x, percent_dF_over_F0_background, color = 'b', label = rois[0][0], alpha=0.7)
    axs[0].set_axis_off()
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
        axs[i].set_ylim([-10,200])
        axs[i].plot(x, percent_dF_over_F0_roi, color = colors[i], label = rois[i][0], alpha=0.7)
        axs[i].set_axis_off()
    plt.savefig(path_output+'delta F over F0 ephys style.svg',bbox_inches='tight', pad_inches=0)
    plt.savefig(path_output+'delta F over F0 ephys style.png',bbox_inches='tight', pad_inches=0)
    plt.close()
