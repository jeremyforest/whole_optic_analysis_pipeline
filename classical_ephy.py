
import numpy as np
import matplotlib.pyplot as plt


def import_ephy_data(main_folder_path, experiment):
    data = np.load(f'{main_folder_path}experiment_{experiment}/voltage.npy')
    data.shape

    new_data = np.zeros((8,5000))
    new_data[0].shape

    for channel in range(data.shape[1]):
        for recording in range(data.shape[0]):
            if recording == 0:
                df = data[recording][channel]
            else:
                df = np.hstack((data[recording][channel], df))

        new_data[channel] = df
    return new_data

def plot_ephy_data(data):
    x = list(range(data.shape[1]))

    fig, axs = plt.subplots(2,4, sharex=True, sharey=True)
    axs = axs.ravel()
    for channel in range(data.shape[0]):
        # plt.plot(x, new_data[channel])
        axs[channel].plot(x, data[channel])
        axs[channel].set_title(f'channel {channel+1}')


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
    timings_ephy_on = timings_data['ephy']['on']
    timings_ephy_off = timings_data['ephy']['off']
    timings_ephy_stim_on = timings_data['ephy_stim']['on']
    timings_ephy_stim_off = timings_data['ephy_stim']['off']
    timings_camera_images = []  ## timings of the images as per the dcam api
    for images_timings in timings_data['camera']:
        for image_timing in images_timings:
            timings_camera_images.append(image_timing)

    timings_camera_images_bis = timings_data['camera_bis'] ## timing of the first frame as per manual clock
    print(f'timing difference between first image metadata and manual log is \
            {(timings_camera_images[0] - timings_camera_images_bis[0]) * 1000}')   ## ms difference

    return timings_dlp_on, timings_dlp_off, timings_camera_images, timings_laser_on, timings_laser_off, timings_camera_images_bis



if __name__ == "__main__":

    main_folder_path = '/home/jeremy/Documents/Postdoc/Projects/Memory/Computational_Principles_of_Memory/optopatch/data/2020_05_25/'
    experiment = 4

    data = import_ephy_data(main_folder_path=main_folder_path, experiment=experiment)
    plot_ephy_data(data = data)
