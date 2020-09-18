
import os
import json

def read_image_size(json_file_path):
    """
    Open json file, read the image size and bin width to determine actuale
    images size in x and y dimensions
    input: json file path
    output: tuple (image size x, image size y)
    """
    with open(json_file_path) as f:
        data = json.load(f)
    image_bin = data['binning'][0]
    image_size_x, image_size_y = data['fov'][0][1], data['fov'][0][3]
    return (int(image_size_x/image_bin), int(image_size_y/image_bin))

def images_list(folder, extension="png", data_type='raw'):
    """
    If the npy file names have no leading zeros then the  list is not ordered
    correctly. This serve to create an ordered list from the length of the
    number of files in the folder.
    input: folder path
    output: ordered npy file name list
    """
    files_len = len(next(os.walk(folder))[2])
    image_list = []
    for i in range(files_len):
        if data_type == 'raw':
            image_name = f'image{str(i)}.{extension}'
        elif data_type == 'denoised':
            image_name = f'image_denoised{str(i)}.{extension}'
        image_list.append(image_name)
    return image_list

def read_fps(json_file_path):
    """
    Open json file, read the fps the data was recorded at.
    input: json file path
    output: int (fps)
    """
    with open(json_file_path) as f:
        data = json.load(f)
    fps = data['fps'][0]
    return fps

def load_timing_data(json_file_path):
    """
    This function allows to load the json file where the timings of the laser, dlp, camera
    and ephys are stored.
    Input: path of the json file
    Output: timings, each in their own variable
    """
    with open(json_file_path) as file:
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

def import_roi_mask(file_path):
    pass
