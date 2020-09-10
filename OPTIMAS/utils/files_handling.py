
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
