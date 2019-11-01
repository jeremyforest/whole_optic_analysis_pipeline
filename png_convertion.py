import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def png_conversion(input_data_folder, output_data_folder):
    # input_data_folder = '/media/jeremy/Data/Data_Jeremy/20_08_2019/experiment_1/raw_data'
    # output_data_folder = '/media/jeremy/Data/Data_Jeremy/20_08_2019/experiment_1/images'
    files = os.listdir(input_data_folder)
    for file in tqdm(files):
        # file=files[3]
        if file.endswith('.npy'):
            try:
                img_array= np.load(input_data_folder+'/'+file)
                size_img = int(np.sqrt(img_array.size))  ## that only works for square images, will need to find something else for rectangular images
                img_array = img_array.reshape(size_img, size_img)
                img_name = output_data_folder+'/'+file.replace('.npy', '.png')
                plt.imsave(img_name, img_array, cmap='gray')
            except:
                print('cannot convert ' + str(file))



if __name__ == "__main__":

    ###############################################################
    ### NOT SURE IF THIS IS STILL WORKING - WOULD NEED TO CHECK ###
    ###############################################################

    experiment = 'experiment_1'

    path_input = '/media/jeremy/Data/Data_Jeremy/2019_10_28/{}/'.format(experiment)
    path_output_video = '/media/jeremy/Data/Data_Jeremy/2019_10_28/{}/{}.avi'. format(experiment, experiment)

    try:
        os.mkdir(path_input + 'images/'.format(experiment))
    except FileExistsError:
        pass

    png_conversion(path_input, path_output_video)
