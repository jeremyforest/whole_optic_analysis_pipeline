import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def png_conversion(input_data_folder, output_data_folder):
    # input_data_folder = path_input_raw_data
    # output_data_folder = '/media/jeremy/Data/local/Data_manip/2020_01_16/experiment_5/images'
    files = os.listdir(input_data_folder)
    for file in tqdm(files):
        # file=files[0]
        if file.endswith('.npy'):
            try:
                img_array= np.load(input_data_folder+'/'+file)
                ## need to query the json file to know the size of the array here
                ## need to take binning into consideration
                size_img = (int(np.sqrt(img_array.size)), int(np.sqrt(img_array.size)))  ## that only works for square images, will need to find something else for rectangular images
                # size_img = (128, 1024)
                img_array = img_array.reshape(size_img[0], size_img[1])
                img_name = output_data_folder+file.replace('.npy', '.png')
                plt.imsave(img_name, img_array, cmap='gray')
            except:
                print('cannot convert ' + str(file))



if __name__ == "__main__":

    experiment = 'experiment_132'

    path_input = '/mnt/home_nas/jeremy/Recherches/Postdoc/Projects/Memory/Computational_Principles_of_Memory/optopatch/data/2020_03_02/{}/'.format(experiment)
    path_input_raw_data = '/mnt/home_nas/jeremy/Recherches/Postdoc/Projects/Memory/Computational_Principles_of_Memory/optopatch/data/2020_03_02/{}/raw_data'.format(experiment)
    path_output = '/mnt/home_nas/jeremy/Recherches/Postdoc/Projects/Memory/Computational_Principles_of_Memory/optopatch/data/2020_03_02/{}/images/'.format(experiment)

    try:
        os.mkdir(path_input + 'images/'.format(experiment))
    except FileExistsError:
        pass

    png_conversion(path_input_raw_data, path_output)
