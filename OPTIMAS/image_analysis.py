import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.ndimage import median_filter

from skimage.restoration import denoise_wavelet


path_input = '/media/jeremy/Data/Data_Jeremy/19_06_2019/experiment_4/'
# path_output = '/media/jeremy/Data/Data_Jeremy/19_06_2019/experiment_4_analysis/background_substraction/'

image_number_start = 4665
image_number_stop = 4770


def images_list():
    image_list = []
    image_range = abs(image_number_start - image_number_stop)
    for i in range(image_range):
        image_name = 'image'+ str(image_number_start+i) + '.npy'
        image_list.append(image_name)
    return image_list

def background_substraction(image_list):
    ## calculate the average pixel intensity over the number of arrays and substract it from each array
    print('background_image_generation')
    temp_dataset = []
    for number in tqdm(range(len(image_list))):
        image = np.load(path_input + image_list[number])
        temp_dataset.append(image)
    mean_array = np.mean([temp_dataset], axis=1)
    return mean_array

def image_diff(img1, img2):
    ## calculate the element-wise difference between two pictures
    diff_images = img1 - img2
    return diff_images

def image_filter(image, size):
        return median_filter(image, size)

def image_save(path, image_name, img_arr):
    np.save(file = str(path) + image_name, arr=img_arr)


def pipeline(bckg = True, diff = True, thresh = True):
    image_list = images_list()  ## list the images to process

    if background_substraction:
        background_image = background_substraction(image_list) ## generate background image
        print('background substraction')
        for number in tqdm(range(len(image_list))):
            img = np.load(path_input + image_list[number])
            removed_background_image = img - background_image
            name = str(image_list[number])
            image_save('/media/jeremy/Data/Data_Jeremy/19_06_2019/experiment_4_analysis/background_substraction/', name, removed_background_image)

    if diff:
        print('image difference')
        for number in tqdm(range(len(image_list)-1)):
            img1 = np.load('/media/jeremy/Data/Data_Jeremy/19_06_2019/experiment_4_analysis/background_substraction/'+image_list[number])
            img2 = np.load('/media/jeremy/Data/Data_Jeremy/19_06_2019/experiment_4_analysis/background_substraction/'+image_list[number+1])
            img_difference = image_diff(img1, img2)
            name = str(image_list[number])
            image_save('/media/jeremy/Data/Data_Jeremy/19_06_2019/experiment_4_analysis/image_difference/', name, img_difference)

    if thresh:
        pass



if __name__ == "__main__":
    pipeline(bckg=True, diff=True)

    #         # filtered_img = median_filter(img, size=20)
    #
    #
    #
    # for number in tqdm(range(len(image_list)-1)):
    #     img1 = np.load(path_input+image_list[number])
    #     img1 = image_filter(img1, size=5)
    #     img1 = img1 - background_image
    #     img2 = np.load(path_input+image_list[number+1])
    #     img2 = image_filter(img2, size=5)
    #     img2 = img2 - background_image
    #     new_image = image_diff(img1, img2)
    #     denoised_image = denoise_wavelet(new_image)
    #     name = str(image_list[number])
    #     image_save(path_output, name, denoised_image)
