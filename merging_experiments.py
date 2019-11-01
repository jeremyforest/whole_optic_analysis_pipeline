import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

merging = True
# png_conversion = False
# animate = False

if merging:

    # experiments = [2,4,6,8,10,12,14,16,18,20]
    # experiments = [1,3,5,7,9,11,13,15,17,19]
    experiments = np.arange(21,31,1)
    file_list_length = []

    for experiment in experiments:
        path_input = "/media/jeremy/Data/Data_Jeremy/20_08_2019/experiment_{}/raw_data/".format(experiment)
        # path_output = "/media/jeremy/Data/Data_Jeremy/16_08_2019/merged_data/experiment_merged_1_10/"
        path_output = "/media/jeremy/Data/Data_Jeremy/20_08_2019/experiment_merged_21_30/raw_data/"

        file_list = os.listdir(path_input)
        file_list_length.append(len(file_list))

    min_file_list_length = np.min(file_list_length)
    for file_number in range(0,min_file_list_length-1,1):
        for experiment in experiments:
            print('working on experiment ' + str(experiment) + '-' + str(file_number))
            path_input = "/media/jeremy/Data/Data_Jeremy/20_08_2019/experiment_{}/raw_data/".format(experiment)
            if experiment == experiments[0]:
                img_size = int(np.sqrt(np.load(path_input + 'image{}.npy'.format(file_number)).size))
                old_img_array = np.zeros((img_size*img_size))
            new_img_array = np.load(path_input + 'image{}.npy'.format(file_number))
            tmp_img_array_averaged = old_img_array + new_img_array/8
            old_img_array = tmp_img_array_averaged
            if experiment == experiments[-1]:
                averaged_array = old_img_array
                np.save(path_output+ 'image{}'.format(file_number), averaged_array)
                print ('image saved')

"""
if png_conversion:
    path_output = "/media/jeremy/Data/Data_Jeremy/16_08_2019/merged_data/experiment_merged_2_21/"
    files = os.listdir(path_output)
    path_output_png = "/media/jeremy/Data/Data_Jeremy/16_08_2019/merged_data/experiment_merged_2_21_images/"
    for file in tqdm(files):
        # file=files[0]
        if file.endswith('.npy'):
            try:
                img_array= np.load(path_output + file)
                size_img = int(np.sqrt(img_array.size))
                img_array = img_array.reshape(size_img, size_img)
                img_name = path_output_png + file + ".png"
                plt.imsave(img_name, img_array, cmap='gray')
            except:
                print('cannot convert ' + str(file))

if animate:
    path_output_video = "/media/jeremy/Data/Data_Jeremy/20_08_2019/merged_data/"
    len_images = len(os.listdir(path_output_png))
    filenames = []
    img_array = []
    for i in range(1,len_images,1):
        filename = 'image{}.npy.png'.format(i)
        filenames.append(filename)

    for filename in filenames:
        img = cv2.imread(path_output_png+filename)
        h, v, z = img.shape
        size = (h,v)
        img_array.append(img)

    out = cv2.VideoWriter('{}merged_2_21_images.avi'.format(path_output_video),cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    # out = cv2.VideoWriter('{}experiment_{}_{}_denoised.avi'.format(path_output, experiment, str(j)),cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

    print("converting into video file")
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
"""
