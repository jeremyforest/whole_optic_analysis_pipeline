from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler
from skimage import data, io, color
import os
import numpy as np
from tqdm import tqdm
from OPTIMAS.utils.files_handling import images_list, read_fps, \
                                         load_timing_data, read_image_size


def ICA(input_data_folder, experiment):

    path_input_images = f"{input_data_folder}/{experiment}/images/"
    try:
        os.mkdir(f'{input_data_folder}/{experiment}/ICAimages/')
    except FileExistsError:
        pass
    path_output = f"{input_data_folder}/{experiment}/ICAimages"

    images = images_list(path_input_images)

    for img, image_nb in tqdm(zip(images, range(len(images)))):
        # image = images[0]
        image = io.imread(f"{path_input_images}/{img}", as_gray=True)

        ica = FastICA(n_components = 20)
                        # whiten=True,
                        # max_iter = 2000,
                        # tol = 0.01)
        image_ica = ica.fit_transform(image)
        image_restored = ica.inverse_transform(image_ica)
        # image_restored = image_restored.astype(np.uint8)
        # show image to screen
        # io.imshow(image_ica)
        io.imsave(f'{path_output}/image{image_nb}.png', image_ica)

def PrCA(input_data_folder, experiment):

    path_input_images = f"{input_data_folder}/{experiment}/images/"
    try:
        os.mkdir(f'{input_data_folder}/{experiment}/PCAimages/')
    except FileExistsError:
        pass
    path_output = f"{input_data_folder}/{experiment}/PCAimages"

    images = images_list(path_input_images)

    for img, image_nb in tqdm(zip(images, range(len(images)))):
        # image = images[0]
        image = io.imread(f"{path_input_images}/{img}", as_gray=True)

        pca = PCA(5)
        image_pca = pca.fit_transform(image)

        image_restored = pca.inverse_transform(image_pca)
        # image_restored = image_restored.astype(np.uint8)
        # show image to screen
        # io.imshow(image_ica)
        io.imsave(f'{path_output}/image{image_nb}.png', image_pca)


if __name__ == "__main__":
    experiment = 'experiment_50'
    input_data_folder = f'/home/jeremy/Desktop/2020_11_20'
    ICA(input_data_folder, experiment)
    PrCA(input_data_folder, experiment)
