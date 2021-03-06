#!/usr/bin/env python

## taken from the Test_3D_Decomp notebook

import numpy as np
import scipy
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import sys
from skimage import io
import scipy.interpolate as interp

from trefide.denoiser import batch_decompose, batch_recompose, overlapping_batch_decompose, overlapping_batch_recompose
from trefide.plot import pixelwise_ranks


def eval_spatial_stat(u):
    tmp1 = np.abs(u[1:,:,:] - u[:-1,:,:])
    tmp2 = np.abs(u[:,1:,:] - u[:,:-1,:])
    tmp3 = np.abs(u[:,:,1:] - u[:,:,:-1])
    return (((np.sum(tmp1) + np.sum(tmp2) + np.sum(tmp3)) * np.prod(u.shape)) /
            ((np.prod(tmp1.shape) + np.prod(tmp2.shape) + np.prod(tmp3.shape)) * np.sum(np.abs(u))))

def eval_temporal_stat(v):
    return np.sum(np.abs(v[:-2] + v[2:] - 2*v[1:-1])) / np.sum(np.abs(v))


def thresholds(fov_height, fov_width, nb_channels, num_frames, Y, block_height, block_width,
                                  consec_failures, max_iters_main, max_iters_init, tol,
                                  d_sub, t_sub, enable_temporal_denoiser, enable_spatial_denoiser,
                                  path_output_data, plot=True):
    print('estimating spatial and temporal stats')
    spatial_stats = []
    temporal_stats = []
    # num_frames = min(6000, num_frames)
    max_blocks = 40
    spatial_thresh_sim = 1e5
    temporal_thresh_sim = 1e5
    max_comp_sim = 3
    num_samples = 240
    conf = .95
    num_repeats = int(num_samples / (max_blocks * consec_failures))

    for rep in tqdm(range(num_repeats)):
        # Generate Noise Movie Of NumSimul Blocks
        Y_sim = np.reshape(np.random.randn(Y.size),
                           (fov_height, fov_width, nb_channels, num_frames))
        # Y_sim.shape
        # Run Denoiser W/ Max Comp 3 and absurdly large thresholds
        out = batch_decompose(fov_height, fov_width, nb_channels, num_frames, Y_sim, block_height, block_width,
                                    spatial_thresh_sim, temporal_thresh_sim,
                                    max_comp_sim, consec_failures, max_iters_main, max_iters_init, tol,
                                    d_sub, t_sub, enable_temporal_denoiser, enable_spatial_denoiser)
        # Collect Test Statistics On All Samples
        for bdx in range(max_blocks):
            for cdx in range(consec_failures):
                spatial_stats.append(eval_spatial_stat(out[0][bdx,cdx]))
                temporal_stats.append(eval_temporal_stat(out[1][bdx,cdx]))

    spatial_stats = np.array(spatial_stats)
    temporal_stats = np.array(temporal_stats)

    # Compute Thresholds
    spatial_thresh =  np.percentile(spatial_stats, conf)
    temporal_thresh = np.percentile(temporal_stats, conf)

    if plot:
        fig, ax = plt.subplots(2,2,figsize=(8,8))
        ax[0,0].scatter(spatial_stats, temporal_stats, marker='x', c='r', alpha = .2)
        ax[0,0].axvline(spatial_thresh)
        ax[0,0].axhline(temporal_thresh)
        ax[0,1].hist(temporal_stats, bins=20, color='r')
        ax[0,1].axvline(temporal_thresh)
        ax[0,1].set_title("Temporal Threshold: {}".format(temporal_thresh))
        ax[1,0].hist(spatial_stats, bins=20, color='r')
        ax[1,0].axvline(spatial_thresh)
        ax[1,0].set_title("Spatial Threshold: {}".format(spatial_thresh))
        plt.savefig(path_output_data + '/thresholds.png')

    return spatial_thresh, temporal_thresh

def play_3d(movie,
            gain=3,
            fr=60,
            offset=0,
            magnification=1,
            repeat=False,
            save = False,
            path = None):

    """ Render Video With OpenCV3 Library's Imshow"""
    fov_height, fov_width, nb_channels, num_frames = movie.shape
    maxmov = np.max(movie)
    looping=True
    terminated=False
    while looping:
        for t in range(num_frames):
            if magnification != 1:
                frame = cv2.resize(np.reshape(movie[:,:,:,t], (fov_height, -1), order='F'),
                                   None,
                                   fx=magnification,
                                   fy=magnification,
                                   interpolation=cv2.INTER_LINEAR)
            else:
                frame = np.reshape(movie[:,:,:,t], (fov_height, -1), order='F')
            if save:
                plt.imsave(str(path) + '/image{}.png'.format(t), frame, cmap='gray')
            # cv2.imshow('frame', (frame - offset) / maxmov*gain)
            if cv2.waitKey(int(1. / fr * 1000)) & 0xFF == ord('q'):
                looping = False
                terminated = True
                break
        if terminated:
            break
        looping=repeat
    cv2.waitKey(100)
    cv2.destroyAllWindows()
    for i in range(10):
        cv2.waitKey(100)


def get_spline_trend(data,
                     nb_knots,
                     order=3,
                     axis=-1,
                     q=.05):
    """
    Fits a robust adaptive b-spline to an input dataset in order to remove slow
    trend and features due to application of step and ramp stimuli.
    Modified from trefide preprocess.py file.

    Parameter:
        data: input the data with format ??
        knots: number of knots
        order: b-spline degree
        axis: the axis on which the spline is applied

    Return:
        trend:
    """
    # data = Y
    # nb_knots = 10
    # order = 3
    data.shape
    data = data.reshape(128*128,-1)
    data.shape
    x = np.arange(data.shape[-1])
    t_bis = np.linspace(x[25], x[-25], nb_knots)
    knots = np.r_[(x[0],)*(order+1),
                t_bis,
                (x[-1],)*(order+1)]

    # y = data.sum(axis=0)
    # x.shape, y.shape, knots.shape, order

    def spline_fit(y):
        bspl = interp.make_lsq_spline(x=x, y=y, t=knots, k=order)
        return bspl(x)

    def robust_spline_fit(y):
        bspl = interp.make_lsq_spline(x=x, y=y, t=knots, k=order)
        resid = np.abs(bspl(x) - y)
        q = 0.05
        keep_idx = resid <= np.percentile(resid, (1 - q) * 100)
        bspl = interp.make_lsq_spline(
            x=x[keep_idx], y=y[keep_idx], t=knots, k=order)

        return bspl(x)

    # trend = np.apply_along_axis(robust_spline_fit, -1, data)
    trend = np.apply_along_axis(robust_spline_fit, axis, data)

    pixel = np.random.choice(range(data.shape[0]), 100)
    # for i in pixel:
    #     print(i)
    #     _y = data[i]
    #     _trend = trend[i]
    #     _y.shape, _trend.shape
    #     plt.scatter(x,_y)
    #     plt.plot(x,_trend)
    #     plt.savefig(f'pixel.png')
    # plt.close()

    return trend

def remove_spline_trend(data,
                        trend):
    """
    Removes the fitted spline trend from the raw data

    Parameters:
        data:
        trend:
    Returns:

    """
    data.shape
    data = data.reshape(128*128,-1)
    detrended = data - trend
    print(detrended)
    return detrended



def denoiser(path_input_npy, path_output_data, path_output_images_denoised, block_height=8, block_width=8):
    ## files imports
    Ytemp = []
    files = os.listdir(path_input_npy)
    for file in tqdm(files):
        if file.endswith('.npy'):
                img_array= np.load(path_input_npy+"/"+file)
                Ytemp.append([img_array])

    ## Dimensions handling
    y = np.vstack(Ytemp)
    size_y = int(np.sqrt(y[0].size)) ## will only work as long as fov is square
    # size_x, size_y = 128, 1024
    size_x, size_y = 128, 128
    y = y.reshape(len(y), size_y, size_x)

    ## dimensions handling for denoising
    y = np.expand_dims(y, axis=3)
    #3D imaging data needs to be of shape (fov_height, fov_width, nb_channels, time)
    Y = np.transpose(y, [1,2,3,0])
    # Y = np.asarray(Y,order='C',dtype=np.float64)[:,:,:1,608:1376]
    Y = np.asarray(Y,order='C',dtype=np.float64)[:,:,:1,602:1370]
    # Y = np.asarray(Y,order='C',dtype=np.float64)[:,:,:1,:]

    # ## normalize each pixel by mu and sigma of each pixel
    # Y.shape
    # Y = Y.reshape(-1,Y.shape[-1])
    # Y.shape
    # Y_sigma = np.std(Y, axis=0)
    # Y_mean = np.average(Y, axis=0)
    # Y_normalized = (Y - Y_mean) / Y_sigma
    # Y_normalized.shape
    # Y = Y_normalized.reshape(128, 128, 1, -1)

    # detrending
    # trend = get_spline_trend(Y, nb_knots = 5)
    # trend.shape
    # new_Y = remove_spline_trend(Y,trend)
    # Y = new_Y
    # Y.shape

    # pixel = np.random.choice(range(Y.shape[0]), 100)
    # x = np.arange(Y.shape[-1])
    # for i in pixel:
    #     print(i)
    #     _Y = Y[i]
    #     plt.scatter(x,_Y)
    #     plt.plot(x,_Y)
    #     plt.savefig(f'detrended.png')
    # plt.close()

    Y = Y.reshape(128,128,1,-1)
    Y.shape

    fov_height, fov_width, nb_channels, num_frames = Y.shape  ## Note: the input data size needs to match 'block_height' & 'block_width' parameter so
                                                                ## that there will be whole blocks when performing batch processing.
    print("data shape: " + str(Y.shape))

    # Specify Decomp
    max_iters_main = 10
    max_iters_init = 40

    d_sub = 1 #4
    t_sub = 1 #19

    max_comp = 40
    consec_failures = 3
    tol = 5e-3

    block_height = block_height
    block_width = block_width

    # block_height = 32
    # block_width = 32

    enable_temporal_denoiser = True
    enable_spatial_denoiser = True
    overlapping = True

    #Iteratively Simulate & Fit Noise To Collect Samples
    spatial_thresh, temporal_thresh = thresholds(fov_height, fov_width, nb_channels, num_frames, Y, block_height, block_width,
                                                  consec_failures, max_iters_main, max_iters_init, tol,
                                                  d_sub, t_sub, enable_temporal_denoiser, enable_spatial_denoiser,
                                                  path_output_data, plot=True)

    # spatial_thresh = 1.1
    # temporal_thresh = 2.10
    print('batch decompose and recompose processing')
    Y = np.asarray(Y,order='C',dtype=np.float64)[:,:,:,:]
    Y.shape
    outs = overlapping_batch_decompose(fov_height, fov_width, nb_channels, num_frames,
                                       Y, block_height, block_width,
                                       spatial_thresh, temporal_thresh,
                                       max_comp, consec_failures, max_iters_main, max_iters_init, tol,
                                       d_sub, t_sub,
                                       enable_temporal_denoiser, enable_spatial_denoiser)
    Y_den = overlapping_batch_recompose(outs, fov_height, fov_width, block_height, block_width)
    # print(Y_den.shape)

    # rank_vec = outs[0][2]
    # pixelwise_ranks(rank_vec, fov_height, fov_width, num_frames,
    #                 block_height, block_width, dataset=None)
    # plt.close()

    ## display denoised imaging data
    Y_denf = np.asfortranarray(Y_den)
    play_3d(Y_denf, magnification=3, gain=2, save=True,
            path = path_output_images_denoised)
    ## display original imaging data
    Y_f = np.asfortranarray(Y)
    ## display background noise
    Y_r = np.asfortranarray(Y_f[:,:,:,:num_frames] - Y_denf)
    ## display original, denoised, background noise imaging data for comparison
    # tmp = np.vstack([Y_f, Y_denf, Y_r])
    # print(tmp.shape)
    # play_3d(tmp, magnification=1, gain=2, save=False,
    #         path=path_output_images_comparison)

    ## saving images for later if needed
    np.savez(path_output_data + 'processed_data.npz', Y_denoised=Y_denf, Y_original=Y_f, Y_background=Y_r)


if __name__ == "__main__":

    experiment = 'experiment_132'

    input_data_folder = '/mnt/home_nas/jeremy/Recherches/Postdoc/Projects/Memory/Computational_Principles_of_Memory/optopatch/data/2020_03_02'
    path_input_npy = input_data_folder + '/{}/raw_data'.format(experiment)
    path_output_data = input_data_folder + '/{}/'.format(experiment)
    path_output_images_denoised = input_data_folder + '/{}/denoised_images'.format(experiment)
    # path_output_images_comparison = input_data_folder + '/{}/comparison_images'.format(experiment)

    try:
        os.mkdir(path_output_images_denoised)
    except FileExistsError:
        pass
    # os.mkdir(path_output_images_comparison)

    denoiser(path_input_npy, path_output_data, path_output_images_denoised, 16, 16)
