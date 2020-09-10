#!/usr/bin/env python
# coding: utf-8

# General Dependencies
import os
import numpy as np
import matplotlib.pyplot as plt

# Preprocessing Dependencies
from trefide.utils import psd_noise_estimate  ## return the variance per pixel. then do pixel - mu / psd_noise_estimate

# PMD Model Dependencies
from trefide.pmd import batch_decompose, \
                        batch_recompose, \
                        overlapping_batch_decompose, \
                        overlapping_batch_recompose, \
                        determine_thresholds

from trefide.reformat import overlapping_component_reformat

# Plotting & Video Rendering Dependencies
import matplotlib.pyplot as plt
from trefide.plot import pixelwise_ranks
from trefide.extras.util_plot import comparison_plot
from trefide.video import play_cv2


def denoising_pipeline(input_data_folder, experiment):

    input_data_folder = f'{input_data_folder}/{experiment}/'
    filename = os.path.join(input_data_folder, "raw_data.npy")
    save_fig_path = input_data_folder

    # ### Load Preprocessed Data and set parameters
    mov = np.load(filename).astype('double')
    num_frames, fov_height, fov_width= mov.shape
    mov = mov.reshape(fov_height, fov_width, num_frames)
    # mov = mov[:,:,600:600+256]
    mov = mov[:,:,:]
    mov = mov.copy(order='C')
    fov_height, fov_width, num_frames = mov.shape

    # Generous maximum of rank 50 blocks (safeguard to terminate early if this is hit)
    max_components = 50

    # Enable Decimation
    max_iters_main = 10
    max_iters_init = 40
    d_sub = 2
    t_sub = 2

    # Defaults
    consec_failures = 3
    tol = 5e-3

    # Set Blocksize Parameters
    block_height = 64 #40
    block_width = 64  #40
    overlapping = True
    enable_temporal_denoiser = True
    enable_spatial_denoiser = True


    # ### Simulate Critical Region Using Noise, determine spatial & temporal threshold

    spatial_thresh, temporal_thresh = determine_thresholds((fov_height, fov_width, num_frames),
                                                           (block_height, block_width),
                                                           consec_failures,
                                                           max_iters_main,
                                                           max_iters_init,
                                                           tol,
                                                           d_sub,
                                                           t_sub,
                                                           5,
                                                           True,
                                                           input_data_folder,
                                                           enable_temporal_denoiser,
                                                           enable_spatial_denoiser)



    #### Decompose Each Block Into Spatial & Temporal Components
    if not overlapping:
        # Blockwise Parallel, Single Tiling
        spatial_components, \
        temporal_components,block_ranks,\
        block_indices = batch_decompose(fov_height, fov_width, num_frames,
                                        mov, block_height, block_width,
                                        spatial_thresh, temporal_thresh,
                                        max_components, consec_failures,
                                        max_iters_main, max_iters_init, tol,
                                        d_sub, t_sub,
                                        enable_temporal_denoiser, enable_spatial_denoiser)
    else:
        # Blockwise Parallel, 4x Overlapping Tiling
        spatial_components,\
        temporal_components,\
        block_ranks,\
        block_indices,\
        block_weights = overlapping_batch_decompose(fov_height, fov_width, num_frames,
                                                    mov, block_height, block_width,
                                                    spatial_thresh, temporal_thresh,
                                                    max_components, consec_failures,
                                                    max_iters_main, max_iters_init, tol,
                                                    d_sub, t_sub,
                                                    enable_temporal_denoiser, enable_spatial_denoiser)


    #### Reconstruct Denoised Video
    if not overlapping:
        # Single Tiling (No need for reqweighting)
        mov_denoised = np.asarray(batch_recompose(spatial_components,
                                                  temporal_components, block_ranks,
                                                  block_indices))
    else:
        # Overlapping Tilings With Reweighting
        mov_denoised = np.asarray(overlapping_batch_recompose(fov_height, fov_width, num_frames,
                                                              block_height, block_width,
                                                              spatial_components,
                                                              temporal_components,
                                                              block_ranks,
                                                              block_indices,
                                                              block_weights))
        np.save(os.path.join(input_data_folder, "denoised_data.npy"), mov_denoised)



    # ### Produce Diagnostics, Single Tiling Pixel-Wise Ranks
    if overlapping:
        pixelwise_ranks(block_ranks['no_skew']['full'],
                        fov_height,
                        fov_width,
                        num_frames,
                        block_height,
                        block_width,
                        save_fig = True,
                        save_fig_path = input_data_folder)
    else:
        pixelwise_ranks(block_ranks,
                        fov_height,
                        fov_width,
                        num_frames,
                        block_height,
                        block_width,
                        True,
                        input_data_folder)


    #### Correlation Images
    comparison_plot([mov,
                    mov_denoised + np.random.randn(np.prod(mov.shape)).reshape(mov.shape)*.01],
                    plot_orientation = "horizontal",
                    save_fig = True,
                    save_fig_path = input_data_folder)


    #### Save Results
    U, V = overlapping_component_reformat(fov_height, fov_width, num_frames,
                                          block_height, block_width,
                                          spatial_components,
                                          temporal_components,
                                          block_ranks,
                                          block_indices,
                                          block_weights)
    np.save(os.path.join(input_data_folder, "denoising_components.npy"), U, V)




if __name__ == "__main__":

    experiment = 'experiment_132'
    input_data_folder = f'/mnt/home_nas/jeremy/Recherches/Postdoc/Projects/Memory/Computational_Principles_of_Memory/optopatch/data/2020_03_02'
    denoising_pipeline(input_data_folder, experiment)
