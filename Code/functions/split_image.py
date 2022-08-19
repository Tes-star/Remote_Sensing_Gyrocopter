import os
import cv2
import numpy as np
import pandas as pd
import spectral as spy
from spectral import envi
import xmltodict as xmltodict
import matplotlib.pyplot as plt


def split_image(hdr_file: str, dat_file: str, window_width: int, window_height: int, export_path: str,
                export_title: str, stop_after_row=-1):
    """
    function which split one Image in multiple Subimages and save it
    :param hdr_file: envi-header-file with ending .hdr
    :param dat_file: envi-image-file with ending .dat
    :param window_width: width of Subimages
    :param window_height: height of Subimages
    :param export_path: path to which the Subimages are to be saved
    :param export_title: title of the image which is included in the export name
    :param stop_after_row: optional number if not all Subimages should be saved
    :return: save Subimages
    """

    # read image
    img = envi.open(file=hdr_file, image=dat_file)

    # define grid_size
    windowsize_r = window_width
    windowsize_c = window_height

    # define export folder name
    grid_folder = export_path + "/" + export_title + "_grid_" + str(windowsize_r) + "_" + str(windowsize_c)

    # create global_folger
    if not os.path.exists(grid_folder):
        os.makedirs(grid_folder)

    # build grid_folder
    grid_subimages = grid_folder + "/subimages"
    if not os.path.exists(grid_subimages):
        os.makedirs(grid_subimages)

    # Split Image in Subimages

    grid_r = 0  # gridrow
    grid_c = 0  # gridcolumn

    # Iterate over image matrix rows in interval windowsize_r
    for r in range(0, img.shape[0], windowsize_r):

        # Iterate over image matrix columns in interval windowsize_c
        for c in range(0, img.shape[1], windowsize_c):
            # select Subimage
            window = img[r:r + windowsize_r, c:c + windowsize_c, :]
            print(str(r), '/', str(c))

            # define export hdf-filename
            path_window = grid_subimages + "/Teilbild_" + export_title + '_' + str(grid_r).zfill(8) + "_" + str(
                grid_c).zfill(8) + "_" + str(r) + "_" + str(c) + "_.hdr"

            # save Subimage
            envi.save_image(hdr_file=path_window, image=window,
                            dtype="float32", ext='.dat', interleave='bsq',
                            metadata=img.metadata, force=True)

            # change to next column
            grid_c = grid_c + 1

        # change to next row
        grid_r = grid_r + 1
        # reset grid_c
        grid_c = 0

        # stop if parameter stop_after_row with characteristic 0 < stop_after_row < max_rows used and equal grid_r
        if grid_r == stop_after_row:
            break

    return grid_subimages
