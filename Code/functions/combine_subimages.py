import os
import numpy as np
from spectral import envi

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


def combine_subimages(hdr_file: str, dat_file: str, path_grid_subimages: str, path_export: str, window_width: int,
                      window_height: int, combine_annotated_images: bool = False):
    """
    function which combine multiple Subimages to one big picture and save it
    :param window_height: height of Subimages
    :param window_width: width of Subimages
    :param path_export: export place of the big picture
    :param hdr_file: original big picture envi-header-file with ending .hdr
    :param dat_file: original big picture envi-image-file with ending .dat
    :param path_grid_subimages: path to grid folder with subimages
    :return: save big pictures and return absolute path of it
    """

    # read image
    img = envi.open(file=hdr_file, image=dat_file)

    # extract grid_size
    windowsize_c = int(window_width)
    windowsize_r = int(window_height)

    # define export folder name
    grid_folder = path_grid_subimages

    # Select all .dat Subimages

    # build list with all files in grid_folder
    files_lst = os.listdir(grid_folder)

    # remove unwanted files from files_lst
    unwanted_pattern = ['.hdr', 'combined_big_picture', 'rgb', '.xml', 'labeled']
    for pattern in unwanted_pattern:
        for file in files_lst:
            if pattern in file:
                files_lst.remove(file)

    # build export folder
    if not os.path.exists(path_export):
        os.makedirs(path_export)

    # define big picture name
    path_big_picture_hdr = path_export + '/grid_' + str(windowsize_r) + '_' + str(
        windowsize_c) + '_combined_big_picture.hdr'

    # use if combine_annotated_images == True to combine annotated and not annotated pictures to one big picture
    if combine_annotated_images:
        img.metadata['bands'] = 110

    # build empty envi file with matching dimension
    grid = envi.create_image(hdr_file=path_big_picture_hdr, metadata=img.metadata, dtype="float32", ext='.dat',
                             interleave='bsq',
                             force=True)

    # create writeable access
    writer = grid.open_memmap(writable=True)

    # combine subimages with writer method

    # Iteration over all .dat subimages
    for file in files_lst:
        # extract grid position from grid filename which has a name convention
        file_split = file.split('_')
        grid_pos_r = int(file_split[2]) * windowsize_r  # Gridposition row
        grid_pos_c = int(file_split[3]) * windowsize_c  # Gridposition column

        # define path-filename-combination from .hdr and .dat file
        path_hdr = grid_folder + "/" + file[:-4] + '.hdr'
        path_dat = grid_folder + "/" + file

        # read subimage
        image_small = envi.open(file=path_hdr, image=path_dat)

        image_small_arr = image_small.load()

        # use if combine_annotated_images == True to combine annotated and not annotated pictures to one big picture
        if combine_annotated_images:
            if image_small_arr.shape[2] == 109:
                zero_mask = np.ones((image_small_arr.shape[0], image_small_arr.shape[1], 1)) * -1
                image_small_arr = np.concatenate((image_small_arr, zero_mask), -1)

        # insert subimage in grid
        writer[grid_pos_r:grid_pos_r + windowsize_r, grid_pos_c:grid_pos_c + windowsize_c, :] = image_small_arr

    # read big picture
    path_big_picture_dat = path_big_picture_hdr[:-4] + '.dat'
    big_picture = envi.open(file=path_big_picture_hdr, image=path_big_picture_dat)

    return big_picture