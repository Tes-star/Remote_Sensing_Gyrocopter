import os
import spectral as spy
from spectral import envi

"""
Spectral Python (SPy) Functions
https://www.spectralpython.net/
"""

# function to read Envi-files
# img = envi.open(file='filename.hdr', image='filename.dat')

# function to convert Envi-file img in array (relates to above img)
# img_arr = img.load()

"""
own functions
"""


def print_overview(hdr_file:str, dat_file:str):
    """
    description
    :param hdr_file:
    :param dat_file:
    :return:
    """

    img = envi.open(file=hdr_file, image=dat_file)
    print(img)

    # TODO finde useful output


def plot_hsi_image(hdr_file:str, dat_file:str, bands:tuple, stretch:tuple):
    pass


def split_image(hdr_file:str, dat_file:str, window_width:int, window_height:int, export_path:str, export_title:str, stop_after_row = -1):

    # read image
    img = envi.open(file=hdr_file, image=dat_file)

    # define grid_size
    windowsize_r = window_width
    windowsize_c = window_height

    # define export folder name
    grid_folder = export_path + "/grid_" + str(windowsize_r) + "_" + str(windowsize_c)

    # create grid_folger
    if not os.path.exists(grid_folder):
        os.makedirs(grid_folder)

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
            path_window = grid_folder + "/Teilbild_" + export_title + '_' + str(grid_r).zfill(8) + "_" + str(grid_c).zfill(
                8) + "_" + str(r) + "_" + str(c) + "_.hdr"

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

if __name__ == '__main__':
    print('Fertig')