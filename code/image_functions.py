import os
import spectral as spy
from spectral import envi
from find_path_nextcloud import find_path_nextcloud

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


def print_overview(hdr_file: str, dat_file: str):
    """
    description
    :param hdr_file:
    :param dat_file:
    :return:
    """

    img = envi.open(file=hdr_file, image=dat_file)
    print(img)

    # TODO finde useful output


def plot_hsi_image(hdr_file: str, dat_file: str, bands: tuple, stretch: tuple):
    pass


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
            path_window = grid_folder + "/Teilbild_" + export_title + '_' + str(grid_r).zfill(8) + "_" + str(
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

    return grid_folder


def combine_subimages(hdr_file: str, dat_file: str, path_grid_subimages: str):
    """
    function which combine multiple Subimages to one big picture and save it
    :param hdr_file: original big picture envi-header-file with ending .hdr
    :param dat_file: original big picture envi-image-file with ending .dat
    :param path_grid_subimages: path to grid folder with subimages
    :return: save big pictures
    """

    # read image
    img = envi.open(file=hdr_file, image=dat_file)

    # extract grid_size
    window_width, window_height = path_grid_subimages.split('/')[-1].split('_')[-2:]
    windowsize_r = int(window_width)
    windowsize_c = int(window_height)

    # define export folder name
    grid_folder = path_grid_subimages

    # Select all .dat Subimages

    # build list with all files in grid_folder
    files_lst = os.listdir(grid_folder)

    # remove unwanted files from files_lst
    unwanted_pattern = ['.hdr', 'combined_big_picture', 'rgb']
    for pattern in unwanted_pattern:
        for file in files_lst:
            if pattern in file:
                files_lst.remove(file)

    # define big picture name
    path_big_picture_hdr = grid_folder + '/grid_' + str(windowsize_r) + '_' + str(windowsize_c) + '_combined_big_picture.hdr'

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

        # insert subimage in grid
        writer[grid_pos_r:grid_pos_r + windowsize_r, grid_pos_c:grid_pos_c + windowsize_c, :] = image_small.open_memmap(
            writable=False)

    # read big picture
    path_big_picture_dat = path_big_picture_hdr[:-4] + '.dat'
    big_picture = envi.open(file=path_big_picture_hdr, image=path_big_picture_dat)

    return big_picture


def save_subimages_rgb(path_grid_subimages: str, rgb_band:tuple):

    # extract grid_size
    window_width, window_height = path_grid_subimages.split('/')[-1].split('_')[-2:]
    windowsize_r = int(window_width)
    windowsize_c = int(window_height)

    # define grid_folder
    grid_folder = path_grid_subimages

    # Select all .dat Subimages

    # build list with all files in grid_folder
    files_lst = os.listdir(grid_folder)

    # remove unwanted files from files_lst
    unwanted_pattern = ['.hdr', 'combined_big_picture', 'rgb']
    for pattern in unwanted_pattern:
        for file in files_lst:
            if pattern in file:
                files_lst.remove(file)

    # build rgb folder
    path_grid_rgb = grid_folder + '/rgb'
    if not os.path.exists(path_grid_rgb):
        os.makedirs(path_grid_rgb)

    # build rgb subimages
    for file in files_lst:

        # extract grid position from grid filename which has a name convention
        file_split = file.split('_')
        grid_pos_r = int(file_split[2]) * windowsize_r  # Gridposition row
        grid_pos_c = int(file_split[3]) * windowsize_c  # Gridposition column
        # Namenskonvention durchführen
        rgb_name = path_grid_rgb + "/Teilbild_Oldenburg_" + str(file_split[2]).zfill(8) + "_" + str(
            file_split[3]).zfill(8) + "_" + str(grid_pos_r) + "_" + str(grid_pos_c) + "_.jpg"

        # Pfade definieren
        path_hdr = grid_folder + '/' + file[:-4] + '.hdr'
        path_dat = grid_folder + '/' + file[:-4] + '.dat'

        # Teilbilder laden
        img = envi.open(file=path_hdr, image=path_dat)

        # RGB für Teilbilder erstellen
        spy.save_rgb(filename=rgb_name, data=img, bands=rgb_band, stretch=(0.1,0.99), stretch_all = True)


if __name__ == '__main__':

    # define shared could path
    path_nextcloud = find_path_nextcloud()

    # define path with data
    path_folder = path_nextcloud + 'Daten_Gyrocopter/Oldenburg/'

    # define HSI filenames
    path_hdr = path_folder + 'Oldenburg_HSI.hdr'
    path_dat = path_folder + 'Oldenburg_HSI.dat'

    # read image and print infos
    print_overview(hdr_file=path_hdr, dat_file=path_dat)

    # split image in subimages
    path_grid_folder = split_image(hdr_file=path_hdr, dat_file=path_dat, window_width=1000, window_height=1000,
                                   export_path='C:/Users/fgrassxx/Desktop', export_title='Oldenburg', stop_after_row=1)

    # combine subimages to big picture
    big_picture = combine_subimages(hdr_file=path_hdr, dat_file=path_dat, path_grid_subimages=path_grid_folder)

    # save subimages as rgb
    save_subimages_rgb(path_grid_subimages = path_grid_folder, rgb_band = (59, 26, 1))

    print('Fertig')
