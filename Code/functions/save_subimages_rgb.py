import os
import spectral as spy
from spectral import envi


def save_subimages_rgb(path_subimages: str, rgb_bands: tuple, path_export_folder: str, window_width: int,
                       window_height: int):
    """
    function which create for each subimage in path_subimages a jpeg-picture and use bands in rgb_bands
    :param path_subimages: folder path
    :param rgb_bands: list with number of bands for red, green, blue
    :param path_export_folder: folter path
    :param window_width: width of Subimages
    :param window_height: height of Subimages
    :return: save rgb Subimages in  path_export_folder
    """
    # extract grid_size
    windowsize_c = int(window_width)
    windowsize_r = int(window_height)

    # define grid_folder
    grid_folder = path_subimages

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
    if not os.path.exists(path_export_folder):
        os.makedirs(path_export_folder)

    # build rgb subimages
    for file in files_lst:
        # extract grid position from grid filename which has a name convention
        file_split = file.split('_')
        grid_pos_r = int(file_split[2]) * windowsize_r  # Gridposition row
        grid_pos_c = int(file_split[3]) * windowsize_c  # Gridposition column
        # Namenskonvention durchführen
        rgb_name = path_export_folder + "/Teilbild_Oldenburg_" + str(file_split[2]).zfill(8) + "_" + str(
            file_split[3]).zfill(8) + "_" + str(grid_pos_r) + "_" + str(grid_pos_c) + "_.jpg"

        # Pfade definieren
        path_hdr = grid_folder + '/' + file[:-4] + '.hdr'
        path_dat = grid_folder + '/' + file[:-4] + '.dat'

        # Teilbilder laden
        img = envi.open(file=path_hdr, image=path_dat)

        # RGB für Teilbilder erstellen
        spy.save_rgb(filename=rgb_name, data=img, bands=rgb_bands, stretch=(0.1, 0.99), stretch_all=True)

