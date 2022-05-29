import io
import os

import numpy as np
import xmltodict as xmltodict
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from matplotlib.collections import PatchCollection
# print(matplotlib.get_backend())
# matplotlib.use('Agg')

# Import Packages
from spectral.io import envi
#from spectral import *
from code.find_path_nextcloud import find_path_nextcloud

# Pfad Nextcloud bestimmen
path_nextcloud = find_path_nextcloud()

# Festlegung, welches Grid zusammengelegt werden soll
windowsize_r = 200
windowsize_c = 200

# Bestimmung annotation_folder
grid_folder = path_nextcloud + "Daten_Gyrocopter/Oldenburg/Teilbilder/grid_" + str(windowsize_r) + "_" + str(
    windowsize_c)
annotation_folder = grid_folder + '/Export_labeled/'

# Liste aller Dateien in annotation_folder erstellen
files = os.listdir(annotation_folder)

# Aus Liste files .hdr Dateien löschen
for file in files:
    if not file.endswith('.xml'):
        files.remove(file)

# labeled Ordner erstellen
path_labeled = grid_folder + '/labeled'
if not os.path.exists(path_labeled):
    os.makedirs(path_labeled)

# labeled Bilder erstellen
for filename in files:

    file_split = filename.split('_')
    grid_pos_r = int(file_split[2]) * windowsize_r  # Gridposition row
    grid_pos_c = int(file_split[3]) * windowsize_c  # Gridposition column
    # Namenskonvention durchführen
    original_name = grid_folder + "/Teilbild_Oldenburg_" + str(file_split[2]).zfill(8) + "_" + str(file_split[3]).zfill(
        8) + "_" + str(grid_pos_r) + "_" + str(grid_pos_c)

    file = annotation_folder + filename

    with open(file) as fd:
        doc = xmltodict.parse(fd.read())

    objects = ['Wiese', 'Straße', 'Auto', 'See', 'Schienen', 'Haus', 'Wald']
    data_holder = np.ndarray(shape=[200, 200, len(objects) + 1])
    obj_nr = 0
    for obj in objects:
        fig, ax = plt.subplots()
        plt.xlim(0, 200)
        plt.ylim(0, 200)

        obj_nr = obj_nr + 1
        patches = []
        for object in doc['annotation']['object']:

            coords = []
            polygon = object['polygon']
            name = object['name']
            if obj != name:
                continue
            for i in range(1, int(len(polygon) / 2) + 1):
                point = [polygon['x' + str(i)], polygon['y' + str(i)]]
                coords.append(point)
            patches.append(Polygon(coords))

        p = PatchCollection(patches, match_original=True)
        ax.add_collection(p)

        ax.set_axis_off()
        fig.tight_layout(pad=0)
        fig.set_size_inches(2, 2)


        with io.BytesIO() as buff:
            fig.savefig(buff, format='raw', pad_inches=0)
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        data2 = data.reshape((int(h), int(w), -1))

        for column in range(0, 200):
            for row in range(0, 200):
                colour = data2[column][row][0:3]
                if (colour != [255, 255, 255]).any():
                    data_holder[row][column][obj_nr - 1] = obj_nr
                    data_holder[row][column][len(objects)] = obj_nr #*100
    path_dat = original_name +'_.dat'
    path_hdr = original_name +'_.hdr'
    #path_dat = r'C:\Users\Timo\Nextcloud\Freigaben\Projekt_Data_Science_1_SS22\Daten_Gyrocopter\Oldenburg\Teilbilder\grid_200_200\Teilbild_Oldenburg_00000000_00000000_0_0_.dat'
    #path_hdr = r'C:\Users\Timo\Nextcloud\Freigaben\Projekt_Data_Science_1_SS22\Daten_Gyrocopter\Oldenburg\Teilbilder\grid_200_200\Teilbild_Oldenburg_00000000_00000000_0_0_.hdr'

    img = envi.open(file=path_hdr, image=path_dat)

    arr_metadata = img.metadata

    for new_band in objects:
        arr_metadata['wavelength'].append(new_band)
        arr_metadata['band names'].append(new_band)
        arr_metadata['fwhm'].append(new_band)
    arr_metadata['wavelength'].append('label')
    arr_metadata['band names'].append('label')
    arr_metadata['fwhm'].append('label')

    arr_metadata['bands'] = int(arr_metadata['bands']) + len(objects) + 2
    name="/Teilbild_Oldenburg_" + str(file_split[2]).zfill(8) + "_" + str(file_split[3]).zfill(
        8) + "_" + str(grid_pos_r) + "_" + str(grid_pos_c)
    path_dat_labbeled = path_labeled +name+'_.dat'
    path_hdr_labbeled = path_labeled +name+'_.hdr'
    #path_dat_labbeled = r'C:\Users\Timo\Nextcloud\Freigaben\Projekt_Data_Science_1_SS22\Daten_Gyrocopter\Oldenburg\Teilbilder\grid_200_200\labeled\Teilbild_Oldenburg_00000000_00000000_0_0_.dat'
    #path_hdr_labbeled = r'C:\Users\Timo\Nextcloud\Freigaben\Projekt_Data_Science_1_SS22\Daten_Gyrocopter\Oldenburg\Teilbilder\grid_200_200\labeled\Teilbild_Oldenburg_00000000_00000000_0_0_.hdr'

    img_labbeled = envi.create_image(hdr_file=path_hdr_labbeled, metadata=arr_metadata, dtype="float32", ext='.dat',
                                     interleave='bsq',
                                     force=True)

    data_holder=np.rot90(data_holder, 1)
    # Zugriff auf Bild Matrix erstellen
    N = 200
    writer = img_labbeled.open_memmap(writable=True)
    writer[:, :, 0:109] = img.open_memmap(writable=False)

    writer[:, :, 110:110 + arr_metadata['bands']] = data_holder
    np.seterr(divide='ignore', invalid='ignore')
    import spectral.io.envi as envi

    path_dat_labbeled = r'C:\Users\Timo\Nextcloud\Freigaben\Projekt_Data_Science_1_SS22\Daten_Gyrocopter\Oldenburg\Teilbilder\grid_200_200\labeled\Teilbild_Oldenburg_00000000_00000000_0_0_.dat'
    path_hdr_labbeled = r'C:\Users\Timo\Nextcloud\Freigaben\Projekt_Data_Science_1_SS22\Daten_Gyrocopter\Oldenburg\Teilbilder\grid_200_200\labeled\Teilbild_Oldenburg_00000000_00000000_0_0_.hdr'
    # del plots
    from matplotlib import pyplot as plt
    fig = plt.figure()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    #img = envi.open(file=path_hdr_labbeled, image=path_dat_labbeled)
    #img


    #img = open_image(file=path_hdr_labbeled)

#imshow(pixel, figsize=(30, 50), bands=[110])
    #for i in range(0, len(objects)+1):
        #imshow(img, figsize=(30, 50), bands=[110 + i])
#imshow(img, figsize=(30, 50), bands=(59,26,1))
#imshow(img, figsize=(30, 50), bands=(59,26,1))

