# Import Modules
import os

import numpy as np
import spectral.io.envi as envi
from sklearn.preprocessing import StandardScaler

from Code.find_path_nextcloud import find_path_nextcloud


def import_labeled_data():
    # Pfad Nextcloud bestimmen

    path_nextcloud = find_path_nextcloud()

    # Festlegung, welches Grid zusammengelegt werden soll
    windowsize_r = 200
    windowsize_c = 200

    # Bestimmung annotation_folder
    grid_folder = path_nextcloud + "Daten_Gyrocopter/Oldenburg/Teilbilder/grid_" + str(windowsize_r) + "_" + str(
        windowsize_c)
    labeled_folder = grid_folder + '/labeled/'

    # Liste aller Dateien in annotation_folder erstellen
    files = os.listdir(labeled_folder)

    # Aus Liste files .hdr Dateien löschen
    for file in files:
        if not file.endswith('.dat'):
            files.remove(file)
    X = []
    Y = []
    # labeled Bilder erstellen
    for filename in files:
        path_dat = labeled_folder + os.path.splitext(filename)[0] + '.dat'
        path_hdr = labeled_folder + os.path.splitext(filename)[0] + '.hdr'
        img = envi.open(file=path_hdr, image=path_dat)
        data = img.open_memmap(writable=False)
        for row in range(0, data.shape[0]):
            for column in range(0, data.shape[1]):
                X.append(data[row][column][0:109])
                Y.append(data[row][column][117])
    return X, Y


def import_labeled_photos():
    # Pfad Nextcloud bestimmen

    path_nextcloud = find_path_nextcloud()

    # Festlegung, welches Grid zusammengelegt werden soll
    windowsize_r = 200
    windowsize_c = 200

    # Bestimmung annotation_folder
    grid_folder = path_nextcloud + "Daten_Gyrocopter/Oldenburg/Teilbilder/grid_" + str(windowsize_r) + "_" + str(
        windowsize_c)
    labeled_folder = grid_folder + '/labeled/'

    # Liste aller Dateien in annotation_folder erstellen
    files = os.listdir(labeled_folder)

    # Aus Liste files .hdr Dateien löschen
    for file in files:
        if not file.endswith('.dat'):
            files.remove(file)
    for file in files:
        if file == 'wrong_image_size':
            files.remove(file)
    X = []
    Y = []
    # labeled Bilder erstellen
    for filename in files:
        path_dat = labeled_folder + os.path.splitext(filename)[0] + '.dat'
        path_hdr = labeled_folder + os.path.splitext(filename)[0] + '.hdr'
        img = envi.open(file=path_hdr, image=path_dat)
        data = img.open_memmap(writable=False)
        # dataholder = np.ndarray(shape=(200, 200, 110))
        dataholder = np.zeros((224, 224, 110))
        # for row in range(0, data.shape[0]):
        #    for column in range(0, data.shape[1]):
        dataholder[0:200,0:200,:] = data[:][:][:]

        # X.append(dataholder[0:200, 0:200, 0:108])
        # Y.append(dataholder[0:200, 0:200, 109])
        X.append(dataholder[:, :, 0:108])
        Y.append(dataholder[:, :, 109])
    return X, Y


def scale_data(data):
    scaler = StandardScaler().fit(data)
    data = scaler.transform(data)
    return data


X, Y = import_labeled_photos()
