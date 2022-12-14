# Import Modules
import os

import numpy as np
import spectral.io.envi as envi
from sklearn.preprocessing import StandardScaler

from Code.find_path_nextcloud import find_path_nextcloud
from Code.functions.class_ids import new_label_mapping_cnn


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


def import_labeled_photos(bands=slice(0, 109), label_mapping=None):
    # Standart Label Mapping
    label_mapping=new_label_mapping_cnn(label_mapping)

    # Pfad Nextcloud bestimmen

    path_nextcloud = find_path_nextcloud()

    # Festlegung, welches Grid zusammengelegt werden soll
    windowsize_r = 200
    windowsize_c = 200

    # Bestimmung annotation_folder
    Oldenburg_folder = path_nextcloud + "Daten_Gyrocopter/Oldenburg/"
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
    #
    path_dat = Oldenburg_folder + 'Oldenburg_combined_HSI_THERMAL_DOM.dat'
    path_hdr = Oldenburg_folder + 'Oldenburg_combined_HSI_THERMAL_DOM.hdr'
    img = envi.open(file=path_hdr, image=path_dat)
    data = img.open_memmap(writable=False)
    max_value = np.zeros(109)
    for i in range(109):
        max_value[i] = data[:, :, i].max()

    for filename in files:
        path_dat = labeled_folder + os.path.splitext(filename)[0] + '.dat'
        path_hdr = labeled_folder + os.path.splitext(filename)[0] + '.hdr'
        img = envi.open(file=path_hdr, image=path_dat)
        data = img.open_memmap(writable=False)
        # dataholder = np.ndarray(shape=(200, 200, 110))
        dataholder = np.zeros((224, 224, 119), dtype=float)
        # for row in range(0, data.shape[0]):
        #    for column in range(0, data.shape[1]):
        dataholder[0:200, 0:200, 0:110] = data[:, :, 0:110]

        # reduce values between 0 and 1
        for i in range(109):
            dataholder[0:200, 0:200, i] = dataholder[0:200, 0:200, i] / max_value[i]

        # one hot encode dataholder with classes
        for row in range(0, 200):
            for column in range(0, 200):
                label_id = int(dataholder[row, column, 109])
                # map label_id to label_mapping if annotation labels are not used labels
                label_id = label_mapping[label_id]

                dataholder[row, column, label_id+ 110] = 1

        # dataholder[np.where(dataholder == 0)] = -1

        # if bands is None:
        #   bands = "0:109"

        # X.append(dataholder[:, :, 0:108])
        X.append(dataholder[:, :, bands])
        Y.append(dataholder[:, :, 110:111+max(label_mapping)])
        # np.unique(dataholder[0:200,0:200,109], return_counts=True)
        # dict(zip(unique, counts))
    return X, Y


if __name__ == '__main__':
    bands = list(range(0, 104, 14))
    bands.append(105)
    bands.append(106)
    bands.append(107)
    bands.append(108)
    x, y = import_labeled_photos(bands=bands, label_mapping='Ohne_Auto_See')

    x_train = x[0:22]
    x_train.append(x[29:])
    y_train = y[0:22]
    y_train.append(y[29:])

    x_test = x[22:29]
    y_test = y[22:29]
    del x,y


