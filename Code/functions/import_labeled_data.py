import os
import cv2
import numpy as np
import pandas as pd
import spectral as spy
from spectral import envi
import xmltodict as xmltodict
import matplotlib.pyplot as plt

def import_labeled_data(path_labeled_folder:str):

    # Liste aller Dateien in annotation_folder erstellen
    files = os.listdir(path_labeled_folder)

    # Aus Liste files .hdr Dateien löschen
    for file in files:
        if not file.endswith('.dat'):
            files.remove(file)

    for file in files:
        if 'wrong_image_size' in file:
            files.remove(file)

    # Spaltennamen des DataFrames bilden
    path_dat = path_labeled_folder + os.path.splitext(files[0])[0] + '.dat'
    path_hdr = path_labeled_folder + os.path.splitext(files[0])[0] + '.hdr'

    # load image
    img = spy.envi.open(file=path_hdr, image=path_dat)

    def isfloat(num):
        try:
            float(num)
            return True
        except ValueError:
            return False

    # convert only wavelength into dataframe and round numbers
    value_bands = ['hsi_band_' + str(int(float(x))) + '_nm' for x in img.metadata['wavelength'] if isfloat(x)]
    value_bands.extend(['thermal', 'dom'])

    label_bands = ['label']

    bands = []
    bands.extend(value_bands)
    bands.extend(label_bands)
    bands.append('picture_name')

    df_annotations = pd.DataFrame(columns=bands)

    # labeled Bilder erstellen
    for filename in files:
        path_dat = path_labeled_folder + '/' + os.path.splitext(filename)[0] + '.dat'
        path_hdr = path_labeled_folder + '/' + os.path.splitext(filename)[0] + '.hdr'
        img = spy.envi.open(file=path_hdr, image=path_dat)

        arr = img.load()

        df = pd.DataFrame(arr.reshape(((arr.shape[0]*arr.shape[1]), arr.shape[2])), columns=bands[:-1])
        df['picture_name'] = os.path.splitext(filename)[0]

        df_annotations = pd.concat([df_annotations, df], ignore_index=True)

        df_annotations['label'] = df_annotations['label'].astype(int)

    return df_annotations
