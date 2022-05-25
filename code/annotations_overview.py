from find_path_nextcloud import find_path_nextcloud
import spectral as spy
from spectral import envi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


@st.cache
def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


@st.cache
def read_data():

    # define path
    path_hdr = '../data/Teilbild_Oldenburg_combined_HSI_THERMAL_DOM.hdr'
    path_dat = path_hdr[:-4] + '.dat'

    # load image
    img = envi.open(file=path_hdr, image=path_dat)

    # load image array
    arr = img.load()

    # convert only wavelength into dataframe and round numbers
    bands = pd.DataFrame({'wavelength': [x for x in img.metadata['wavelength'] if isfloat(x)]})
    bands['wavelength'] = bands['wavelength'].astype(float).round(1)
    bands = bands['wavelength'].to_list()
    bands.append('thermal')
    bands.append('dom')

    # convert array in dataframe
    # every line represent one pixel
    # every column represent one band
    df_bands = pd.DataFrame(arr.reshape((40000, 109)), columns=bands)

    return img, arr, df_bands


# Read Data #############################################################################

img, arr_img, df_bands = read_data()

print('Ende')

