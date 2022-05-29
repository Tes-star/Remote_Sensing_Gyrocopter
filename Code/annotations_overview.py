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
    path_hdr = '../data/Oldenburg_grid_200_200/Teilbild_Oldenburg_labeled_00000000_00000000_0_0_.hdr'
    path_dat = path_hdr[:-4] + '.dat'

    # load image
    img = envi.open(file=path_hdr, image=path_dat)

    # load image array
    arr = img.load()

    # convert only wavelength into dataframe and round numbers
    bands = pd.DataFrame({'wavelength': [x for x in img.metadata['wavelength'] if isfloat(x)]})
    bands['wavelength'] = bands['wavelength'].astype(float).round(1)
    bands = bands['wavelength'].to_list()
    bands.extend(['thermal', 'dom', 'label'])

    # convert array in dataframe
    # every line represent one pixel
    # every column represent one band
    df_bands = pd.DataFrame(arr.reshape((40000, 110)), columns=bands)

    return img, arr, df_bands


# Read Data #############################################################################

img, arr_img, df_bands = read_data()

#########################################################################################

@st.cache
def rgb_image():
    rgb_image = spy.get_rgb(arr_img, bands=(59, 26, 1), stretch=(0.01, 0.99), stretch_all=True)
    return rgb_image


# plot image
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(rgb_image())
st.pyplot(fig)

# plot label mask
v = spy.imshow(classes=arr_img[:,:,109])
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(v.class_rgb)
st.pyplot(fig)

# plot image with mask
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(rgb_image())
ax.imshow(arr_img[:,:,109], alpha=0.35)
st.pyplot(fig)

st.markdown('### Vergleich von Pixeln')

w = df_bands.loc[df_bands['label'] == 1, ]

# Plot Spectrum
fig, ax = plt.subplots(figsize=(8, 4))
for i in range (5,55):
    ax.plot(w.columns[:106], w.iloc[i, :106], '-y')
st.pyplot(fig)




print('Ende')

