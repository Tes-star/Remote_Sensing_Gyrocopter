from find_path_nextcloud import find_path_nextcloud
from spectral import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="centered")
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

st.title('PDS Gyrocopter')
st.markdown('## Datenexploration')


# Read Data #############################################################################

@st.cache
def read_img():
    # calculate path to data
    path_nextcloud = find_path_nextcloud()

    # define path
    path_hdr = path_nextcloud + 'Daten_Gyrocopter/Oldenburg/Teilbilder/grid_200_200/subimages/Teilbild_Oldenburg_00000000_00000000_0_0_.hdr'
    path_dat = path_hdr[:-4] + '.dat'

    # load image
    img = envi.open(file=path_hdr, image=path_dat)

    # load image array
    arr = img.load()

    # convert only wavelength into dataframe and round numbers
    bands = pd.DataFrame({'wavelength': img.metadata['wavelength'][:-2]})
    bands['wavelength'] = bands['wavelength'].astype(float).round(1)
    bands = bands['wavelength'].to_list()
    bands.append('thermal')
    bands.append('dom')

    # convert array in dataframe
    # every line represent one pixel
    # every column represent one band
    df_bands = pd.DataFrame(arr.reshape((40000, 109)), columns=bands)

    return img, arr, df_bands


img, arr_img, df_bands = read_img()

#########################################################################################

# overview bands and wavelength

# convert only wavelength into dataframe and round numbers
bands = pd.DataFrame({'wavelength': img.metadata['wavelength'][:-2]})
bands['wavelength'] = bands['wavelength'].astype(float).round(1)

# plot bands
fig, ax = plt.subplots(figsize=(8, 4))
ax.eventplot(bands['wavelength'])
plt.yticks([])
plt.xlabel('Wellenlänge in nm')
st.pyplot(fig)

# add colors https://stackoverflow.com/questions/44959955/matplotlib-color-under-curve-based-on-spectral-color

st.markdown('___________________________\n')

#########################################################################################

st.markdown('### Ausgabe Spectrum für einen Pixel')

x_value = st.slider('x-Koordinate', min_value=0, max_value=200, value=100)
y_value = st.slider('y-Koordinate', min_value=0, max_value=200, value=100)


@st.cache
def rgb_image():
    rgb_image = get_rgb(arr_img, bands=(59, 26, 1), stretch=(0.01, 0.99), stretch_all=True)
    return rgb_image


# Plot Subimage
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 8))
ax1.imshow(rgb_image())
ax1.scatter(x_value, y_value, c='y', s=60)

# Plot Spectrum
ax2.plot(df_bands.columns[:106], df_bands.iloc[x_value * y_value, :106], '-b')
st.pyplot(fig)

#########################################################################################

st.markdown('### Vergleich von Pixeln')

# Plot Spectrum
fig, ax = plt.subplots(figsize=(8, 4))
line1, = ax.plot(df_bands.columns[:106], df_bands.iloc[147, :106], '-y', label='Pixel1')
line2, = ax.plot(df_bands.columns[:106], df_bands.iloc[150, :106], '-r', label='Pixel2')
line3, = ax.plot(df_bands.columns[:106], df_bands.iloc[151, :106], '-g', label='Pixel3')
ax.legend(handles=[line1, line2, line3])
st.pyplot(fig)

#########################################################################################

