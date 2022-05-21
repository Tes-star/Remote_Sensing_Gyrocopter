from glob import glob
from spectral.algorithms.algorithms import TrainingClass, TrainingClassSet
from Daten_einlesen.find_path_nextcloud import find_path_nextcloud
from spectral import *
import numpy as np
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go

np.seterr(divide='ignore', invalid='ignore')


# Read Data #############################################################################

# Pfad Nextcloud bestimmen
path_nextcloud = find_path_nextcloud()

# Pfade definieren
path_hdr = path_nextcloud + 'Daten_Gyrocopter/Oldenburg/Teilbilder/grid_250_250/Teilbild_Oldenburg_00000000_00000000_0_0_.hdr'
path_dat = path_hdr[:-4] + '.dat'

# Teilbild laden
img = envi.open(file=path_hdr, image=path_dat)

# Laden der Bildinformationen
arr_img = img.load()

#########################################################################################

# Plot Teilbild
plt.imshow(get_rgb(arr_img, bands=(59, 26, 1), stretch=(0.01, 0.99), stretch_all = True))
plt.show()

#########################################################################################

# overview bands

# global picture
print(img)

# describe band
print(img.metadata['wavelength'])

# convert only wavelength into dataframe and round numbers
df_bands = pd.DataFrame({'wavelength': img.metadata['wavelength'][:-2]})
df_bands['wavelength'] = df_bands['wavelength'].astype(float).round(1)


plt.eventplot(df_bands['wavelength'])
plt.yticks([])
plt.xlabel('Wellenlänge in nm')
plt.show()

# add colorscale https://stackoverflow.com/questions/44959955/matplotlib-color-under-curve-based-on-spectral-color

# distances between bands ?

# convert array in dataframe
# every line represent one pixel
# every column represent one band
df_bands = pd.DataFrame(arr_img.reshape((62500,109)))

#df_bands.describe()

plt.boxplot(df_bands[0])
plt.show()

plt.boxplot(df_bands.iloc[:,:10])
plt.show()

plt.boxplot(df_bands)
plt.show()

plt.boxplot(df_bands)
plt.show()

plt.hist(df_bands[107], bins=100)
plt.show()

plt.plot(df_bands.columns, df_bands.iloc[800,:], marker='o')
plt.show()


plt.plot(df_bands.columns[:106], df_bands.iloc[500,:106], '-b')
plt.show()
