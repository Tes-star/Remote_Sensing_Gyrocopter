import io

import numpy as np
import xmltodict as xmltodict
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# Import Packages
from spectral.io import envi

from Daten_einlesen.find_path_nextcloud import find_path_nextcloud

# Pfad Nextcloud bestimmen
path_nextcloud = find_path_nextcloud()

# Festlegung, welches Grid zusammengelegt werden soll
windowsize_r = 200
windowsize_c = 200

# Bestimmung grid Ordnername
annotation_folder = path_nextcloud + "Daten_Gyrocopter/Oldenburg/Teilbilder/grid_" + str(windowsize_r) + "_" + str(
    windowsize_c) + '/Export_labeled/'

# parse an xml file by name
file = annotation_folder + 'Teilbild_Oldenburg_00000000_00000000_0_0__jpg.rf.36224be5c78237378b09eb42a52e83b1.xml'

with open(file) as fd:
    doc = xmltodict.parse(fd.read())

fig, ax = plt.subplots()

plt.xlim(0, 200)
plt.ylim(0, 200)

for object in doc['annotation']['object']:

    patches = []
    coords = []
    polygon = object['polygon']
    name = object['name']
    if (name == 'Haus'):
        colour = 'red'
        Ebene=5
    if (name == 'Auto'):
        colour = 'black'
        Ebene = 2
    if (name == 'Straße' or name == 'Strase'):
        colour = 'grey'
        Ebene = 1
    if (name == 'Wiese'):
        colour = 'green'
        Ebene=0
    if (name == 'Wald'):
        colour = 'Yellow'
        Ebene=6
    if (name == 'See'):
        colour = 'Blue'
        Ebene = 3
    if (name == 'Schienen'):
        colour = 'Magenta'
        Ebene=4

    for i in range(1, int(len(polygon) / 2) + 1):
        point = [polygon['x' + str(i)], polygon['y' + str(i)]]
        coords.append(point)
    patches.append(Polygon(coords, ec=colour, color=colour,zorder=Ebene,linewidth=0,aa=None))
    p = PatchCollection(patches, match_original=True,zorder=Ebene)

    ax.add_collection(p)

# plt.ioff()
# plt.axis('off')
# plt.show()
ax.set_axis_off()
# fig = ax.figure
fig.tight_layout(pad=0)
fig.set_size_inches(2, 2)
plt.pause(0.05)
fig.canvas.draw()

with io.BytesIO() as buff:
    fig.savefig(buff, format='raw', pad_inches=0,dpi=100)
    buff.seek(0)
    data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
w, h = fig.canvas.get_width_height()
data2 = data.reshape((int(h), int(w), -1))
# data3 = data2.transpose((2, 0, 1))

data_holder = np.ndarray(shape=[200, 200])

for column in range(0, 200):
    for row in range(0, 200):
        colour = data2[column][row][0:3]
        name = ""
        category = 0

        if (colour == [255, 0, 0]).all():
            name = 'Haus'
            category = 1
        if (colour == [0, 0, 0]).all():
            name = 'Auto'
            category = 2
        if (colour == [128, 128, 128]).all():
            name = 'Straße'
            category = 3
        if (colour == [0, 128, 0]).all():
            name = 'Wiese'
            category = 4
        if (colour == [255, 255, 0]).all():
            name = 'Wald'
            category = 5
        if (colour == [0, 0, 255]).all():
            name = 'See'
            category = 6
        if (colour == [255, 0, 255]).all():
            name = 'Schienen'
            category = 7
        data_holder[column][row] = category

path_dat = r'C:\Users\Timo\Nextcloud\Freigaben\Projekt_Data_Science_1_SS22\Daten_Gyrocopter\Oldenburg\Teilbilder\grid_200_200\Teilbild_Oldenburg_00000000_00000000_0_0_.dat'
path_hdr = r'C:\Users\Timo\Nextcloud\Freigaben\Projekt_Data_Science_1_SS22\Daten_Gyrocopter\Oldenburg\Teilbilder\grid_200_200\Teilbild_Oldenburg_00000000_00000000_0_0_.hdr'
img = envi.open(file=path_hdr, image=path_dat)

new_bands = ['Haus', 'Wiese', 'See', 'Straße']
arr_metadata = img.metadata

for new_band in new_bands:
    arr_metadata['wavelength'].append(new_band)
    arr_metadata['band names'].append(new_band)
    arr_metadata['fwhm'].append(new_band)

arr_metadata['bands'] = int(arr_metadata['bands']) + len(new_bands)
path_dat_labbeled = r'C:\Users\Timo\Nextcloud\Freigaben\Projekt_Data_Science_1_SS22\Daten_Gyrocopter\Oldenburg\Teilbilder\grid_200_200\labeled\Teilbild_Oldenburg_00000000_00000000_0_0_.dat'
path_hdr_labbeled = r'C:\Users\Timo\Nextcloud\Freigaben\Projekt_Data_Science_1_SS22\Daten_Gyrocopter\Oldenburg\Teilbilder\grid_200_200\labeled\Teilbild_Oldenburg_00000000_00000000_0_0_.hdr'
img_labbeled = envi.create_image(hdr_file=path_hdr_labbeled, metadata=arr_metadata, dtype="float32", ext='.dat',
                                 interleave='bsq',
                                 force=True)

# Zugriff auf Bild Matrix erstellen
N = 200
writer = img_labbeled.open_memmap(writable=True)
writer[:, :, 0:109] = img.open_memmap(writable=False)

writer[:, :, 110] = data_holder
np.seterr(divide='ignore', invalid='ignore')
import spectral.io.envi as envi


path_dat_labbeled = r'C:\Users\Timo\Nextcloud\Freigaben\Projekt_Data_Science_1_SS22\Daten_Gyrocopter\Oldenburg\Teilbilder\grid_200_200\labeled\Teilbild_Oldenburg_00000000_00000000_0_0_.dat'
path_hdr_labbeled = r'C:\Users\Timo\Nextcloud\Freigaben\Projekt_Data_Science_1_SS22\Daten_Gyrocopter\Oldenburg\Teilbilder\grid_200_200\labeled\Teilbild_Oldenburg_00000000_00000000_0_0_.hdr'


img = envi.open(file=path_hdr_labbeled, image=path_dat_labbeled)
img
from spectral import *

img = open_image(file=path_hdr_labbeled)
img.__class__
img.shape

print(img)
pixel = img[0:200, 0:200]

pixel.shape

arr = img.load()
arr.__class__


arr.shape
plt.pause(0.05)
imshow(pixel, figsize=(30, 50),bands=[110])
plt.show()
view
