from os import read

import xmltodict as xmltodict
from spectral.io import envi
import numpy as np
from Daten_einlesen.find_path_nextcloud import find_path_nextcloud
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from spectral import *
import io
from Daten_einlesen.find_path_nextcloud import find_path_nextcloud

from xml.dom import minidom
# Import Packages
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
for object in doc['annotation']['object']:


    plt.xlim(0, 200)
    plt.ylim(0, 200)
    name = object['name']
    polygon = object['polygon']
    patches = []

    coords = []
    for i in range(1, int(len(polygon) / 2)):
        point = [polygon['x' + str(i)], polygon['y' + str(i)]]
        coords.append(point)
    patches.append(Polygon(coords, True))
    p = PatchCollection(patches)

    ax.add_collection(p)
    fig.set_size_inches(20 / 7.2, 20 / 7.2)
    fig.canvas.draw()

    plt.axis('off')
    fig = ax.figure
    fig.tight_layout(pad=0)
fig.canvas.draw()