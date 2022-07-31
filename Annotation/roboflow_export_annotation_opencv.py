import io

import matplotlib
import numpy as np
import xmltodict as xmltodict
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
print(matplotlib.get_backend())
matplotlib.use('Agg')

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
from matplotlib.axis import Axis

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
    patches.append(Polygon(coords, ec=colour, color=colour,zorder=Ebene,snap=True))


