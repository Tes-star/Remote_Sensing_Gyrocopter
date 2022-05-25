# Import Packages
import numpy as np
from spectral import envi
from image_functions import *
from find_path_nextcloud import find_path_nextcloud

print('Anfang')

# Bestimmung Pfad Nextcloud
path_nextcloud = find_path_nextcloud()

#######################################################################################################################
#######################################               Oldenburg                ########################################
#######################################################################################################################

# Bestimmung Ordner-Pfad
path_folder = path_nextcloud + 'Daten_Gyrocopter/Oldenburg'

# combine HSI, THERMAL and DOM image
path_out = combine_image_bands(path_data=path_folder,
                               hdr_file_hsi='Oldenburg_HSI.hdr', dat_file_hsi='Oldenburg_HSI.dat',
                               hdr_file_thermal='Oldenburg_THERMAL.hdr', dat_file_thermal='Oldenburg_THERMAL.dat',
                               hdr_file_dom='Oldenburg_DOM.hdr', dat_file_dom='Oldenburg_DOM.dat',
                               export_title='Oldenburg')

path_hdr = path_out
path_dat = path_out[:-4] + ".dat"

# Zusammengeführtes Bild öffnen
img_new = envi.open(file=path_hdr, image=path_dat)

# Ausgabe Info
print('Oldenburg:')
print(img_new)

#######################################################################################################################
#######################################                 Dessau                 ########################################
#######################################################################################################################


# Einlesen der Einzelnen Dateien

# Bestimmung Ordner-Pfad
path_folder = path_nextcloud + 'Daten_Gyrocopter/Dessau/'

## Hyperspektral
path_hdr = path_folder + 'Dessau_HSI.hdr'
path_dat = path_folder + 'Dessau_HSI.dat'

img_his = envi.open(file=path_hdr, image=path_dat)
print(img_his)

## Thermal
# gibt es von Dessau nicht

# Digitales Oberflächenmodell (DOM)
path_hdr = path_folder + 'Dessau_DOM.hdr'
path_dat = path_folder + 'Dessau_DOM.dat'

img_dom = envi.open(file=path_hdr, image=path_dat)
print(img_dom)

# Dessau Daten können aufgrund der Größe nicht zusammengeführt werden

print('\nEnde')
