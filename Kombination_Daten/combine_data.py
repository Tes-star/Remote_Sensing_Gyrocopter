# Import Packages
import os
import numpy as np
from spectral import envi
from Daten_einlesen.find_path_nextcloud import find_path_nextcloud

print('Anfang')

# Bestimmung Pfad Nextcloud
path_nextcloud = find_path_nextcloud()


#######################################################################################################################
#######################################               Oldenburg                ########################################
#######################################################################################################################

# Einlesen der Einzelnen Dateien

# Bestimmung Ordner-Pfad
path_folder = path_nextcloud + 'Daten_Gyrocopter/Oldenburg/'

## Hyperspektral
path_hdr = path_folder + 'Oldenburg_HSI.hdr'
path_dat = path_folder + 'Oldenburg_HSI.dat'

img_his = envi.open(file=path_hdr, image=path_dat)


## Thermal
path_hdr = path_folder + 'Oldenburg_THERMAL.hdr'
path_dat = path_folder + 'Oldenburg_THERMAL.dat'

img_thermal = envi.open(file=path_hdr, image=path_dat)

# Digitales Oberflächenmodell (DOM)
path_hdr = path_folder + 'Oldenburg_DOM.hdr'
path_dat = path_folder + 'Oldenburg_DOM.dat'

img_dom = envi.open(file=path_hdr, image=path_dat)


# Laden der Bildinformationen
arr_his = img_his.load()
arr_thermal = img_thermal.load()
arr_dom = img_dom.load()


# Zusammenführen Bildinformationen
combined_arr = np.concatenate((arr_his, arr_thermal),-1)
combined_arr = np.concatenate((combined_arr, arr_dom),-1)

new_bands = ['thermal', 'dom']
arr_metadata = img_his.metadata

for new_band in new_bands:
    arr_metadata['wavelength'].append(new_band)
    arr_metadata['band names'].append(new_band)
    arr_metadata['fwhm'].append(new_band)

arr_metadata['bands'] = len(arr_metadata['wavelength'])

# Zusammengeführtes Bild speichern

path_out = path_folder + "Oldenburg_combined_HSI_THERMAL_DOM.hdr"

envi.save_image(hdr_file=path_out, image=combined_arr,
                dtype = "float32", ext='.dat', interleave = 'bsq',
                metadata=arr_metadata, force = True)

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

# Laden der Bildinformationen
arr_his = img_his.load()
arr_dom = img_dom.load()


# Zusammenführen Bildinformationen
combined_arr = np.concatenate((arr_his, arr_dom),-1)

new_bands = ['dom']
arr_metadata = img_his.metadata

for new_band in new_bands:
    arr_metadata['wavelength'].append(new_band)
    arr_metadata['band names'].append(new_band)
    arr_metadata['fwhm'].append(new_band)

arr_metadata['bands'] = len(arr_metadata['wavelength'])

# Zusammengeführtes Bild speichern

path_out = path_folder + "Oldenburg_combined_HSI_DOM.hdr"

envi.save_image(hdr_file=path_out, image=combined_arr,
                dtype = "float32", ext='.dat', interleave = 'bsq',
                metadata=arr_metadata, force = True)

path_hdr = path_out
path_dat = path_out[:-4] + ".dat"

# Zusammengeführtes Bild öffnen
img_new = envi.open(file=path_hdr, image=path_dat)

# Ausgabe Info
print('Dessau:')
print(img_new)

print('\nEnde')
