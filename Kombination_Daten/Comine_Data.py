
# Import Packages
import os
import numpy as np
from spectral import envi


# Functions
from Daten_einlesen.find_path_nextcloud import find_path_nextcloud

print('Anfang')

# Bestimmung Pfad Nextcloud
path_nextcloud = find_path_nextcloud()


#######################################################################################################################
#######################################               Oldenburg                ########################################
#######################################################################################################################

# Einlesen der Einzelnen Dateien

## Hyperspektral
path_folder = path_nextcloud + 'Daten_Gyrocopter/Oldenburg/Hyperspektral/'
path_hdr = path_folder + 'Oldenburg-all.hdr'
path_dat = path_folder + 'Oldenburg-all.dat'

img_his = envi.open(file=path_hdr, image=path_dat)


## Thermal
path_folder = path_nextcloud + 'Daten_Gyrocopter/Oldenburg/Thermal/'
path_hdr = path_folder + 'Thermal-Resized.hdr'
path_dat = path_folder + 'Thermal-Resized'

img_thermal = envi.open(file=path_hdr, image=path_dat)

# Digitales Oberflächenmodell (DOM)
path_folder = path_nextcloud + 'Daten_Gyrocopter/Oldenburg/DigitalesOberflaechenModel_normiert/'
path_hdr = path_folder + 'DGM-DTM-Resized.hdr'
path_dat =  path_folder + 'DGM-DTM-Resized'

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

path_out = path_nextcloud + "Daten_Gyrocopter/Zusammengefuehrte_Daten/Oldenburg_combined.hdr"

envi.save_image(hdr_file=path_out, image=combined_arr,
                dtype = "float32", ext='.dat', interleave = 'bsq',
                metadata=arr_metadata, force = True)

path_hdr = path_out
path_dat = path_nextcloud + "Daten_Gyrocopter/Zusammengefuehrte_Daten/Oldenburg_combined.dat"

# Zusammengeführtes Bild öffnen
img_new = envi.open(file=path_hdr, image=path_dat)

# Ausgabe Info
print('Oldenburg:')
print(img_new)


#######################################################################################################################
#######################################                 Dessau                 ########################################
#######################################################################################################################
#
# # Einlesen der Einzelnen Dateien
#
# ## Hyperspektral
# path_folder = path_nextcloud + 'Daten_Gyrocopter/Dessau/Hyperspektral/'
# path_hdr = path_folder + 'Dessau_unsigned_mosaik_all.hdr'
# path_dat = path_folder + 'Dessau_unsigned_mosaik_all.dat'
#
# img_his = envi.open(file=path_hdr, image=path_dat)
#
#
# ## Thermal
# path_folder = path_nextcloud + 'Daten_Gyrocopter/Oldenburg/Thermal/'
# path_hdr = path_folder + 'Thermal-Resized.hdr'
# path_dat = path_folder + 'Thermal-Resized'
#
# img_thermal = envi.open(file=path_hdr, image=path_dat)
#
# # Digitales Oberflächenmodell (DOM)
# path_folder = path_nextcloud + 'Daten_Gyrocopter/Oldenburg/DigitalesOberflaechenModel_normiert/'
# path_hdr = path_folder + 'DGM-DTM-Resized.hdr'
# path_dat =  path_folder + 'DGM-DTM-Resized'
#
# img_dom = envi.open(file=path_hdr, image=path_dat)
#
#
# # Laden der Bildinformationen
# arr_his = img_his.load()
# arr_thermal = img_thermal.load()
# arr_dom = img_dom.load()
#
#
# # Zusammenführen Bildinformationen
# combined_arr = np.concatenate((arr_his, arr_thermal),-1)
# combined_arr = np.concatenate((combined_arr, arr_dom),-1)
#
# new_bands = ['thermal', 'dom']
# arr_metadata = img_his.metadata
#
# for new_band in new_bands:
#     arr_metadata['wavelength'].append(new_band)
#     arr_metadata['band names'].append(new_band)
#     arr_metadata['fwhm'].append(new_band)
#
# arr_metadata['bands'] = len(arr_metadata['wavelength'])
#
# # Zusammengeführtes Bild speichern
#
# path_out = path_nextcloud + "Daten_Gyrocopter/Zusammengefuehrte_Daten/Oldenburg_combined.hdr"
#
# envi.save_image(hdr_file=path_out, image=combined_arr,
#                 dtype = "float32", ext='.dat', interleave = 'bsq',
#                 metadata=arr_metadata, force = True)
#
# path_hdr = path_out
# path_dat = path_nextcloud + "Daten_Gyrocopter/Zusammengefuehrte_Daten/Oldenburg_combined.dat"
#
# # Zusammengeführtes Bild öffnen
# img_new = envi.open(file=path_hdr, image=path_dat)
#
# # Ausgabe Info
# print('Dessau:')
# print(img_new)

print('\nEnde')
