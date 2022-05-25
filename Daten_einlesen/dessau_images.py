from spectral import *
from spectral import envi
from find_path import find_path_nextcloud


print('Anfang')

# Bestimmung Pfad Nextcloud
path_nextcloud = find_path_nextcloud()

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
# arr_his = img_his.load()
arr_dom = img_dom.load()

view = imshow(arr_dom, figsize=(30,50))


