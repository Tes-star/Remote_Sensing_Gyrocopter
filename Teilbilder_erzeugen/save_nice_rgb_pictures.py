# Import Packages
import os
from spectral import *
from Daten_einlesen.find_path_nextcloud import find_path_nextcloud

# Pfad Nextcloud bestimmen
path_nextcloud = find_path_nextcloud()

# Festlegung, welches Grid zusammengelegt werden soll
windowsize_r = 250
windowsize_c = 250

# Bestimmung grid Ordnername
grid_folder = path_nextcloud + "Daten_Gyrocopter/Oldenburg/Teilbilder/grid_" + str(windowsize_r) + "_" + str(
    windowsize_c)


# Auswahl .dat Teilbilder Dateien

# Liste aller Dateien in Gridordner erstellen
files = os.listdir(grid_folder)

# Aus Liste files .hdr Dateien löschen
for file in files:
    if not file.endswith('.hdr'):
        files.remove(file)

# zusammengeführte grid Dateien aus Liste entfernen
for file in files:
    if 'grid' in file:
        files.remove(file)

# rgb Ordner aus Liste entfernen
for file in files:
    if 'rgb' in file:
        files.remove(file)

# rgb Ordner erstellen
path_grid_rgb = grid_folder + '/rgb_color_test'
if not os.path.exists(path_grid_rgb):
    os.makedirs(path_grid_rgb)


# RGB Bilder erstellen

for value in [(0.01, 0.99), (0.02, 0.98), (0.03, 0.97), (0.03, 0.97), (0.04, 0.96), (0.05, 0.95), (0.06, 0.94),
              (0.07, 0.93), (0.08, 0.92), (0.09, 0.91), (0.10, 0.90)]:
    for file in files:
        # Daten für Namenskonvention
        file_split = file.split('_')
        grid_pos_r = int(file_split[2]) * windowsize_r  # Gridposition row
        grid_pos_c = int(file_split[3]) * windowsize_c  # Gridposition column
        # Namenskonvention durchführen
        rgb_name = path_grid_rgb + "/Teilbild_Oldenburg_" + str(file_split[2]).zfill(8) + "_" + str(file_split[3]).zfill(
            8) + "_" + str(grid_pos_r) + "_" + str(grid_pos_c) + '_' + str(value) + "_.jpg"

        # Pfade definieren
        path_hdr = grid_folder + '/' + file[:-4] + '.hdr'
        path_dat = grid_folder + '/' + file[:-4] + '.dat'

        # Teilbilder laden
        img = envi.open(file=path_hdr, image=path_dat)

        # RGB für Teilbilder erstellen
        save_rgb(filename=rgb_name, data=img, bands=(59, 26, 1), stretch=value, stretch_all = True)
