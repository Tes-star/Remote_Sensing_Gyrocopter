# Import Packages
import os
from spectral import *
from Daten_einlesen.find_path_nextcloud import find_path_nextcloud

# Pfad Nextcloud bestimmen
path_nextcloud = find_path_nextcloud()

# Pfad + Dateiname der einzulesenden Daten
path_hdr = path_nextcloud + "Daten_Gyrocopter/Oldenburg/Oldenburg_combined_HSI_THERMAL_DOM.hdr"
path_dat = path_hdr[:-4] + '.dat'

# Zusammengeführtes Bild öffnen
img = envi.open(file=path_hdr, image=path_dat)
print(img)

# Gridgröße definieren
windowsize_r = 250
windowsize_c = 250

# Festlegung grid Ordnername
grid_folder = path_nextcloud + "Daten_Gyrocopter/Oldenburg/Teilbilder/grid_" + str(windowsize_r) + "_" + str(
    windowsize_c)

# Grid Ordner ggf. erzeugen
if not os.path.exists(grid_folder):
    os.makedirs(grid_folder)


# Image in Teilbilder aufteilen

grid_r = 0  # gridrow
grid_c = 0  # gridcolumn

# Weise stop_after_row ein Zahl größer 0 zu, wenn nicht alle Teilbilder erzeugt werden sollen
# Wenn alle Bilder erzeugt werden sollen, muss stop_after_row == -1 sein
stop_row = 1

# im Intervall windowsize_r über Bildmatrix Reihen iterieren
for r in range(0, img.shape[0], windowsize_r):

    # im Intervall windowsize_c über Bildmatrix Spalten iterieren
    for c in range(0, img.shape[1], windowsize_c):

        # Teilbild auswählen
        window = img[r:r + windowsize_r, c:c + windowsize_c, :]
        print(str(r), '/', str(c))

        # Namenskonvention
        path_window = grid_folder + "/Teilbild_Oldenburg_" + str(grid_r).zfill(8) + "_" + str(grid_c).zfill(
            8) + "_" + str(r) + "_" + str(c) + "_.hdr"

        envi.save_image(hdr_file=path_window, image=window,
                        dtype="float32", ext='.dat', interleave='bsq',
                        metadata=img.metadata, force=True)
        # next column
        grid_c = grid_c + 1

    # next row
    grid_r = grid_r + 1
    # reset grid_c
    grid_c = 0

    # Abbruchbedingung
    if grid_r == stop_row:
        break


print("Fertig!")
