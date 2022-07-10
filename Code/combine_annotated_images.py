import matplotlib.pyplot as plt

from find_path_nextcloud import find_path_nextcloud
from image_functions import *
import spectral as spy

# find project path in nextcloud
path_nextcloud = find_path_nextcloud()

# define path with data
path_folder = path_nextcloud + 'Daten_Gyrocopter/Oldenburg'

# define HSI filenames
path_combined_hdr = path_folder + '/Oldenburg_combined_HSI_THERMAL_DOM.hdr'
path_combined_dat = path_folder + '/Oldenburg_combined_HSI_THERMAL_DOM.dat'

# define path
path_grid = path_folder + '/Teilbilder/grid_200_200/combine_annotations'
path_export = path_folder + '/Teilbilder/grid_200_200/combine_annotations_big_picture'

# Zum Ausführen muss in image_functions.py in der funktion combine_subimages muss das
# label-band bei nicht annotierten Bilder auskommentiert werden

big_picture = combine_subimages(hdr_file=path_combined_hdr, dat_file=path_combined_dat,
                                path_grid_subimages=path_grid,
                                path_export=path_export, window_width=200, window_height=200)


arr = big_picture.load()

fig, ax = plt.subplots(figsize=(50, 30))
ax.imshow(arr[:, :, -1])
fig.show()
fig.savefig('big_picture_annotations', dpi=300)

print('Fertig')
