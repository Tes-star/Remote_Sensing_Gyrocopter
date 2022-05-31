from find_path_nextcloud import find_path_nextcloud
from image_functions import *
import spectral as spy

# find project path in nextcloud
path_nextcloud = find_path_nextcloud()

# define path with data
path_folder = 'C:/Users/fgrassxx/Desktop/Oldenburg'

# define HSI filenames
path_combined_hdr = path_folder + '/Oldenburg_combined_HSI_THERMAL_DOM.hdr'
path_combined_dat = path_folder + '/Oldenburg_combined_HSI_THERMAL_DOM.dat'

# split image in subimages
path_grid_folder = split_image(hdr_file=path_combined_hdr, dat_file=path_combined_dat, window_width=200,
                               window_height=200, export_path='C:/Users/fgrassxx/Desktop/Oldenburg', export_title='Oldenburg')

# save subimages as rgb
save_subimages_rgb(path_grid_subimages=path_grid_folder, rgb_band=(59, 26, 1))

# # export annotated polygon as mask
# convert_xml_annotation_to_mask(path_data='../data/Oldenburg_grid_200_200',
#                                xml_file='Teilbild_Oldenburg_Annotation.xml')

print('Fertig')
