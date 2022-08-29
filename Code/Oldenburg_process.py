from find_path_nextcloud import find_path_nextcloud
from Code.functions.save_subimages_rgb import save_subimages_rgb
from Code.functions.combine_image_bands import combine_image_bands
from Code.functions.convert_annotations import convert_all_annotations

# find project path in nextcloud
path_nextcloud = find_path_nextcloud()

# define path with data
path_folder = path_nextcloud + 'Daten_Gyrocopter/Oldenburg'

# combine HSI, THERMAL and DOM image
# path_out = combine_image_bands(path_data=path_folder,
#                                hdr_file_hsi='Oldenburg_HSI.hdr', dat_file_hsi='Oldenburg_HSI.dat',
#                                hdr_file_thermal='Oldenburg_THERMAL.hdr', dat_file_thermal='Oldenburg_THERMAL.dat',
#                                hdr_file_dom='Oldenburg_DOM.hdr', dat_file_dom='Oldenburg_DOM.dat',
#                                export_title='Oldenburg')

# define HSI filenames
path_combined_hdr = path_folder + '/Oldenburg_combined_HSI_THERMAL_DOM.hdr'
path_combined_dat = path_folder + '/Oldenburg_combined_HSI_THERMAL_DOM.dat'

# split image in subimages
# path_grid_folder = split_image(hdr_file=path_combined_hdr, dat_file=path_combined_dat, window_width=200,
#                                window_height=200, export_path='C:/Users/fgrassxx/Desktop/Oldenburg', export_title='Oldenburg')

# save subimages as rgb
# save_subimages_rgb(path_grid_subimages=path_grid_folder, rgb_band=(59, 26, 1))

# define paths
path_grid_folder = path_nextcloud + 'Daten_Gyrocopter/Oldenburg/Teilbilder/grid_200_200'
path_roboflow_annotations = path_grid_folder + '/Export_roboflow'
path_subimages = path_grid_folder + '/subimages'
path_export_labeled = path_grid_folder + '/labeled'


# export annotated polygon as mask
convert_all_annotations(path_annotations=path_roboflow_annotations,
                        path_pictures=path_subimages,
                        path_export=path_export_labeled,
                        windowsize_c=200,
                        windowsize_r=200)

print('Fertig')
