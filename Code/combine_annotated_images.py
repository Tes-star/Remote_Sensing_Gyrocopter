# Packages
from find_path_nextcloud import find_path_nextcloud
from image_functions import *
from Code.functions.class_ids import map_float_id2rgb
import matplotlib.pyplot as plt

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
                                path_export=path_export, window_width=200, window_height=200,
                                combine_annotated_images=True)


arr = big_picture.load()

# Umwandlung in DataFrame um class_id auf class_rgb zu mappen
df = pd.DataFrame(arr.reshape(6930000,110))
df = map_float_id2rgb(dataframe=df, column=109)

# extract color values
df['class_color1'] = df['class_color'].apply(lambda x: x[0])
df['class_color2'] = df['class_color'].apply(lambda x: x[1])
df['class_color3'] = df['class_color'].apply(lambda x: x[2])

# reshape pixel to image for annotation picture
img_arr = np.array(df[['class_color1', 'class_color2', 'class_color3']])
img_arr = np.reshape(img_arr, (1980, 3500, 3))

fig, ax = plt.subplots(figsize=(50, 30))
ax.imshow(img_arr)
fig.show()
fig.savefig('data/annotated_picture/big_picture_annotations', dpi=300)

print('Fertig')
