# Packages
from Code.functions.class_ids import map_float_id2rgb, map_int_id2name
from Code.functions.import_labeled_data import import_labeled_data
from Code.functions.combine_subimages import combine_subimages
from Code.find_path_nextcloud import find_path_nextcloud
import matplotlib.pyplot as plt
import spectral as spy
import pandas as pd
import numpy as np

# Import data

# Pfad Nextcloud bestimmen
path_nextcloud = find_path_nextcloud()

# Festlegung, welches Grid zusammengelegt werden soll
windowsize_r = 200
windowsize_c = 200

# Bestimmung annotation_folder
grid_folder = path_nextcloud + "Daten_Gyrocopter/Oldenburg/Teilbilder/grid_" + str(windowsize_r) + "_" + str(windowsize_c)
labeled_folder = grid_folder + '/labeled/'

df_annotations = import_labeled_data(path_labeled_folder=labeled_folder)

# Wie viele Pixel umfassen die annotiert Bilder insgesamt? 

print('Anzahl Pixel der annotierten Bilder\t', str(df_annotations.shape[0]))

# Wie viele Pixel wurden in den Bildern insgesamt annotiert? 

df_objects = df_annotations.loc[df_annotations['label'] != 0, ]
print('Anzahl annotierter Pixel\t\t\t', str(df_objects.shape[0]))

# Wie viel Prozent der Pixel enthalten eine Klasse ungleich None? 

print('Anteil annotierter Pixel\t\t\t', str((df_objects.shape[0] / df_annotations.shape[0]*100).__round__(1)), ' %')

# Wie viel Prozent der Pixel wurden pro Bild annotiert? 

w = df_objects[['picture_name']].groupby(['picture_name'], as_index = False).size()
w['percent'] = (w['size'] / 40000 * 100).__round__(1)
print(w)

# Wie sehen die Annotationen auf den Bildern aus? 

for image_name in df_annotations['picture_name'].unique():

    # select pixel for current image
    image = df_annotations.loc[df_annotations['picture_name'] == image_name, ]
    image = image.drop(columns=['picture_name'])

    # convert annotation ID to class_color
    df = map_float_id2rgb(dataframe=image, column='label')

    # extract color values
    df['class_color1'] = df['class_color'].apply(lambda x: x[0])
    df['class_color2'] = df['class_color'].apply(lambda x: x[1])
    df['class_color3'] = df['class_color'].apply(lambda x: x[2])

    # reshape pixel to image for rgb picture and select rgb channels
    img_rgb = np.reshape(np.array(image), (200, 200, 110))
    rgb_image = spy.get_rgb(img_rgb, bands=(59, 26, 1), stretch=(0.01, 0.99), stretch_all=True)

    # reshape pixel to image for annotation picture
    img_arr = np.array(df[['class_color1', 'class_color2', 'class_color3']])
    img_arr = np.reshape(img_arr, (200, 200, 3))

    # count pixel for each class
    count_class = image[['label']].groupby('label', as_index=False).size()
    count_class = map_int_id2name(dataframe=count_class, column='label')

    # create plot
    fig, ax = plt.subplots(nrows=1, ncols=3)
    fig.set_figwidth(15)
    ax[0].imshow(rgb_image)
    ax[0].set_title('RBG-Bild')
    ax[1].imshow(img_arr)
    ax[1].set_title('Annotation')
    label = ax[2].bar(count_class['class_name'], count_class['size'], label=count_class['size'])
    ax[2].bar_label(label, label_type='edge')
    ax[2].set_title('Klassenverteilung')
    plt.suptitle(image_name, fontsize=14)
    plt.show()
    fig.savefig('data/annotated_picture/' + image_name)

# Wie sieht die Verteilung der Anzahl der annotierten Pixel ??ber alle Klassen aus? 

# count pixel for each class
count_class = df_annotations[['label']].groupby('label', as_index=False).size()
count_class = map_int_id2name(dataframe=count_class, column='label')

# create plot
fig, ax = plt.subplots(nrows=1, ncols=1)
fig.set_figwidth(15)
label = ax.bar(count_class['class_name'], count_class['size'], label=count_class['size'])
ax.bar_label(label, label_type='edge')
plt.show()
fig.savefig('data/annotated_picture/Anzahl_Pixel_pro_Klasse.png')

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

# Zum Ausf??hren muss in image_functions.py in der funktion combine_subimages muss das
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
