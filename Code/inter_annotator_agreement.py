from Code.functions.convert_annotations import convert_xml_annotation_to_mask
from Code.functions.class_ids import map_float_id2rgb, get_class_list
from Code.functions.import_labeled_data import import_labeled_data
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import tensorflow
import spectral as spy
import pandas as pd
import numpy as np
import keras

# define parameter
windowsize_r = 200
windowsize_c = 200
path_pictures = '../data/inter_annotator_agreement/unlabeled'
path_labeled = '../data/inter_annotator_agreement/labeled'


# convert Felix annotation
# path_xml_file = '../data/inter_annotator_agreement/Export_roboflow/Teilbild_Oldenburg_00000006_00000011_1200_2200_Felix.xml'
#
# convert_xml_annotation_to_mask(xml_file=path_xml_file,
#                                path_picture=path_pictures,
#                                path_export=path_labeled,
#                                windowsize_r=windowsize_r,
#                                windowsize_c=windowsize_c)

# convert Timo annotation
# path_xml_file = '../data/inter_annotator_agreement/Export_roboflow/Teilbild_Oldenburg_00000006_00000011_1200_2200_Timo.xml'

# convert_xml_annotation_to_mask(xml_file=path_xml_file,
#                               path_picture=path_pictures,
#                               path_export=path_labeled,
#                               windowsize_r=windowsize_r,
#                               windowsize_c=windowsize_c)

# compare annotations


def create_files(filename):
    path_dat = filename + '.dat'
    path_hdr = filename + '.hdr'
    img = spy.envi.open(file=path_hdr, image=path_dat)

    img_arr = img.load()

    img_df = pd.DataFrame(img_arr.reshape(((img_arr.shape[0] * img_arr.shape[1]), img_arr.shape[2])))

    # convert annotation ID to class_color
    img_df = map_float_id2rgb(dataframe=img_df, column=109)

    # extract color values
    img_df['class_color1'] = img_df['class_color'].apply(lambda x: x[0])
    img_df['class_color2'] = img_df['class_color'].apply(lambda x: x[1])
    img_df['class_color3'] = img_df['class_color'].apply(lambda x: x[2])

    # reshape pixel to image for rgb picture and select rgb channels
    rgb_arr = np.reshape(np.array(img_df), (200, 200, 114))
    rgb_arr = spy.get_rgb(rgb_arr, bands=(59, 26, 1), stretch=(0.03, 0.97), stretch_all=True)

    # reshape pixel to image for annotation picture
    annot_arr = np.array(img_df[['class_color1', 'class_color2', 'class_color3']])
    annot_arr = np.reshape(annot_arr, (200, 200, 3))

    return img_df, annot_arr, rgb_arr


# define filenames
filename_fg = '../data/inter_annotator_agreement/labeled/Teilbild_Oldenburg_00000006_00000011_1200_2200_Felix'
filename_tvw = '../data/inter_annotator_agreement/labeled/Teilbild_Oldenburg_00000006_00000011_1200_2200_Timo'

# create images
img_df_fg, annot_arr_fg, rgb_arr = create_files(filename=filename_fg)
img_df_tvw, annot_arr_tvw, rgb_arr = create_files(filename=filename_tvw)

# show rgb regions with different annotations
test = annot_arr_fg.copy()
test[:50, :50, :] = 0

mask = annot_arr_fg != annot_arr_tvw
mask = mask * 1
diff_annot = rgb_arr * mask

# compare plot
fig, ax = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(10)
ax[0, 0].imshow(annot_arr_fg)
ax[0, 0].set_title('Annotation Felix')
ax[0, 1].imshow(annot_arr_tvw)
ax[0, 1].set_title('Annotation Timo')
ax[1, 0].imshow(rgb_arr)
ax[1, 0].set_title('RGB Bild')
ax[1, 1].imshow(diff_annot)
ax[1, 1].set_title('Unterschiede in der Annotation')
plt.suptitle('Vergleich Annotationen', fontsize=14)
plt.show()
fig.savefig('../data/inter_annotator_agreement/Vergleich_Annotation.png')

# Vertauschungsmatrix
cm = confusion_matrix(y_true=img_df_fg[109], y_pred=img_df_tvw[109],
                      labels=[0, 1, 2, 3, 4, 5, 6, 7], normalize='true')
cmp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=get_class_list())
fig, ax = plt.subplots(figsize=(10, 10))
cmp.plot(ax=ax, cmap='Blues')
ax.set_ylabel('Annotation Felix')
ax.set_xlabel('Annotation Timo')
plt.title('Vertauschungsmatrix Annotation Felix vs. Timo')
plt.show()
fig.savefig('../data/inter_annotator_agreement/Vertauschungsmatrix_Annotation.png')

# calculate cohen_kappa_score
iia_score = cohen_kappa_score(y1=img_df_fg[109], y2=img_df_tvw[109])
print('cohen_kappa_score', str(iia_score))  # 0.66 -> substantial agreement

# calculate accuracy_score
acc = accuracy_score(y_true=img_df_fg[109], y_pred=img_df_tvw[109])
print('accuracy_score', str(acc))

# create prediction for annotated subimage

# load models
nn_for_pixel = keras.models.load_model('../data/models/baseline3_nn_for_pixel.h5')
# cnn = keras.models.load_model('../data/models/CNN_1.h5')

# import data nn_for_pixel
df_subimage = import_labeled_data(path_labeled_folder='../data/inter_annotator_agreement/labeled')
X_nn_for_pixel = df_subimage.drop(columns=['picture_name', 'label'])
X_nn_for_pixel = tensorflow.convert_to_tensor(X_nn_for_pixel, dtype=tensorflow.float32)

# prediction  nn_for_pixel
y_pred_nn_for_pixel = nn_for_pixel.predict(X_nn_for_pixel)
df_subimage['y_pred'] = y_pred_nn_for_pixel.argmax(axis=1)

# split in fg und tvw DataFrame subimage
df_subimage_fg = df_subimage.loc[df_subimage['picture_name'] == 'Teilbild_Oldenburg_00000006_00000011_1200_2200_Felix',]
df_subimage_tvw = df_subimage.loc[df_subimage['picture_name'] == 'Teilbild_Oldenburg_00000006_00000011_1200_2200_Timo',]

# compare scores

# complete annotation felix
cm = confusion_matrix(y_true=df_subimage_fg['label'], y_pred=df_subimage_fg['y_pred'],
                      labels=[0, 1, 2, 3, 4, 5, 6, 7], normalize='true')
cmp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=get_class_list())
fig, ax = plt.subplots(figsize=(10, 10))
cmp.plot(ax=ax, cmap='Blues')
plt.title('confusion matrix Felix annotation')
plt.show()

# complete annotation timo
cm = confusion_matrix(y_true=df_subimage_tvw['label'], y_pred=df_subimage_tvw['y_pred'],
                      labels=[0, 1, 2, 3, 4, 5, 6, 7], normalize='true')
cmp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=get_class_list())
fig, ax = plt.subplots(figsize=(10, 10))
cmp.plot(ax=ax, cmap='Blues')
plt.title('confusion matrix Timo annotation')
plt.show()

# equal annotation felix and timo
label_fg = df_subimage_fg['label'].values
label_tvw = df_subimage_tvw['label'].values
df_subimage_equal = df_subimage_fg.loc[label_fg==label_tvw, ]
cm = confusion_matrix(y_true=df_subimage_equal['label'], y_pred=df_subimage_equal['y_pred'],
                      labels=[0, 1, 2, 3, 4, 5, 6, 7], normalize='true')
cmp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=get_class_list())
fig, ax = plt.subplots(figsize=(10, 10))
cmp.plot(ax=ax, cmap='Blues')
plt.title('confusion matrix equal annotation')
plt.show()
