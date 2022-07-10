import tensorflow
from Code.image_functions import *
from Code.find_path_nextcloud import find_path_nextcloud
import keras
import numpy as np
import matplotlib.pyplot as plt

# load model
model = keras.models.load_model('baseline1.h5')

# load data
path_nextcloud = find_path_nextcloud()
path_labeled_folder = path_nextcloud + "Daten_Gyrocopter/Oldenburg/Teilbilder/grid_200_200/labeled/"
df_annotations = import_labeled_data(path_labeled_folder)

# convert input
X = df_annotations.drop(columns=['label', 'picture_name'])
X = tensorflow.convert_to_tensor(X, dtype=tensorflow.float32)

# prediction
y_pred = model.predict(X)

# choose argmax
df_annotations['y_pred'] = y_pred.argmax(axis=1)

# export images
for image_name in df_annotations['picture_name'].unique():
    image = df_annotations.loc[df_annotations['picture_name'] == image_name, ]
    image = image.drop(columns=['picture_name'])
    img_arr = np.array(image, dtype=float)
    img_arr = np.reshape(img_arr, (200, 200, 111))
    plt.imshow(img_arr[:, :, -1])
    plt.show()



