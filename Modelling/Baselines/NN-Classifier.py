import numpy as np
import pandas as pd
import tensorflow
from Code.image_functions import *
from Code.find_path_nextcloud import find_path_nextcloud
import wandb
from sklearn.model_selection import train_test_split
from wandb.keras import WandbCallback
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

# login
wandb.init(project="NN_for_pixels", entity="pds_project", name='Test_CPU')

# import data
path_nextcloud = find_path_nextcloud()
path_labeled_folder = path_nextcloud + "Daten_Gyrocopter/Oldenburg/Teilbilder/grid_200_200/labeled/"
df_annotations = import_labeled_data(path_labeled_folder=path_labeled_folder)

# cut df_annotations in x and y
X = df_annotations.drop(columns=['label', 'picture_name'])
label = np.array(df_annotations['label'])

# build for each class one column (0 and 1)
y = pd.DataFrame()
dic = {0:'None',1:'Wiese', 2:'Straße',3: 'Auto', 4:'See', 5:'Schienen', 6:'Haus', 7:'Wald'}
for key, value in dic.items():
    y[value] = np.where(label == key, 1, 0)
y = np.array(y)


# split in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_train = tensorflow.convert_to_tensor(X_train, dtype=tensorflow.float32)
y_train = tensorflow.convert_to_tensor(y_train, dtype=tensorflow.float32)
X_test = tensorflow.convert_to_tensor(X_test, dtype=tensorflow.float32)
y_test = tensorflow.convert_to_tensor(y_test, dtype=tensorflow.float32)

# define the keras model
model = Sequential()
model.add(Dense(30, input_dim=109,  activation='relu'))
model.add(Dense(15, input_dim=109,  activation='relu'))
model.add(Dense(8, activation='sigmoid'))

# compile the keras model
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy', keras.metrics.MeanSquaredError()])
# model.fit(X_train, y_train, epochs=50)
model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test), callbacks=[WandbCallback()])
