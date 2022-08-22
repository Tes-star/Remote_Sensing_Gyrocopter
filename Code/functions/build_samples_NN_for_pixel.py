import numpy as np
import pandas as pd
import tensorflow
from Code.image_functions import *
from Code.find_path_nextcloud import find_path_nextcloud
from Code.functions.class_ids import new_label_mapping
from Modelling.Data_Preprocessing import import_labeled_photos
from sklearn.model_selection import StratifiedKFold
from keras.utils.np_utils import to_categorical


def import_samples_for_baseline(label_mapping=None):
    # import data
    path_nextcloud = find_path_nextcloud()
    path_labeled_folder = path_nextcloud + "Daten_Gyrocopter/Oldenburg/Teilbilder/grid_200_200/labeled/"
    df_annotations = import_labeled_data(path_labeled_folder=path_labeled_folder)

    new_label_mapping(datafrane=df_annotations, map_column='label', label_mapping=label_mapping)

    for train_index, test_index in StratifiedKFold(random_state=0, shuffle=True, n_splits=5).split(df_annotations, df_annotations['label']):

        # cut df_annotations in x and y
        train = df_annotations.iloc[train_index]
        X_train = train.drop(columns=['label', 'picture_name'])
        y_train = train['label']

        test = df_annotations.iloc[test_index]
        X_test = test.drop(columns=['label', 'picture_name'])
        y_test = test['label']

        break

    y_train = to_categorical(y_train, num_classes=8)
    y_test = to_categorical(y_test, num_classes=8)

    # convert to tensor
    X_train = tensorflow.convert_to_tensor(X_train, dtype=tensorflow.float32)
    y_train = tensorflow.convert_to_tensor(y_train, dtype=tensorflow.int8)
    X_test = tensorflow.convert_to_tensor(X_test, dtype=tensorflow.float32)
    y_test = tensorflow.convert_to_tensor(y_test, dtype=tensorflow.int8)

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = import_samples_for_baseline(label_mapping='Ohne_Auto_See')
    print('Fertig')
