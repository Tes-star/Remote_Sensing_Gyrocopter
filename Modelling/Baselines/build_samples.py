import numpy as np
import pandas as pd
import tensorflow
from Code.image_functions import *
from Code.find_path_nextcloud import find_path_nextcloud
from Modelling.Data_Preprocessing import import_labeled_photos
from sklearn.model_selection import StratifiedKFold


def import_samples_for_baseline(label_mapping=None):
    # import data
    global y_train, y_test, X_train, X_test
    path_nextcloud = find_path_nextcloud()
    path_labeled_folder = path_nextcloud + "Daten_Gyrocopter/Oldenburg/Teilbilder/grid_200_200/labeled/"
    df_annotations = import_labeled_data(path_labeled_folder=path_labeled_folder)

    match label_mapping:
        case None:
            label_mapping = [0, 1, 2, 3, 4, 5, 6, 7]
        case 'Ohne_Auto_See':
            label_mapping = [0, 1, 2, 0, 0, 3, 4, 5]
            # Auto=0
            # See=0
        case 'Grünflächen':
            label_mapping = [0, 1, 2, 0, 0, 3, 4, 1]
            # Auto=0
            # See=0
            # Wald=Wiese

    df_annotations['label'] = df_annotations['label'].replace([0, 1, 2, 3, 4, 5, 6, 7], label_mapping)

    for train_index, test_index in StratifiedKFold(random_state=0, shuffle=True, n_splits=5).split(df_annotations, df_annotations['label']):
        # cut df_annotations in x and y
        train = df_annotations.iloc[train_index]
        X_train = train.drop(columns=['label', 'picture_name'])
        y_train = train['label']

        test = df_annotations.iloc[test_index]
        X_test = test.drop(columns=['label', 'picture_name'])
        y_test = test['label']

    # build for each class one column (0 and 1)
    y = pd.DataFrame()
    for value in df_annotations['label'].drop_duplicates():
        y['id_' + str(value)] = np.where(y_train == value, 1, 0)
    y_train = np.array(y)

    y = pd.DataFrame()
    for value in df_annotations['label'].unique():
        y['id_' + str(value)] = np.where(y_test == value, 1, 0)
    y_test = np.array(y)

    # convert to tensor
    X_train = tensorflow.convert_to_tensor(X_train, dtype=tensorflow.float32)
    y_train = tensorflow.convert_to_tensor(y_train, dtype=tensorflow.int8)
    X_test = tensorflow.convert_to_tensor(X_test, dtype=tensorflow.float32)
    y_test = tensorflow.convert_to_tensor(y_test, dtype=tensorflow.int8)

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = import_samples_for_baseline(label_mapping='Ohne_Auto_See')
    print('Fertig')
