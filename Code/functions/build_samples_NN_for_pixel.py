from Code.functions.train_test_images import train_test_image_names
from Code.functions.import_labeled_data import import_labeled_data
from Code.find_path_nextcloud import find_path_nextcloud
from Code.functions.class_ids import new_label_mapping
from keras.utils.np_utils import to_categorical
import pandas as pd
import tensorflow


def import_samples_NN_for_pixel(label_mapping=None):
    """
    functions which import train and test data with x and y for model NN_for_pixel
    :param label_mapping: replace specific class ids to other class ids
    :return:  tensor X_train, y_train, X_test, y_test
    """

    # import data
    path_nextcloud = find_path_nextcloud()
    path_labeled_folder = path_nextcloud + "Daten_Gyrocopter/Oldenburg/Teilbilder/grid_200_200/labeled/"
    df_annotations = import_labeled_data(path_labeled_folder=path_labeled_folder)

    df_annotations = new_label_mapping(dataframe=df_annotations, map_column='label', label_mapping=label_mapping)

    # import train and test image names
    df_train, df_test = train_test_image_names()

    # join
    train = pd.merge(left=df_train, right=df_annotations, how='left', on='picture_name')
    test = pd.merge(left=df_test, right=df_annotations, how='left', on='picture_name')

    X_train = train.drop(columns=['label', 'picture_name'])
    y_train = train['label']

    X_test = test.drop(columns=['label', 'picture_name'])
    y_test = test['label']

    y_train = to_categorical(y_train, num_classes=8)
    y_test = to_categorical(y_test, num_classes=8)

    # convert to tensor
    X_train = tensorflow.convert_to_tensor(X_train, dtype=tensorflow.float32)
    y_train = tensorflow.convert_to_tensor(y_train, dtype=tensorflow.int8)
    X_test = tensorflow.convert_to_tensor(X_test, dtype=tensorflow.float32)
    y_test = tensorflow.convert_to_tensor(y_test, dtype=tensorflow.int8)

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = import_samples_NN_for_pixel(label_mapping='Ohne_Auto_See')
    print('Fertig')
