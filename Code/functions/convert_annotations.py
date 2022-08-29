import os
import cv2
import numpy as np
import pandas as pd
import spectral as spy
from spectral import envi
import xmltodict as xmltodict
import matplotlib.pyplot as plt
from Code.functions.class_ids import map_float_id2rgb


def convert_xml_annotation_to_mask(xml_file: str, path_picture: str, path_export: str, windowsize_r: int,
                                   windowsize_c: int):
    """
    function which convert xml annotation from Roboflow to mask and export annotated picture with label-band,
    requires picture in envi format in path_picture with same picture name as xml_file
    :param xml_file: absolute path plus filename from xml annotation file
    :param path_picture: path to subimages in envi format
    :param path_export: path to save labeled picture
    :param windowsize_r: width of Subimages
    :param windowsize_c: height of Subimages
    :return: save labeled images with label-band in path_export and return image array and mask
    """

    # read xml-file and convert to dictionary
    with open(xml_file) as fd:
        doc = xmltodict.parse(fd.read())

    # extract image size
    width = int(doc['annotation']['size']['width'])
    height = int(doc['annotation']['size']['height'])

    # extract image name and calculate grid position
    filename = doc['annotation']['filename']
    file_split = filename.split('_')
    grid_pos_r = int(file_split[2]) * windowsize_r  # Gridposition row
    grid_pos_c = int(file_split[3]) * windowsize_c  # Gridposition column

    # build original_name
    original_name = file_split[0] + "_" + file_split[1] + "_" + str(file_split[2]).zfill(8) + "_" + \
                    str(file_split[3]).zfill(8) + "_" + str(grid_pos_r) + "_" + str(grid_pos_c)

    # define annotated objects in dictionary
    class_objects = {0: 'None', 1: 'Wiese', 2: 'Strase', 3: 'Auto', 4: 'See', 5: 'Schienen', 6: 'Haus', 7: 'Wald'}

    # build array with same shape as annotated picture
    # 0 = standard value for unannotated pixels
    mask = np.zeros((height, width, 1))

    # do for every annotated objects in objects
    for class_key, class_value in class_objects.items():

        # do for every polygon in annotated picture
        for annotation_object in doc['annotation']['object']:

            # initialization empty list for coordinates from polygon
            coords = []

            # extract polygon coordinates and name of Klass
            polygon = annotation_object['polygon']
            name = annotation_object['name']

            # skip if current polygon class is not class_value
            if class_value != name:
                continue

            # extract all coordinates from polygon in list format [[x1,y1], [x2,y2], ...]
            for i in range(1, int(len(polygon) / 2) + 1):
                point = [int((float(polygon['x' + str(i)]))), int(float(polygon['y' + str(i)]))]
                coords.append(point)

            # convert list to array
            coords = np.array(coords, dtype=np.int32)

            # use cv2.fillPoly to convert area in polygon to mask with obj_key as value in dat_holder_matrix
            mask = cv2.fillPoly(mask, [coords], class_key)


    # creat plot with class_rgb
    df = pd.DataFrame(mask.reshape(mask.shape[0]*mask.shape[1], mask.shape[2]))
    df = map_float_id2rgb(dataframe=df, column=mask.shape[2]-1)

    # extract color values
    df['class_color1'] = df['class_color'].apply(lambda x: x[0])
    df['class_color2'] = df['class_color'].apply(lambda x: x[1])
    df['class_color3'] = df['class_color'].apply(lambda x: x[2])

    # reshape pixel to image for annotation picture
    annotation_arr = np.array(df[['class_color1', 'class_color2', 'class_color3']])
    annotation_arr = np.reshape(annotation_arr, (mask.shape[0], mask.shape[1], 3))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(annotation_arr)
    fig.show()
    #fig.savefig(original_name)

    # read original_name
    path_dat = path_picture + "/" + original_name + '_.dat'
    path_hdr = path_picture + "/" + original_name + '_.hdr'
    img = envi.open(file=path_hdr, image=path_dat)
    img_arr = img.load()

    # add label array at the last band to image array
    combined_arr = np.concatenate((img_arr, mask), -1)

    # extract metadata
    arr_metadata = img.metadata

    # add label information
    arr_metadata['wavelength'].append('label')
    arr_metadata['band names'].append('label')
    arr_metadata['fwhm'].append('label')

    # change number of bands
    arr_metadata['bands'] = len(arr_metadata['wavelength'])

    # define export path
    if np.logical_and(width == windowsize_c, height == windowsize_r):
        # build export folder
        if not os.path.exists(path_export):
            os.makedirs(path_export)
        path_hdr_labeled = path_export + '/' + original_name + '_.hdr'
    else:
        # build wrong_size_images folder
        folder = path_export + '/wrong_image_size'
        if not os.path.exists(folder):
            os.makedirs(folder)

        path_hdr_labeled = path_export + '/wrong_image_size/' + original_name + '_.hdr'

    # save image with new band label
    envi.save_image(hdr_file=path_hdr_labeled, image=combined_arr, metadata=arr_metadata, dtype="float32", ext='.dat',
                    interleave='bsq', force=True)

    return img_arr, mask


def convert_all_annotations(path_annotations: str, path_pictures: str, path_export: str, windowsize_r: int,
                            windowsize_c: int):
    """

    :param path_annotations: path to folder with xml annotation files
    :param path_pictures: path to subimages in envi format
    :param path_export: path to save labeled picture
    :param windowsize_r: width of Subimages
    :param windowsize_c: height of Subimages
    :return: save labeled images with label-band in path_export
    """

    # build list with all files in path_data
    files = os.listdir(path_annotations)

    # remove all files which not end with '.xml'
    for file in files:
        if not file.endswith('.xml'):
            files.remove(file)

    # create folder for export envi files with added label band
    path_labeled = path_export
    if not os.path.exists(path_labeled):
        os.makedirs(path_labeled)

    for file in files:
        path_xml_file = path_annotations + '/' + file
        convert_xml_annotation_to_mask(xml_file=path_xml_file, path_picture=path_pictures, path_export=path_labeled,
                                       windowsize_r=windowsize_r, windowsize_c=windowsize_c)

