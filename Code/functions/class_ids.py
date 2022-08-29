def get_class_dictionary():
    """
    :return: dictionary key = class_id, value = class_name  
    """
    class_dict = {0: 'None', 1: 'Wiese', 2: 'Strase', 3: 'Auto', 4: 'See', 5: 'Schienen', 6: 'Haus', 7: 'Wald'}
    return class_dict


def get_class_list():
    """
    :return: list with ordered class_names 
    """
    class_lst = ['None', 'Wiese', 'Strase','Auto','See','Schienen', 'Haus', 'Wald']
    return class_lst


def map_float_id2rgb(dataframe, column):
    """
    Map class_id to rgb values
    :param dataframe: dataframe which columns represent bands and every row is one pixel 
    :param column: column with map-values (label-column)
    :return: dataframe with new colum class_color consist of list with rgb values
    """
    
    df = dataframe.copy()

    df['class_color'] = df[column].map(
        {-1.0: [0, 0, 0],       # schwarz -> Pixel im nicht annotiertem Bild
         0.0: [51, 51, 51],     # grauschwarz -> nicht annotierter Pixel im annotierten Bild
         1.0: [188, 238, 104],  # hellgrün -> Wiese
         2.0: [140, 140, 140],  # grau -> Straße
         3.0: [255, 215, 0],    # gelb -> Auto
         4.0: [0, 191, 255],    # blau -> See
         5.0: [255, 20, 147],   # pink -> Schienen
         6.0: [205, 38, 38],    # rot -> Haus
         7.0: [34, 139, 34]     # dunkelgrün -> Wald
         })

    return df


def map_int_id2name(dataframe, column):
    """
    Map class_id to rgb values
    :param dataframe: dataframe which columns represent bands and every row is one pixel 
    :param column: column with map-values (label-column)
    :return: dataframe with new colum class_color consist of list with rgb values
    """
    
    df = dataframe.copy()

    df['class_name'] = df[column].map(
        {0.0: 'None',
         1.0: 'Wiese',
         2.0: 'Strase',
         3.0: 'Auto',
         4.0: 'See',
         5.0: 'Schienen',
         6.0: 'Haus',
         7.0: 'Wald'
         })

    return df


def new_label_mapping(dataframe, map_column, label_mapping):
    """
    function which replace specific class ids to other class ids
    :param dataframe: dataframe which columns represent bands and every row is one pixel
    :param map_column: column with map-values (label-column)
    :param label_mapping: string which represent map-values
    :return: input dataframe with mapped values in map_column
    """

    match label_mapping:
        case None:
            new_labels = [0, 1, 2, 3, 4, 5, 6, 7]
        case 'Ohne_Auto':
            new_labels = [0, 1, 2, 2, 4, 5, 6, 7]
            # Auto= zaehlt auch zur Straße
        case 'Ohne_Auto_See':
            new_labels = [0, 1, 2, 2, 0, 5, 6, 7]
            # Auto= zaehlt auch zur Straße
            # See=0
        case 'Gruenflaechen':
            new_labels = [0, 1, 2, 2, 0, 5, 6, 1]
            # Auto=2
            # See=0
            # Wald=Wiese=1

    dataframe[map_column] = dataframe[map_column].replace([0, 1, 2, 3, 4, 5, 6, 7], new_labels)

    return dataframe


def new_label_mapping_cnn(label_mapping):
    """
    function which replace specific class ids to other class ids
    :param label_mapping: string which represent map-values
    :return: list with label_mapping
    """
    match label_mapping:
        case None:
            label_mapping = [0, 1, 2, 3, 4, 5, 6, 7]
        case 'Ohne_Auto_See':
            label_mapping = [0, 1, 2, 2, 0, 3, 4, 5]
            # Auto= zaehlt auch zur Straße
            # See=0
        case 'Grünflächen':
            label_mapping = [0, 1, 2, 2, 0, 3, 4, 1]
            # Auto=2
            # See=0
            # Wald=Wiese=1


    return label_mapping
