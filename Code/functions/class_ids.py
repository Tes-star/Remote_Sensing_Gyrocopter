# class_objects = {0: 'None', 1: 'Wiese', 2: 'Strase', 3: 'Auto', 4: 'See', 5: 'Schienen', 6: 'Haus', 7: 'Wald'}

def map_float_id2rgb(dataframe, column):

    df = dataframe.copy()

    df['class_color'] = df[column].map(
        {-1.0: [0, 0, 0],       # schwarz -> Pixel im nicht annotiertem Bild
         0.0: [51, 51, 51],     # grauschwarz -> nicht annotierter Pixel im annotierten Bild
         1.0: [188, 238, 104],  # hellgrün -> Wiese
         2.0: [140, 140, 140],  # grau -> Straße
         3.0: [255, 255, 255],  # Weiss -> Auto
         4.0: [0,191,255],      # blau -> See
         5.0: [255, 20, 147],   # pink -> Schienen
         6.0: [205, 38, 38],    # rot -> Haus
         7.0: [34, 139, 34]     # dunkelgrün -> Wald
         })

    return df


def map_int_id2name(dataframe, column):

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
