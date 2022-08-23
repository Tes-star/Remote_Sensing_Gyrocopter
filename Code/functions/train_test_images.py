import pandas as pd


def train_test_image_names():
    train_images = ['Teilbild_Oldenburg_00000000_00000000_0_0_',
                    'Teilbild_Oldenburg_00000000_00000002_0_400_',
                    'Teilbild_Oldenburg_00000000_00000008_0_1600_',
                    'Teilbild_Oldenburg_00000000_00000009_0_1800_',
                    'Teilbild_Oldenburg_00000000_00000011_0_2200_',
                    'Teilbild_Oldenburg_00000000_00000012_0_2400_',
                    'Teilbild_Oldenburg_00000000_00000013_0_2600_',
                    'Teilbild_Oldenburg_00000000_00000014_0_2800_',
                    'Teilbild_Oldenburg_00000000_00000015_0_3000_',
                    'Teilbild_Oldenburg_00000001_00000000_200_0_',
                    'Teilbild_Oldenburg_00000001_00000005_200_1000_',
                    'Teilbild_Oldenburg_00000001_00000012_200_2400_',
                    'Teilbild_Oldenburg_00000001_00000014_200_2800_',
                    'Teilbild_Oldenburg_00000001_00000016_200_3200_',
                    'Teilbild_Oldenburg_00000002_00000000_400_0_',
                    'Teilbild_Oldenburg_00000002_00000002_400_400_',
                    'Teilbild_Oldenburg_00000002_00000008_400_1600_',
                    'Teilbild_Oldenburg_00000002_00000016_400_3200_',
                    'Teilbild_Oldenburg_00000003_00000016_600_3200_',
                    'Teilbild_Oldenburg_00000004_00000012_800_2400_',
                    'Teilbild_Oldenburg_00000005_00000001_1000_200_',
                    'Teilbild_Oldenburg_00000005_00000011_1000_2200_',
                    'Teilbild_Oldenburg_00000008_00000000_1600_0_',
                    'Teilbild_Oldenburg_00000008_00000005_1600_1000_',
                    'Teilbild_Oldenburg_00000008_00000010_1600_2000_',
                    'Teilbild_Oldenburg_00000008_00000012_1600_2400_']

    test_images = ['Teilbild_Oldenburg_00000005_00000013_1000_2600_',
                   'Teilbild_Oldenburg_00000006_00000000_1200_0_',
                   'Teilbild_Oldenburg_00000006_00000010_1200_2000_',
                   'Teilbild_Oldenburg_00000006_00000014_1200_2800_',
                   'Teilbild_Oldenburg_00000007_00000000_1400_0_',
                   'Teilbild_Oldenburg_00000007_00000006_1400_1200_',
                   'Teilbild_Oldenburg_00000007_00000008_1400_1600_']

    df_train = pd.DataFrame({'picture_name': train_images})
    df_test = pd.DataFrame({'picture_name': test_images})

    return df_train, df_test
