import operator

import numpy as np
import tensorflow as tf
from keras.losses import CategoricalCrossentropy
from segmentation_models.losses import JaccardLoss, DiceLoss
from sklearn.metrics import precision_score, recall_score, f1_score
import wandb
from numpy import float16
from segmentation_models import Unet
from wandb.integration.keras import WandbCallback
from Modelling.Data_Preprocessing import import_labeled_photos

# Configs
batch_size = 17
epochs = 1
metrics = ['categorical_accuracy']
label_mapping = 'Ohne_Auto_See'  # Alternative:'Grünflächen'

match label_mapping:
    case 'Ohne_Auto_See':
        # labels = {0: 'None', 1: 'Wiese', 2: 'Straße', 3: 'Schienen', 4: 'Haus', 5: 'Wald'}
        labels = ['None', 'Wiese', 'Straße', 'Schienen', 'Haus', 'Wald']
    case 'Grünflächen':
        # labels = {0: 'None', 1: 'Grünflächen', 2: 'Straße', 3: 'Schienen', 4: 'Haus'}
        labels = ['None', 'Grünflächen', 'Straße', 'Schienen', 'Haus']
    case _:  # else
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]

""" wird aktuell nicht mehr benötigt. Bitte stehen lassen
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(gpus[0], 'GPU')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
"""


def cnn_sweep():
    wandb.init()
    # Access all hyperparameter values through wandb.config
    config = wandb.config
    # set configs
    optim = tf.keras.optimizers.Adam(config.learning_rate)

    loss = JaccardLoss()  # to avoid errors
    match config.loss:
        case 'JaccardLoss':
            loss = JaccardLoss()
        case 'DiceLoss':
            loss = DiceLoss()
        case 'CategoricalCrossentropy':
            loss = CategoricalCrossentropy()

    """wird aktuell nicht mehr benötigt. Bitte stehen lassen
    import wandb
    # login
    run = wandb.init(project="cnn_segmentation_models", entity="pds_project", name='segmentation_models_Unet',
                     )
    run.config.update({"epochs": epochs, "batch_size": batch_size, 'metrics': metrics, 'loss': loss, 'optim': optim,
                       'learning_rate': learning_rate,
                       'backbone_name': backbone_name,
                       'activation': activation,
                       'encoder_freeze': encoder_freeze,
                       'augmentation': augmentation
                       })
    run.config.update({'images_train': len(x_train_data),
                       'images_test': len(x_test_data),
                       })
    """
    # wandb.log({"label_mapping": str(label_mapping)})

    # Choose bands
    bands = list(range(0, 104, config.band_dist))
    bands.append(105)
    bands.append(106)
    bands.append(107)
    bands.append(108)

    # import Data
    x, y = import_labeled_photos(bands=bands, label_mapping=label_mapping)

    x_train = x[6:]
    y_train = y[6:]
    x_test = x[0:6]
    y_test = y[0:6]
    del x, y

    # add augmentations
    import albumentations as A

    transform_light = A.Compose([
        A.HorizontalFlip(p=1),
        A.RandomBrightnessContrast(p=1, brightness_by_max=False,
                                   contrast_limit=(-config.aug_contrast_light, config.aug_contrast_light),
                                   brightness_limit=(-config.aug_brightness_light, config.aug_brightness_light)),
        A.MultiplicativeNoise(p=1, multiplier=(1 - config.aug_noise_light, 1 + config.aug_noise_light),
                              elementwise=False),
        A.PixelDropout(p=1, dropout_prob=config.aug_dropout_light)
    ])
    transform_middle = A.Compose([
        A.VerticalFlip(p=1),
        A.RandomBrightnessContrast(p=1, brightness_by_max=False,
                                   contrast_limit=(-config.aug_contrast_middle, config.aug_contrast_middle),
                                   brightness_limit=(-config.aug_brightness_middle, config.aug_brightness_middle)),
        A.MultiplicativeNoise(p=1, multiplier=(1 - config.aug_noise_middle, 1 + config.aug_noise_middle),
                              elementwise=False),
        A.GaussianBlur(p=1),
        A.PixelDropout(p=1, dropout_prob=config.aug_dropout_middle)
    ])
    transform_hard = A.Compose([
        A.RandomCrop(width=150, height=150),
        # A.ElasticTransform(p=1),
        A.GaussianBlur(p=1),
        A.RandomBrightnessContrast(p=1, brightness_by_max=False,
                                   contrast_limit=(-config.aug_contrast_hard, config.aug_contrast_hard),
                                   brightness_limit=(-config.aug_brightness_hard, config.aug_brightness_hard)),
        A.MultiplicativeNoise(p=1, multiplier=(1 - config.aug_noise_hard,
                                               1 + config.aug_noise_hard), elementwise=True),
        A.PixelDropout(p=1, dropout_prob=config.aug_dropout_hard),
        A.PadIfNeeded(min_height=224, min_width=224, p=1)
    ])

    aug_list = [transform_light, transform_middle, transform_hard]

    pict_with_labels = zip(x_train.copy(), y_train.copy())
    for image, mask in pict_with_labels:
        for aug in aug_list:
            transformed = aug(image=image, mask=mask)
            x_train.append(transformed['image'])
            y_train.append(transformed['mask'])

    # Transform Data to tensor
    x_train = tf.convert_to_tensor(x_train, dtype=float16)
    y_train = tf.convert_to_tensor(y_train, dtype=float16)
    x_test = tf.convert_to_tensor(x_test, dtype=float16)
    y_test = tf.convert_to_tensor(y_test, dtype=float16)

    # define number of channels
    N = x_train.shape[-1]

    # define model
    model = Unet(backbone_name=config.backbone, encoder_freeze=config.encoder_freeze, encoder_weights=None,
                 input_shape=(None, None, N), activation=config.activation,
                 classes=len(labels))

    # model.summary()
    model.compile(optim, loss=loss, metrics=metrics)

    # train
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,  # config.batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[WandbCallback()]
    )
    # make predictions train
    y_train = tf.reshape(tf.math.argmax(y_train, axis=3), [-1]).numpy()
    pred_y_train = model.predict(x_train)
    pred_y_train = tf.reshape(tf.math.argmax(pred_y_train, axis=3), [-1]).numpy()

    # make predictions test
    y_test = tf.reshape(tf.math.argmax(y_test, axis=3), [-1]).numpy()
    pred_y_test = model.predict(x_test)
    pred_y_test = tf.reshape(tf.math.argmax(pred_y_test, axis=3), [-1]).numpy()

    labels_train = np.unique(np.concatenate((y_train, pred_y_train)))
    labels_test = np.unique(np.concatenate((y_test, pred_y_test)))

    # calculate and log metrics
    precision_train = precision_score(y_train, pred_y_train, labels=labels_train, average='micro')
    precision_test = precision_score(y_test, pred_y_test, labels=labels_test, average='micro')

    recall_train = recall_score(y_train, pred_y_train, labels=labels_train, average='micro')
    recall_test = recall_score(y_test, pred_y_test, labels=labels_test, average='micro')

    f1_train = f1_score(y_train, pred_y_train, labels=labels_train, average='micro')
    f1_test = f1_score(y_test, pred_y_test, labels=labels_test, average='micro')

    labels_test_str = operator.itemgetter(*labels_test)(labels)
    wandb.log(
        {"conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=y_test, preds=pred_y_test,
                                                 class_names=labels_test_str)})

    wandb.log({'precision_train': precision_train})
    wandb.log({'precision_test': precision_test})

    wandb.log({'recall_train': recall_train})
    wandb.log({'recall_test': recall_test})

    wandb.log({'f1_train': f1_train})
    wandb.log({'f1_test': f1_test})
    print("Finshed Job")
    wandb.finish()

if __name__ == '__main__':
    """
    Better use Sweep_upload_data.ipynb to avoid errors and bad visualisation
    """

    # define sweep_id
    sweep_id = 'ovzbvxa7'
    # wandb sweep sweep.yaml

    # run the sweep
    wandb.agent(sweep_id, function=cnn_sweep, project="cnn_segmentation_models", entity="pds_project")
