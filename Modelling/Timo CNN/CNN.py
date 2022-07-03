import tensorflow as tf
from wandb.integration.keras import WandbCallback


def cnn_segmentation_models(augmentation, encoder_freeze, epochs, batch_size, optim, learning_rate, metrics, loss, name,
                            backbone_id, activation):
    import segmentation_models as sm
    import wandb
    import keras
    import os
    from Modelling.Baselines.Data_Preprocessing import import_labeled_photos
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    backbone_list = ['efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'resnet18', 'resnet34']
    backbone_name = backbone_list[backbone_id]

    # login
    run=wandb.init(project="cnn_segmentation_models", entity="pds_project", name='segmentation_models_Unet', reinit=True)
              # , dir=r'C:\Users\vdwti\PycharmProjects\pds_gyrocopter\Modelling\Timo CNN')
    run.config.update({"epochs": epochs, "batch_size": batch_size, 'metrics': metrics, 'loss': loss, 'optim': optim,
                         'learning_rate': learning_rate,
                         'backbone_name': backbone_name,
                         'activation': activation,
                         'encoder_freeze': encoder_freeze,
                         'augmentation': augmentation
                         })

    # import Data
    x, y = import_labeled_photos()
    x_train_data = x[6:]
    y_train_data = y[6:]
    x_test_data = x[0:6]
    y_test_data = y[0:6]

    run.config.update({'images_train': len(x_train_data),
                         'images_test': len(x_test_data),
                         })

    # add augmentations
    # train
    import albumentations as A
    aug_noise_light = 0.05
    aug_noise_middle = 0.05
    aug_noise_hard = 0.05
    aug_dropout_light = 0.05
    aug_dropout_middle = 0.05
    aug_dropout_high = 0.05
    aug_contrast_low = 0.05
    aug_contrast_middle = 0.05
    aug_contrast_high = 0.05
    aug_brightness_low = 0.05
    aug_brightness_middle = 0.05
    aug_brightness_high = 0.05

    transform_light = A.Compose([
        A.HorizontalFlip(p=1),
        A.RandomBrightnessContrast(p=1, brightness_by_max=False, contrast_limit=(-aug_contrast_low, aug_contrast_low),
                                   brightness_limit=(-aug_brightness_low, aug_brightness_low)),
        A.MultiplicativeNoise(p=1, multiplier=(1 - aug_noise_light, 1 + aug_noise_light), elementwise=False),
        A.PixelDropout(p=1, dropout_prob=aug_dropout_light)
    ])
    transform_middle = A.Compose([
        A.VerticalFlip(p=1),
        A.RandomBrightnessContrast(p=1, brightness_by_max=False,
                                   contrast_limit=(-aug_contrast_middle, aug_contrast_middle),
                                   brightness_limit=(-aug_brightness_middle, aug_brightness_middle)),
        A.MultiplicativeNoise(p=1, multiplier=(1 - aug_noise_middle, 1 + aug_noise_middle), elementwise=False),
        A.GaussianBlur(p=1),
        A.PixelDropout(p=1, dropout_prob=aug_dropout_middle)
    ])
    transform_hard = A.Compose([
        A.RandomCrop(width=150, height=150),
        # A.ElasticTransform(p=1),
        A.GaussianBlur(p=1),
        A.RandomBrightnessContrast(p=1, brightness_by_max=False, contrast_limit=(-aug_contrast_high, aug_contrast_high),
                                   brightness_limit=(-aug_brightness_high, aug_brightness_high)),
        A.MultiplicativeNoise(p=1, multiplier=(1 - aug_noise_hard,
                                               1 + aug_noise_hard), elementwise=True),
        A.PixelDropout(p=1, dropout_prob=aug_dropout_high),
        A.PadIfNeeded(min_height=224, min_width=224, p=1)
    ])

    aug_list = [transform_light, transform_middle, transform_hard]
    wandb.config.update({'augmentations': len(aug_list)
                         })
    pict_with_labels = zip(x_train_data.copy(), y_train_data.copy())
    for image, mask in pict_with_labels:
        for aug in aug_list:
            transformed = aug(image=image, mask=mask)
            x_train_data.append(transformed['image'])
            y_train_data.append(transformed['mask'])

    # Transform Data to tensor
    x_train = tf.convert_to_tensor(x_train_data, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train_data, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test_data, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test_data, dtype=tf.float32)
    del x_train_data, y_train_data, x_test_data, y_test_data

    # define number of channels
    N = x_train.shape[-1]

    # define model
    model = sm.Unet(backbone_name=backbone_name, encoder_freeze=encoder_freeze, encoder_weights=None,
                    input_shape=(None, None, N), activation=activation,
                    classes=8)

    #model.summary()
    model.compile(optim, loss=loss, metrics=metrics)

    model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[WandbCallback()]
    )
    run.finish()


if __name__ == '__main__':
    # augmentation
    aug_noise_light = 0.05
    aug_noise_middle = 0.05
    aug_noise_hard = 0.05
    aug_dropout_light = 0.05
    aug_dropout_middle = 0.05
    aug_dropout_high = 0.05
    aug_contrast_low = 0.05
    aug_contrast_middle = 0.05
    aug_contrast_high = 0.05
    aug_brightness_low = 0.05
    aug_brightness_middle = 0.05
    aug_brightness_high = 0.05
    augmentation = "Test 2 Augmentations Soft middle hard"

    epochs = 1
    batch_size = 1
    activation = 'softmax'
    metrics = ['categorical_accuracy']  # categorical_accuracy
    loss = tf.keras.losses.CategoricalCrossentropy()
    # loss=segmentation_models_pytorch.losses.constants.MULTICLASS_MODE
    name = 'segmentation_models_Unet_V1'
    learning_rate = 0.0001
    encoder_freeze = False
    optim = tf.keras.optimizers.Adam(learning_rate)
    backbone_id = 3
    cnn_segmentation_models(augmentation=augmentation,
                            encoder_freeze=encoder_freeze,
                            epochs=epochs,
                            batch_size=batch_size,
                            optim=optim,
                            learning_rate=learning_rate,
                            metrics=metrics,
                            loss=loss,
                            name=name,
                            backbone_id=backbone_id,
                            activation=activation)
    """
    batch_size = 5
    cnn_segmentation_models(augmentation=augmentation,
                            encoder_freeze=encoder_freeze,
                            epochs=epochs,
                            batch_size=batch_size,
                            optim=optim,
                            learning_rate=learning_rate,
                            metrics=metrics,
                            loss=loss,
                            name=name,
                            backbone_name=backbone_name,
                            activation=activation)


    cnn_segmentation_models(augmentation=augmentation,
                            encoder_freeze=encoder_freeze,
                            epochs=epochs,
                            batch_size=batch_size,
                            optim=optim,
                            learning_rate=learning_rate,
                            metrics=metrics,
                            loss=loss,
                            name=name,
                            backbone_name=backbone_name,
                            activation=activation)
    
    encoder_freeze = False
    cnn_segmentation_models(augmentation=augmentation,
                            encoder_freeze=encoder_freeze,
                            epochs=epochs,
                            batch_size=batch_size,
                            optim=optim,
                            learning_rate=learning_rate,
                            metrics=metrics,
                            loss=loss,
                            name=name,
                            backbone_name=backbone_name,
                            activation=activation)
    import tensorflow_addons as tfa

    loss = segmentation_models.losses.JaccardLoss()
    cnn_segmentation_models(augmentation=augmentation,
                            encoder_freeze=encoder_freeze,
                            epochs=epochs,
                            batch_size=batch_size,
                            optim=optim,
                            learning_rate=learning_rate,
                            metrics=metrics,
                            loss=loss,
                            name=name,
                            backbone_name=backbone_name,
                            activation=activation)
    loss = segmentation_models.losses.DiceLoss()
    cnn_segmentation_models(augmentation=augmentation,
                            encoder_freeze=encoder_freeze,
                            epochs=epochs,
                            batch_size=batch_size,
                            optim=optim,
                            learning_rate=learning_rate,
                            metrics=metrics,
                            loss=loss,
                            name=name,
                            backbone_name=backbone_name,
                            activation=activation)
    loss = segmentation_models.losses.CategoricalFocalLoss()
    cnn_segmentation_models(augmentation=augmentation,
                            encoder_freeze=encoder_freeze,
                            epochs=epochs,
                            batch_size=batch_size,
                            optim=optim,
                            learning_rate=learning_rate,
                            metrics=metrics,
                            loss=loss,
                            name=name,
                            backbone_name=backbone_name,
                            activation=activation)
    loss = tfa.losses.focal_loss.SigmoidFocalCrossEntropy()
    cnn_segmentation_models(augmentation=augmentation,
                            encoder_freeze=encoder_freeze,
                            epochs=epochs,
                            batch_size=batch_size,
                            optim=optim,
                            learning_rate=learning_rate,
                            metrics=metrics,
                            loss=loss,
                            name=name,
                            backbone_name=backbone_name,
                            activation=activation)
                            """

"""
# Configuration
    epochs = 1000
    batch_size = 5
    metrics = ['accuracy']
    loss = tf.keras.losses.CategoricalCrossentropy()
    name = 'segmentation_models_Unet_V1'
    wandb.config.update({"epochs": epochs, "batch_size": batch_size,'metrics':metrics,'loss':loss})
"""
