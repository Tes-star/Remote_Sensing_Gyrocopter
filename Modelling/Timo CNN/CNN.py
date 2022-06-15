import numpy as np
import segmentation_models
import tensorflow as tf


def cnn_segmentation_models(augmentation, encoder_freeze, epochs, batch_size, optim, learning_rate, metrics, loss, name,
                            backbone_name, activation):
    from segmentation_models import Unet
    import wandb
    from wandb.keras import WandbCallback

    from Modelling.Baselines.Data_Preprocessing import import_labeled_data, import_labeled_photos
    import os



    # login
    wandb.init(project="cnn_segmentation_models", entity="pds_project", name='segmentation_models_Unet', reinit=True
               , dir=r'C:\Users\vdwti\PycharmProjects\pds_gyrocopter\Modelling\Timo CNN')
    wandb.config.update({"epochs": epochs, "batch_size": batch_size, 'metrics': metrics, 'loss': loss, 'optim': optim,
                         'learning_rate': learning_rate,
                         'backbone_name': backbone_name,
                         'activation': activation,
                         'encoder_freeze': encoder_freeze,
                         'augmentation': augmentation})

    # import Data
    x, y = import_labeled_photos()
    x_train_data = x[4:]
    y_train_data = y[4:]
    x_test_data = x[0:4]
    y_test_data = y[0:4]

    # add augmentations
    # train
    x_train_data_original = x_train_data.copy()
    for x in x_train_data_original:
        x_train_data.append(np.rot90(x, 1))
        x_train_data.append(np.flipud(x))

    y_train_data_original = y_train_data.copy()
    for y in y_train_data_original:
        y_train_data.append(np.rot90(y))
        y_train_data.append(np.flipud(y))

    # Transofrm Data to tensor
    x_train = tf.convert_to_tensor(x_train_data, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train_data, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test_data, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test_data, dtype=tf.float32)

    # define number of channels
    N = x_train.shape[-1]

    # define model
    model = Unet(backbone_name=backbone_name, encoder_freeze=encoder_freeze, encoder_weights=None,
                 input_shape=(None, None, N), activation=activation,
                 classes=8)

    model.summary()
    model.compile(optim, loss=loss, metrics=metrics)

    model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[WandbCallback()]
    )
    # wandb.finish()


if __name__ == '__main__':
    augmentation = "1x rot90,1x flipud"
    epochs = 300
    batch_size = 1
    activation = 'softmax'
    metrics = ['categorical_accuracy']  # categorical_accuracy
    loss = tf.keras.losses.CategoricalCrossentropy()
    # loss=segmentation_models_pytorch.losses.constants.MULTICLASS_MODE
    name = 'segmentation_models_Unet_V1'
    learning_rate = 0.0001
    encoder_freeze = True
    optim = tf.keras.optimizers.Adam(learning_rate)
    backbone_name = 'efficientnetb2'
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
# Configuration
    epochs = 1000
    batch_size = 5
    metrics = ['accuracy']
    loss = tf.keras.losses.CategoricalCrossentropy()
    name = 'segmentation_models_Unet_V1'
    wandb.config.update({"epochs": epochs, "batch_size": batch_size,'metrics':metrics,'loss':loss})
"""
