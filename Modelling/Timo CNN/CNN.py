import segmentation_models
import tensorflow as tf


def cnn_segmentation_models(epochs, batch_size, optim, learning_rate, metrics, loss, name, backbone_name, activation):
    from segmentation_models import Unet
    import wandb
    from wandb.keras import WandbCallback

    from Modelling.Baselines.Data_Preprocessing import import_labeled_data, import_labeled_photos
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # login
    wandb.init(project="cnn_segmentation_models", entity="pds_project", name='segmentation_models_Unet', reinit=True,
               dir=r"C:\Users\vdwti\PycharmProjects\pds_gyrocopter\Modelling\Timo CNN")
    wandb.config.update({"epochs": epochs, "batch_size": batch_size, 'metrics': metrics, 'loss': loss, 'optim': optim,
                         'learning_rate': learning_rate,
                         'backbone_name': backbone_name,
                         'activation': activation})

    # import Data
    x, y = import_labeled_photos()
    x_train_data = x[4:]
    y_train_data = y[4:]
    x_test_data = x[0:4]
    y_test_data = y[0:4]

    # Transofrm Data to tensor
    x_train = tf.convert_to_tensor(x_train_data, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train_data, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test_data, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test_data, dtype=tf.float32)

    # define number of channels
    N = x_train.shape[-1]

    # define model
    model = Unet(backbone_name=backbone_name, encoder_weights=None, input_shape=(None, None, N), activation=activation,
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
    epochs = 5000
    batch_size = 2
    activation = 'softmax'
    metrics = ['categorical_accuracy']  # categorical_accuracy
    loss = tf.keras.losses.CategoricalHinge()
    # loss=segmentation_models_pytorch.losses.constants.MULTICLASS_MODE
    name = 'segmentation_models_Unet_V1'
    learning_rate = 0.001
    optim = tf.keras.optimizers.Adam(learning_rate)
    backbone_name = 'resnet34'

    cnn_segmentation_models(epochs, batch_size, optim, learning_rate, metrics, loss, name, backbone_name,activation)

"""
# Configuration
    epochs = 1000
    batch_size = 5
    metrics = ['accuracy']
    loss = tf.keras.losses.CategoricalCrossentropy()
    name = 'segmentation_models_Unet_V1'
    wandb.config.update({"epochs": epochs, "batch_size": batch_size,'metrics':metrics,'loss':loss})
"""
