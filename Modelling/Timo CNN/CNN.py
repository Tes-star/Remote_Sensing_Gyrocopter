from segmentation_models import Unet
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
from Modelling.Baselines.Data_Preprocessing import import_labeled_data, import_labeled_photos
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# login
wandb.init(project="NN_for_pixels", entity="pds_project", name='segmentation_models_Unet')
wandb.init(config={})

# Configuration
epochs = 1000
batch_size = 5
metrics = ['accuracy']
loss = tf.keras.losses.CategoricalCrossentropy()
name = 'segmentation_models_Unet_V1'
wandb.config.update({"epochs": epochs, "batch_size": batch_size,'metrics':metrics,'loss':loss})


# import Data
x, y = import_labeled_photos()
x_train_data = x[0:10]
y_train_data = y[0:10]
x_test_data = x[10:]
y_test_data = y[10:]

# Transofrm Data to tensor
x_train = tf.convert_to_tensor(x_train_data, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train_data, dtype=tf.float32)
x_test = tf.convert_to_tensor(x_test_data, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test_data, dtype=tf.float32)

# define number of channels
N = x_train.shape[-1]

# define model
model = Unet(backbone_name='resnet34', encoder_weights=None, input_shape=(None, None, N), activation='sigmoid')

model.summary()
model.compile('Adam', loss=loss, metrics=metrics)

wandb.init(project="NN_for_pixels", entity="pds_project",
           name=name)
model.fit(
    x=x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[WandbCallback()],
)
wandb.finish()
