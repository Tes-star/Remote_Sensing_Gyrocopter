import tensorflow as tf
from tensorflow import keras
import numpy as np

print("Start")

print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU'))
print(tf.__version__)
"""
import torch

x = torch.rand(5, 3)
print(x)

if not torch.cuda.is_available():
   print ("Cuda is available")
   device_id = torch.cuda.current_device()
   gpu_properties = torch.cuda.get_device_properties(device_id)
   print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
          "%.1fGb total memory.\n" %
          (torch.cuda.device_count(),
          device_id,
          gpu_properties.name,
          gpu_properties.major,
          gpu_properties.minor,
          gpu_properties.total_memory / 1e9))
else:
   print ("Cuda is not available")
"""
from tensorflow.python.client import device_lib
assert 'GPU' in str(device_lib.list_local_devices())
print(device_lib.list_local_devices())
from keras import backend


# data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_images = np.expand_dims(train_images, axis=3) / 255.
test_images = np.expand_dims(test_images, axis=3) / 255.

print(train_images[0].shape)
print(train_labels[0])

# model
num_filters = 8
stride = 1
filter_size = 3
pool_size = 2

inputs = keras.Input((28, 28, 1,))
conv_layer = keras.layers.Conv2D(num_filters, filter_size, strides=stride)
conv = conv_layer(inputs)
pool_layer = keras.layers.MaxPooling2D(pool_size=pool_size)
pool = pool_layer(conv)
flatten_layer = keras.layers.Flatten()
flat_features = flatten_layer(pool)
output = keras.layers.Dense(10, activation='softmax')(flat_features)

model = keras.Model(inputs, output)
model.compile(loss='categorical_crossentropy', metrics='accuracy')
model.summary()

# training
model.fit(
    train_images, keras.utils.to_categorical(train_labels),
    epochs=5,
    validation_data=(test_images, keras.utils.to_categorical(test_labels))
)

print("Ende")