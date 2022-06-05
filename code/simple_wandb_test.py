import random
import wandb
from sklearn.model_selection import train_test_split
from wandb.keras import WandbCallback
from tensorflow import keras

# login
wandb.init(project="test-project", entity="pds_project", name='Test1')


# model
inputs = keras.Input(2)
dense_layer = keras.layers.Dense(1, use_bias=False)
output = dense_layer(inputs)

model = keras.Model(inputs, output)
model.summary()

# data
x = []
y = []
for i in range(1000):
    x_0 = random.randint(0, 100)
    x_1 = random.randint(0, 100)
    x.append((x_0, x_1))
    y.append(x_0 + x_1)

# split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

print(f'weights before training: {dense_layer.get_weights()}')

# training
model.compile(loss='mse', metrics=['MeanSquaredError'])
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), callbacks=[WandbCallback()])

# weights
print(f'weights after training: {dense_layer.get_weights()}')

# inference
print(f'13 + 7 = {model.predict([(13, 7), ])}')

