import wandb
from build_samples import import_samples_for_baseline
from wandb.keras import WandbCallback
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, Dropout


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# login
wandb.init(project="NN_for_pixels", entity="pds_project", name='GPU')

# import data
X_train, y_train, X_test, y_test = import_samples_for_baseline(label_mapping='Ohne_Auto_See')

dropout_rate = 0.05

# define the keras model
input_x = Input(X_train.shape[1], name='input_layer')
hl1 = Dense(512, activation='relu')(input_x)
drop1 = Dropout(rate=dropout_rate)(hl1)
hl2 = Dense(256, activation='relu')(drop1)
drop2 = Dropout(rate=dropout_rate)(hl2)
hl3 = Dense(128, activation='relu')(drop2)
drop3 = Dropout(rate=dropout_rate)(hl3)
hl4 = Dense(64, activation='relu')(drop3)
drop4 = Dropout(rate=dropout_rate)(hl3)
hl5 = Dense(32, activation='relu')(drop4)
output = Dense(y_train.shape[1], activation='softmax')(hl5)

model = Model(inputs=input_x, outputs=output, name='model_nn_for_pixel')

# compile the keras model
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=10000, batch_size=10000, validation_data=(X_test, y_test), callbacks=[WandbCallback()])
