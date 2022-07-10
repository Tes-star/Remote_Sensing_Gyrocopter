from Modelling.Baselines.build_samples import import_samples_for_baseline
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
import keras
import numpy as np
import matplotlib.pyplot as plt

# load model
model = keras.models.load_model('baseline1.h5')

# import data
X_train, y_train, X_test, y_test = import_samples_for_baseline(label_mapping='Ohne_Auto_See')

# prediction
y_pred_train = model.predict(X_train)

# choose argmax
y_pred_train = y_pred_train.argmax(axis=1)
y_train = np.array(y_train).argmax(axis=1)

print(accuracy_score(y_true=y_train, y_pred=y_pred_train))

cm = confusion_matrix(y_true=y_train, y_pred=y_pred_train, labels=[0, 1, 2, 3, 4, 5])
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[0, 1, 2, 3, 4, 5])
disp.plot()

plt.show()



