from spectral.algorithms.algorithms import TrainingClass, TrainingClassSet
from code.find_path_nextcloud import find_path_nextcloud
from spectral import *
import numpy as np


# Pfad Nextcloud bestimmen
path_nextcloud = find_path_nextcloud()

# Pfade definieren
path_hdr = path_nextcloud + 'Daten_Gyrocopter/Oldenburg/Teilbilder/grid_1000_1000/Teilbild_Oldenburg_00000000_00000000_0_0_.hdr'
path_dat = path_hdr[:-4] + '.dat'

# Teilbild laden
img = envi.open(file=path_hdr, image=path_dat)

# Laden der Bildinformationen
arr_img = img.load()

# Neues "Band" für Klasse erzeugen
new_array = np.zeros(shape=(1000,1000,1))

# Werte ersetzen
new_array[:50,:50,:] = 1
new_array[50:100,50:100,:] = 2


# Trainingsdaten anlegen

# Set für alle Trainingsdaten anlegen
train_set = TrainingClassSet()
#training_data = create_training_classes(image=arr_img, class_mask=new_array) Unterschied noch herausfinden

# Trainingsdaten anlegen
training_data = TrainingClass(image=arr_img, mask=new_array)

# Berechnet Statistiken der Klassen
# training_data.calc_stats()
# print(training_data.stats.mean)

# Anzahl annotierter Pixel bestimmen
print(training_data.size())

# Trainingsdaten dem Trainingsdaten-Set hinzufügen
train_set.add_class(training_data)

# Ausgabe wie viele Trainingsbilder in Trainingsdaten-Set sind
print(train_set.__len__())

# Erster Test
clf = GaussianClassifier(training_data=train_set, min_samples=1)

result = clf.classify_image(image=arr_img)
