from spectral.algorithms.algorithms import TrainingClass, TrainingClassSet
import matplotlib.pyplot as plt
import spectral as spy

"""
Spectral Python (SPy) Functions
https://www.spectralpython.net/
"""

# function to read Envi-files
# img = envi.open(file='filename.hdr', image='filename.dat')

# function to convert Envi-file img in array (relates to above img)
# img_arr = img.load()

##############################################################################################

# example with spectral_python_files

# read image
img = spy.open_image('../../data/spectral_python_files/92AV3C.lan').load()

##############################################################################################

# example k-means Clustering
(m, c) = spy.kmeans(img, 20, 30)

# show spectrogram of cluster centers
plt.figure()

for i in range(c.shape[0]):
    plt.plot(c[i])

plt.grid()

##############################################################################################

# Classification

# import label mask
gt = spy.open_image('../../data/spectral_python_files/92AV3GT.GIS').read_band(0)

# plot label mask
v = spy.imshow(classes=gt)
plt.imshow(v.class_rgb)
plt.show()

# create trainclass
classes = spy.create_training_classes(img, gt)

# Gaussian Maximum Likelihood Classification
gmlc = spy.GaussianClassifier(classes)

# plot results
clmap = gmlc.classify_image(img)
v = spy.imshow(classes=clmap)
plt.imshow(v.class_rgb)
plt.show()

# compare labeled pixels with predicted labeling
gtresults = clmap * (gt != 0)
v = spy.imshow(classes=gtresults)
plt.imshow(v.class_rgb)
plt.show()

# compare results difference in labels
gterrors = gtresults * (gtresults != gt)
v = spy.imshow(classes=gterrors)
plt.imshow(v.class_rgb)
plt.show()

# # Trainingsdaten anlegen
#
# # Set für alle Trainingsdaten anlegen
# train_set = TrainingClassSet()
# #training_data = create_training_classes(image=arr_img, class_mask=new_array) Unterschied noch herausfinden
#
# # Trainingsdaten anlegen
# training_data = TrainingClass(image=arr_img, mask=new_array)
#
# # Berechnet Statistiken der Klassen
# # training_data.calc_stats()
# # print(training_data.stats.mean)
#
# # Anzahl annotierter Pixel bestimmen
# print(training_data.size())
#
# # Trainingsdaten dem Trainingsdaten-Set hinzufügen
# train_set.add_class(training_data)
#
# # Ausgabe wie viele Trainingsbilder in Trainingsdaten-Set sind
# print(train_set.__len__())
#
# # Erster Test
# clf = GaussianClassifier(training_data=train_set, min_samples=1)
#
# result = clf.classify_image(image=arr_img)
