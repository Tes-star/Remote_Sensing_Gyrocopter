# README Ordner `functions`
__________________________

## Überblick


## Erläuterung der Scripte
__________________________

### `spectral_python_functions.py`
In `spectral_python_functions.py` ist die Programmierbibliothek Spectral Python verlinkt und unter anderem ein Beispiel 
zum Import von Bildern im envi-Format angegeben. 

<br>

### `combine_image_bands.py`
In `combine_image_bands.py` wir die Funktion erstellt, um die verschiedenen Aufnahmen (HSI, THERMAL und DOM) in ein
Bild zusammenzuführen. Als Ergebnis werden zwei Dateien mit der Endung `_combined_HSI_THERMAL_DOM.hdr` und 
`_combined_HSI_THERMAL_DOM.dat` abgespeichert.

<br>

### `split_image.py`
In `split_image.py` wird die Funktion erstellt, die ein Bild in Teilbilder zerlegt. 
Die Zerlegung des Gesamtbilds erfolgt durch die Anwendung eines Gitternetzes. 
Beim Abspeichern der `.hdr` und `.dat` - Dateien wird die Position des Teilbilds mit den Gitterkoordinaten festgehalten,
weshalb einer späten Zusammenführung zu dem Originalbild wieder möglich ist.

<br>

### `save_subimages_rgb.py`
In `save_subimages_rgb.py` wird die Funktion erstellt, die alle Teilbilder in einem Ordner für die Annotation als
RGB Bild abgespeichert.

<br>

### `convert_annotations.py`
In `convert_annotations.py`  sind die Funktionen `convert_xml_annotation_to_mask` und `convert_all_annotations` enthalten, 
die die Roboflow Annotation im xml-Format dem jeweiligen Bild als zusätzliches Label-Band hinzufügt.

<br>

### `class_ids.py`
In `class_ids.py`  sind die Funktionen enthalten, die die Klassen-IDs definieren und das Mapping von ID zum Klassenname 
oder RGB-Wert zentral definiert.

<br>

### `train_test_images.py`
In `train_test_images.py`  ist die Funktion enthalten, die die Namen der Trainings- und Testbilder definiert.

<br>

### `import_labeled_data.py`
In `import_labeled_data.py`  ist die Funktion enthalten, die alle gelabelten Teilbilder in einem DataFrame zusammenführt. 
Jede Spalte repräsentiert ein Band (plus die Spalte 'subimage_name') und jede Zeile stellt die Werte eines Pixels dar.

<br>

### `build_samples_NN_for_pixel.py`
In `build_samples_NN_for_pixel.py`  ist die Funktion enthalten, die die Trainings- und Testdaten für das Model 
NN_for_pixel importiert.

<br>

### `combine_subimages.py`
In `combine_subimages.py`  wird die Funktion erstellt, die Teilbilder zu einem Gesamtbild anhand der Grid-Position 
zusammenführt und abspeichert.

