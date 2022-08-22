# README Ordner `functions`
__________________________

## Überblick


## Erläuterung der Scripte
__________________________


### Zusammenführung HSI, THERMAL und DOM in `combine_data.py`

In `combine_data.py` werden die verschiedenen Aufnahmen (HSI, THERMAL und DOM) in ein Bild zusammengeführt. 
Als Ergebnis werden zwei Dateien mit der Endung `_combined_HSI_THERMAL_DOM.hdr` und  `_combined_HSI_THERMAL_DOM.dat`
abgespeichert.

<br>

### Erzeugung von Teilbildern in `Grid_spilt.py`
In `Grid_spilt.py` wird das in `combine_data.py` zusammengeführte Bild in Teilbilder zerlegt. 
Die Zerlegung des Gesamtbilds erfolgt durch die Anwendung eines Gitternetzes. 
Beim Abspeichern der `.hdr` und `.dat` - Dateien wird die Position des Teilbilds mit den Gitterkoordinaten festgehalten,
weshalb einer späten Zusammenführung zu dem Originalbild wieder möglich ist.

<br>

### RGB-Export von Teilbildern in `Grid_create_RGB_Pictures.py`
In `Grid_create_RGB_Pictures.py` werden die in `Grid_spilt.py` erzeugten Teilbilder für die Annotation als
RGB Bild abgespeichert.

<br>

### Teilbildern zusammenführen in `Grid_combine.py`
In `Grid_combine.py` werden Teilbilder zu einem Gesamtbild anhand der Grid-Position zusammengeführt und abgespeichert.

