##------------------------------------------------------------------------------
## PROJEKT:       R-Help
##------------------------------------------------------------------------------
## INSTITUT:      Hochschule Anhalt, FB AFG
## AUTOR:         S. Prokoph
## DATUM:         16.12.2020
##------------------------------------------------------------------------------
## BESCHREIBUNG:  
##
## Hilfestellungen für die Abschlussübung mit R.
## 
##------------------------------------------------------------------------------

# Arbeitsspeicher leeren
rm(list=ls())

# Bibliotheken laden (ggf. vorher installieren)
install.packages("sp")
install.packages("raster")
install.packages("rgdal")
library(raster)
library(rgdal)
library(sp)

# Pfadangabe Hyperspektralbild
path_hyp <- "C:/Users/fgrassxx/Nextcloud/Master_Data_Science/2_Semester/Projekt_Data_Science_1_SS22/Daten_Gyrocopter/Dessau/Hyperspektral/Dessau_unsigned_mosaik_all.dat"

# Rasterbild (ein Layer) erzeugen
r <- raster(path_hyp)

# Eigenschaften des Rasterbildes anzeigen
r

# Layer (Kanal1) als Bild anzeigen
plot(r, main = "Titel")

# Rasterbild (mehrere Layer -> Brick) erzeugen
hyper_image <- brick(path_hyp)

# Eigenschaften des Rasterbildes anzeigen
hyper_image

# Band 30 als Bild anzeigen
plot(hyper_image[[30]])

# RGB-Darstellung des Rasterbildes
plotRGB(hyper_image, r=28, g=21, b=8)
# CIR-Darstellung des Rasterbildes
plotRGB(hyper_image, r=48, g=28, b=21)

# Bandauswahl 
bands <- c(1:80) #wenn alle Bänder genutzt werden sollen
#bands <- c(8,15,20,29,38,47) # sinnvoller ist eine Auswahl, wenn nicht alle Bänder eingelesen werden sollen

#leerer Rasterbrick mit der gleichen räumlichen Ausdehnung wie hyper_image
mybrick <- brick(x = extent(hyper_image), 
                 nrows = nrow(hyper_image), 
                 ncols = ncol(hyper_image), 
                 crs = crs(hyper_image))

#funktion, die mybrick mit ausgewählten Bändern bestückt
makerasterbrick <- function(path_pic) {
  for(i in 1 : length(bands)){
    band_i <- raster(path_pic, band = bands[i])
    mybrick <- addLayer(mybrick, band_i)
    i <- i+1
  }
  return(mybrick)
}


#Funktion anwenden
rasterstack <- makerasterbrick(path_hyp)
rasterstack # zeigt Eigenschaften von rasterstack an

# sinnvolle Namensvergabe für ausgewählte Bänder
# names(rasterstack)[1] <- "band8"
# names(rasterstack)[2] <- "band15"
# names(rasterstack)[3] <- "band20"
# names(rasterstack)[4] <- "band29"
# names(rasterstack)[5] <- "band38"
# names(rasterstack)[6] <- "band47"
# sinnvolle Namensvergabe bei allen 80 Bändern
for(i in 1: length(bands)){
  names(rasterstack)[i] <- paste("band_",i, sep="")
}

#rasterbild als tif-Datei speichern
path_out <- "I:/LehreSP/FE-Übungen_SNAP/Übung 6/HySpex/HypDessau_.tif"
writeRaster(rasterstack, path_out, format = "GTiff")
