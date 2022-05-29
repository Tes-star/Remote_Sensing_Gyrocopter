# Packages
import os
import pathlib
from pathlib import Path

def find_path_nextcloud():
    # Bestimmung des aktuellen Pfads


    current_dir = Path(os.getcwd())
    # Wenn falsche Working Directory eingestellt
    if(pathlib.PurePath(current_dir).name!='pds_gyrocopter'):
        # Ändere Working Directory
        current_dir = [p for p in current_dir.parents if p.parts[-1] == 'pds_gyrocopter'][0]
        os.chdir(current_dir)
    path_nextcloud = None
    current_path = os.path.abspath(os.getcwd())

    # Der Benutzername der Endgeräte unterscheidet sich
    # Festlegen des Pfads der Nextcloud in Abhängigkeit vom Benutzername
    if 'fgrassxx' in current_path:
        path_nextcloud_fg = open('Daten_einlesen/path_nextcloud_fgrassxx.txt')
        path_nextcloud = path_nextcloud_fg.read()

    if 'Timo' in current_path:
        path_nextcloud_timo = open('Daten_einlesen/path_nextcloud_timo.txt')
        path_nextcloud = path_nextcloud_timo.read()

    if 'vdwti' in current_path:
        path_nextcloud_timo = open('Daten_einlesen/path_nextcloud_vdwti.txt')
        path_nextcloud = path_nextcloud_timo.read()

    return path_nextcloud
