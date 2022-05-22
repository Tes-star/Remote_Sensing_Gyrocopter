# Packages
import os


def find_path_nextcloud():
    # Bestimmung des aktuellen Pfads
    current_path = os.path.abspath(os.getcwd())
    path_nextcloud = None

    # Der Benutzername der Endgeräte unterscheidet sich
    # Festlegen des Pfads der Nextcloud in Abhängigkeit vom Benutzername
    if 'fgrassxx' in current_path:
        path_nextcloud_fg = open('../Daten_einlesen/path_nextcloud_fgrassxx.txt')
        path_nextcloud = path_nextcloud_fg.read()

    if 'Timo' in current_path:
        path_nextcloud_timo = open('../Daten_einlesen/path_nextcloud_timo.txt')
        path_nextcloud = path_nextcloud_timo.read()

    if 'vdwti' in current_path:
        path_nextcloud_timo = open('../Daten_einlesen/path_nextcloud_vdwti.txt')
        path_nextcloud = path_nextcloud_timo.read()

    return path_nextcloud
