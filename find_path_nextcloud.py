# Packages
import os

def find_path_nextcloud():
    # Bestimmung des aktuellen Pfads
    current_path = os.path.abspath(os.getcwd())

    # Der Benutzername der Endgeräte unterscheidet sich
    # Festlegen des Pfads der Nextcloud in Abhängigkeit vom Benutzername
    if 'fgrassxx' in current_path:
        path_nextcloud_fg = open('path_nextcloud_fgrassxx.txt')
        path_nextcloud = path_nextcloud_fg.read()

    elif 'timo' in current_path:
        path_nextcloud_timo = open('path_nextcloud_timo.txt')
        path_nextcloud = path_nextcloud_timo.read()

    return path_nextcloud
