import numpy as np
from spectral import envi


def combine_image_bands(path_data: str,
                        hdr_file_hsi: str, dat_file_hsi: str,
                        hdr_file_thermal: str, dat_file_thermal: str,
                        hdr_file_dom: str, dat_file_dom: str,
                        export_title: str):
    """
    function which combine different data formats from one picture, requires images with same shape in length and width
    :param path_data: folder where hsi, thermal and dom pictures are saves
    :param hdr_file_hsi: filename
    :param dat_file_hsi: filename
    :param hdr_file_thermal: filename
    :param dat_file_thermal: filename
    :param hdr_file_dom: filename
    :param dat_file_dom: filename
    :param export_title: name of image with combined bands
    :return: save image with combined bands and return absolute path to it
    """

    # read hyperspectral image
    path_hdr = path_data + '/' + hdr_file_hsi
    path_dat = path_data + '/' + dat_file_hsi
    img_his = envi.open(file=path_hdr, image=path_dat)

    # read thermal image
    path_hdr = path_data + '/' + hdr_file_thermal
    path_dat = path_data + '/' + dat_file_thermal
    img_thermal = envi.open(file=path_hdr, image=path_dat)

    # read dom image
    path_hdr = path_data + '/' + hdr_file_dom
    path_dat = path_data + '/' + dat_file_dom
    img_dom = envi.open(file=path_hdr, image=path_dat)

    # load arrays
    arr_his = img_his.load()
    arr_thermal = img_thermal.load()
    arr_dom = img_dom.load()

    # add band to the last position(s)
    combined_arr = np.concatenate((arr_his, arr_thermal), -1)
    combined_arr = np.concatenate((combined_arr, arr_dom), -1)

    # add new band information in metadata
    new_bands = ['thermal', 'dom']
    arr_metadata = img_his.metadata

    for new_band in new_bands:
        arr_metadata['wavelength'].append(new_band)
        arr_metadata['band names'].append(new_band)
        arr_metadata['fwhm'].append(new_band)

    # change number of bands
    arr_metadata['bands'] = len(arr_metadata['wavelength'])

    # save combined image
    path_out = path_data + '/' + export_title + "_combined_HSI_THERMAL_DOM.hdr"

    envi.save_image(hdr_file=path_out, image=combined_arr,
                    dtype="float32", ext='.dat', interleave='bsq',
                    metadata=arr_metadata, force=True)

    return path_out
