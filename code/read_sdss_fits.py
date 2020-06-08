import os
import numpy as np

def read_sdss_fits(data_file=None):
    """Loader for SDSS Galaxies w
    
    Returns
    -------
    data : recarray, shape = (327260,)
        record array containing pipeline parameters


    """
    # pyfits is an optional dependency: don't import globally
    from astropy.io import fits

    if not os.path.exists(data_file):
        print("***error! data file", data_file," does not exist!")
        return 0
    hdulist = fits.open(data_file)
    return np.asarray(hdulist[1].data)
