#
# code extracted from the AstroML library
# and modified for this course to be stand-alone
#
from __future__ import print_function, division

import os

import numpy as np
from setup import setup 

#FILTER_URL = 'http://classic.sdss.org/dr7/instruments/imager/filters/%s.dat'


def fetch_sdss_filters(fname):
    """Loader for SDSS Filter profiles
    Parameters
    ----------
    fname : str
        filter name: must be one of 'ugriz'
    Returns
    -------
    data : ndarray
        data is an array of shape (5, Nlam)
        first row: wavelength in angstroms
        second row: sensitivity to point source, airmass 1.3
        third row: sensitivity to extended source, airmass 1.3
        fourth row: sensitivity to extended source, airmass 0.0
        fifth row: assumed atmospheric extinction, airmass 1.0
    """
    if fname not in 'ugriz':
        raise ValueError("Unrecognized filter name '%s'" % fname)

    data_home = setup.sdss_filter_dir()
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    archive_file = os.path.join(data_home, '%s.dat' % fname)

    if not os.path.exists(archive_file):
        raise ValueError("Error in fetch_sdss_filter: filter file '%s' does not exist!" % archive_file )
    
    F = open(archive_file)

    return np.loadtxt(F, unpack=True)

if __name__ == '__main__':
    for f in 'ugriz':
        fetch_sdss_filters(f)
        
    import py_compile
    py_compile.compile(os.path.join(setup.code_home_dir(),'fetch_sdss_filters.py'))

