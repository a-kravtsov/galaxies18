import os
import numpy as np
import sys
python_version = sys.version_info[0]

try:
    from astropy.io import fits
except ImportError:
    print("astropy.io is required to read fits file! but does not seem to be available...")
    
if python_version >=3: 
    from .setup.setup import data_home_dir
else:
    from setup.setup import data_home_dir
    
def read_sdss_fits(data_file=None):
    """Loader for SDSS Galaxies w
    
    Returns
    -------
    data : recarray, shape = (327260,)
        record array containing pipeline parameters


    """

    if not os.path.exists(data_file):
        print("***error! data file", data_file," does not exist!")
        return 0
    hdulist = fits.open(data_file)
    return np.asarray(hdulist[1].data)

def read_meert_catalog(phot_type=None):
    """Loader for the Meert et al. 2015 catalog of improved photometric measurements
    for galaxies in the SDSS DR7 main galaxy catalog 
    input: phot_type - integer corresponding to the photometry model fit type from the catalog
        1=best fit, 2=deVaucouleurs, 3=Sersic, 4=DeVExp, 5=SerExp
    returns combined structured array with all the data
    """

    if (phot_type < 1) or (phot_type > 5):
        raise Exception('unsupported type of Meert et al. photometry: %d, choose number between 1 and 5')

    datadir = data_home_dir()
    datameertnonpar = datadir+r'/Meert2015_v2/UPenn_PhotDec_nonParam_rband.fits'
    datameertnonparg = datadir+r'/Meert2015_v2/UPenn_PhotDec_nonParam_gband.fits'
    datameert = datadir+r'/Meert2015_v2/UPenn_PhotDec_Models_rband.fits'
    datasdss = datadir+r'/Meert2015_v2/UPenn_PhotDec_CAST.fits'
    datasdssmodels = datadir+r'/Meert2015_v2/UPenn_PhotDec_CASTmodels.fits'
    datameertg = datadir+r'/Meert2015_v2/UPenn_PhotDec_Models_gband.fits'
    datamorph = datadir+r'/Meert2015_v2/UPenn_PhotDec_H2011.fits' # morphology probabilities from Huertas-Company et al. 2011

    # mdata tables: 1=best fit, 2=deVaucouleurs, 3=Sersic, 4=DeVExp, 5=SerExp
    mdata = fits.open(datameert)[phot_type].data
    mdatag = fits.open(datameertg)[phot_type].data
    mnpdata = fits.open(datameertnonpar)[1].data
    mnpdatag = fits.open(datameertnonparg)[1].data
    sdata = fits.open(datasdss)[1].data
    phot_r = fits.open(datasdssmodels)[1].data
    morph = fits.open(datamorph)[1].data

    # eliminate galaxies with bad photometry
    fflag = mdata['finalflag']
    print("%d galaxies in Meert et al. sample initially"%np.size(fflag))

    def isset(flag, bit):
        """Return True if the specified bit is set in the given bit mask"""
        return (flag & (1 << bit)) != 0
        
    # use minimal quality cuts and flags recommended by Alan Meert
    igood = [(phot_r['petroMag'] > 0.) & (phot_r['petroMag'] < 100.) & (mnpdata['kcorr'] > 0) &
             (mdata['m_tot'] > 0) & (mdata['m_tot'] < 100) &
             (isset(fflag, 1) | isset(fflag, 4) | isset(fflag, 10) | isset(fflag, 14))]

    sdata = sdata[igood]; phot_r = phot_r[igood]; mdata = mdata[igood]
    mnpdata = mnpdata[igood]; mdatag = mdatag[igood]; mnpdatag = mnpdatag[igood]; morph = morph[igood]

    return sdata, mdata, mnpdata, phot_r, mdatag, mnpdatag, morph

def read_alfalfa(aafile):
    """ read ALFALFA catalog data file from: 
        http://egg.astro.cornell.edu/alfalfa/data/a40files/a40.datafile1.txt
    """ 
    ancat = []; aname = []; vhelio = []; w50 = []; ew50 = []; HIflux = []; eHIflux = []
    dist = []; lMHI = []; HIcode = []; Ocode = [];
    with open(aafile) as input_file:
        lines = input_file.readlines()
        for line in lines[84:]:
            ancatd  = line[0:6]; anamed = line[7:16]
            vheliod = line[48:53]; w50d = line[54:57]; ew50d = line[58:61]
            HIfluxd = line[63:70]; eHIfluxd = line[71:75]
            HIcoded = line[102:103]; Ocoded = line[104:105]
            # handle missing distances and masses 
            if line[90:95] == '     ':
                distd = -1000.0; lMHId = -1000.0
            else:
                distd   = float(line[90:95]); lMHId = float(line[96:101])  

            ancat.append(ancatd); aname.append(anamed)
            vhelio.append(vheliod); w50.append(w50d); ew50.append(ew50d)
            HIflux.append(HIfluxd); eHIflux.append(eHIfluxd); 
            dist.append(distd); lMHI.append(lMHId); HIcode.append(HIcoded); Ocode.append(Ocoded)
            
    aalist = np.zeros((len(ancat),), dtype = [('AGCnr','i8'),('Name','a8'),
                      ('Vhelio','i8'),('W50','i8'),('errW50','i8'),
                      ('HIflux','f8'),('errHIflux','f8'),('Dist','f8'),
                      ('logMsun','f8'),('HIcode','i8'),('Ocode','a8') ])
    aalist['AGCnr'] = np.array(ancat); aalist['Name'] = np.array(aname); 
    aalist['Vhelio'] = np.array(vhelio); aalist['W50'] = np.array(w50); aalist['errW50'] = np.array(ew50);
    aalist['HIflux'] = np.array(HIflux); aalist['errHIflux'] = np.array(eHIflux);
    aalist['Dist'] = np.array(dist); 
    aalist['logMsun'] = np.array(lMHI)
    aalist['HIcode'] = np.array(HIcode)
    aalist['Ocode'] = np.array(Ocode)

    return aalist

def read_alfalfa_sdss_crosslist(aasfile):
    """ read ALFALFA SDSS cross listing table from file: 
        http://egg.astro.cornell.edu/alfalfa/data/a40files/a40.datafile3.txt
    """ 
    ancats = []; sdss_photo_objID = []; sdss_spec_objID = [];
    modelmag = []; ur = []; zsdss = [] 
    
    with open(aasfile) as input_file:
        lines = input_file.readlines()
        for line in lines[74:]:
            ancatsd = line[0:6]
            sdss_photoid = line[11:29]; sdss_specid = line[30:48]
            modelmagd = line[49:54]; urcol = line[55:60]; 
            if line[61:68] == '       ':
                # assign easy to weed out value to galaxies w/o z
                zsdssd = 1000.0
            else:
                zsdssd = float(line[61:68]) 
            ezsdssd = line[69:76]
            ancats.append(ancatsd); sdss_photo_objID.append(sdss_photoid); sdss_spec_objID.append(sdss_specid)
            modelmag.append(modelmagd); ur.append(urcol); zsdss.append(zsdssd) 
            dummy = 0*zsdssd        

    sdsslist = np.zeros((len(ancats),), dtype = [('AGCnr','i'),('PhotoObjID','a18'),('SpectObjID','a18'),('rmodelmag','f'),('uminusr','f'),('zsdss','f'),('logMsun','f')])
    sdsslist['AGCnr']=np.array(ancats); sdsslist['PhotoObjID']=np.array(sdss_photo_objID); sdsslist['SpectObjID']=np.array(sdss_spec_objID)
    sdsslist['rmodelmag'] = np.array(modelmag); 
    sdsslist['uminusr'] = np.array(ur); 
    sdsslist['zsdss'] = np.array(zsdss)
    sdsslist['logMsun'] = np.array(dummy)
    return sdsslist

def alfalfa_sdss_crossmatch(aalist, sdsslist):
    # crossmatch them 
    # match sdsslist to aalist
    imatch = np.in1d(aalist['AGCnr'],sdsslist['AGCnr'])
    # match aalist to sdsslist
    imatch2 = np.in1d(sdsslist['AGCnr'],aalist['AGCnr'])
    aamatch = aalist[imatch]
    sdssmatch = sdsslist[imatch2]
    # carry out some basic  cuts
    aatf = aamatch[aamatch['Dist']>1.0]
    sdsstf = sdssmatch[aamatch['Dist']>1.0]
    aatf = aamatch[aamatch['Dist']<500.0]
    sdsstf = sdssmatch[aamatch['Dist']<500.0]

    aatf = aatf[sdsstf['zsdss']<10.]
    sdsstf = sdsstf[sdsstf['zsdss']<10.]
    aatf = aatf[sdsstf['zsdss']>0.0001]
    sdsstf = sdsstf[sdsstf['zsdss']>0.0001]
    return aatf, sdsstf


if __name__ == '__main__':
    
    import py_compile

    py_compile.compile(os.path.join(setup.code_home_dir(),'read_data.py'))

    sd, md, mnpd, phot_r, mdg, mnpdg, morph = read_meert_catalog(phot_type=3)
