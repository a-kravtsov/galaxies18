import matplotlib.pylab as plt
import numpy as np

from astroML.datasets import fetch_sdss_spectrum
from scipy.interpolate import interp1d
from scipy.integrate import simps

from PIL import Image
import os

from .setup.setup import setup 

def plot_image_spec_sdss_galaxy(sdss_obj, save_figure=None):
    #plot image and spectrum of a specified SDSS galaxy
    # input sdss_obj = individual SDSS main galaxy data base entry
    # save_figure specifies whether PDF of the figure will be saved for the record
    #   if save_figure != None, figure will be saved into path+filename given by string in save_figure
    # define plot with 2 horizonthal panels of appropriate size
    fig,(ax0,ax1) = plt.subplots(1,2,figsize=(6,2.))
    # set an appropriate font size for labels
    plt.rc('font',size=8)

    #get RA and DEC of the galaxy and form the filename to save SDSS image
    RA = sdss_obj['ra']; DEC = sdss_obj['dec']; scale=0.25
    outfile = image_home_dir()+str(sdss_obj['objID'])+'.jpg'
    fetch_sdss_image(outfile, RA, DEC, scale)
    img = Image.open(outfile)
    # do not plot pixel axis labels
    ax0.axis('off')
    ax0.imshow(img)

    # fetch SDSS spectrum using plate number, epoch, and fiber ID
    plate = sdss_obj['plate']; mjd = sdss_obj['mjd']; fiber = sdss_obj['fiberID']
    spec = fetch_sdss_spectrum(plate, mjd, fiber)

    # normalize spectrum for plotting
    spectrum = 0.5 * spec.spectrum  / spec.spectrum.max()
    lam = spec.wavelength()
    text_kwargs = dict(ha='center', va='center', alpha=0.5, fontsize=10)

    # set axis limits for spectrum plot
    ax1.set_xlim(3000, 10000)
    ax1.set_ylim(0, 0.6)

    color = np.zeros(5)
    for i, f, c, loc in zip([0,1,2,3,4],'ugriz', 'bgrmk', [3500, 4600, 6100, 7500, 8800]):
        data_home = setup.sdss_filter_dir()
        archive_file = os.path.join(data_home, '%s.dat' % f)
        if not os.path.exists(archive_file):
            raise ValueError("Error in plot_img_spec_sdss_galaxy: filter file '%s' does not exist!" % archive_file )
        F = open(archive_file)
        filt = np.loadtxt(F, unpack=True)
        ax1.fill(filt[0], filt[2], ec=c, fc=c, alpha=0.4)
        fsp = interp1d(filt[0],filt[2], bounds_error=False, fill_value=0.0)
        ax1.text(loc, 0.03, f, color=c, **text_kwargs)
        # compute magnitude in each band using simple Simpson integration of spectrum and filter using eq 1.2 in the notes
        fspn = lam*fsp(lam)/simps(fsp(lam)/lam,lam)
        lamf = fspn; #lamf = lamf.clip(0.0)
        specf = spec.spectrum * lamf
        color[i] = -2.5 * np.log10(simps(specf,lam)) 
        
    # print estimated g-r color for the object
    grcol = color[1]-color[2]
    grcatcol = sdss_obj['modelMag_g']-sdss_obj['modelMag_r'] 
    print("computed g-r = %.2f"%(color[1]-color[2]))
    print("catalog g-r = %.2f"%grcatcol)

    xdum = 0.0; ydum = 0.0
    ax1.scatter(xdum, ydum)
    ax1.plot(lam, spectrum, '-k', lw=0.5, label=r'$(g-r)_{\rm cat}=%.2f; \ \ (g-r)_{\rm spec}=%.2f$'%(grcatcol,grcol) )
    ax1.set_xlabel(r'$\lambda {(\rm \AA)}$')
    ax1.set_ylabel(r'$\mathrm{norm.\ specific\ flux}\ \ \ f_{\lambda}$')
    #ax.set_title('Plate = %(plate)i, MJD = %(mjd)i, Fiber = %(fiber)i' % locals())
    #ax.set_title('%s' % objects[obj_]['name'])
    ax1.legend(frameon=False,fontsize=6)

    if save_figure !=None: 
        plt.savefig(save_figure,bbox_inches='tight')

    plt.subplots_adjust(wspace = 0.2)
    plt.show()
    return
    
if __name__ == '__main__':
    from read_sdss_fits import read_sdss_fits
    from setup import data_home_dir

    # read fits file with the SDSS DR8 main spectroscopic sample
    data = read_sdss_fits(data_home_dir()+'SDSSspecgalsDR8.fit')

    z_max = 0.04

    # redshift cut
    sdata = data[data['z'] < z_max]
    mr = sdata['modelMag_r']
    gr = sdata['modelMag_g'] - sdata['modelMag_r']
    r50 = sdata['petroR50_r']
    sb = mr - 2.5*np.log10(0.5) + 2.5*np.log10(np.pi*(r50)**2)

    
    from random import randint
    import numpy as np

    # select a random galaxy from sdata
    iran = randint(0,np.size(sdata)-1)
    randobj = sdata[iran]

    # plot its image and spectrum
    plot_image_spec_sdss_galaxy(randobj)

    import py_compile
    from setup import setup
    py_compile.compile(os.path.join(setup.code_home_dir(),'fetch_sdss_image.py'))
    
    