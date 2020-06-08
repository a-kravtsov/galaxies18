import matplotlib.pylab as plt
import numpy as np
import scipy.optimize as opt
from matplotlib.colors import LogNorm

from astroML.datasets import fetch_sdss_spectrum
from scipy.interpolate import interp1d
from scipy.integrate import simps
from PIL import Image
import os
import sys
python_version = sys.version_info[0]

if python_version >=3: 
    from .setup.setup import image_home_dir, sdss_filter_dir
    from .fetch_sdss_image import fetch_sdss_image
    from .cosmology import d_l
    from .calc_kcor import calc_kcor 

else:
    from setup.setup import image_home_dir, sdss_filter_dir
    from fetch_sdss_image import fetch_sdss_image
    from setup import setup 
    from cosmology import d_l
    from calc_kcor import calc_kcor 


d_H = 2997.92 # c/(100 km/s/Mpc) 

def plot_pretty(dpi=175,fontsize=9):
    # import pyplot and set some parameters to make plots prettier
    import matplotlib.pyplot as plt

    plt.rc("savefig", dpi=dpi)
    plt.rc("figure", dpi=dpi)
    plt.rc('text', usetex=True)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', direction='in') 
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=5) 
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5) 
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [0.5, 1.1])

    return

def plot_Mz(x,y, xlim=[0,1], ylim=[0,1], nxbins=151, nybins=151, 
            xlabel='x', ylabel='y', Om0=0.3, OmL=0.7, h=0.7, savefig=None):
    fig, ax = plt.subplots(figsize=(3., 3.))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    #
    plot_2d_dist(x,y, xlim,ylim,nxbins,nybins,xlabel=xlabel,ylabel=ylabel,fig_setup=ax)

    zd = np.linspace(x.min(), x.max(),100)
    dld = d_l(zd, Om0, OmL) * d_H / h
    # this k-correction is not design for z>0.5, so limit the z for correction calculation
    grzd = np.ones_like(zd); grzd = 0.8*grzd
    kcorr = calc_kcor('r', zd, 'g - r', grzd)

    # main galaxy magnitude limit
    Mlim = 17.77 - 5.*np.log10(dld*1.e5) 
    
    mcandle = -23.*np.ones_like(zd)
    mcandlevol = -23. - 1.3*zd
    mcandlevolkcorr = -23. - 1.3*zd + kcorr

    plt.plot(zd, Mlim, '--', c='m', lw=2., label=r'$\mathrm{SDSS\ spec.\ sample}\ m_{\rm lim}=17.77$')
    plt.plot(zd, mcandle, '--', c='r', lw=2., label=r'$\mathrm{st.\ candle}\ M_r=-23$')
    plt.plot(zd, mcandlevol, '--', c='darkorange', lw=1.5, label=r'$\mathrm{+evo\ correction}$')
    plt.plot(zd, mcandlevolkcorr, '--', c='y', lw=1., label=r'$\mathrm{+evo+k\ corrections}$')

    if savefig != None:
        plt.savefig(savefig,bbox_inches='tight')
    plt.legend(loc='upper right', fontsize=8, frameon=False)
    plt.show()
    return

def plot_mz(x, y, xlim=[0.1,], ylim=[0,1], nxbins=151, nybins=151, xlabel='x', ylabel='y', Om0=0.3, OmL=0.7, h=0.7, savefig=None):
    """
    plot a binned histogram showing distribution of galaxies in the apparent magnitude-redshift plane
    along with the expected m-z relation expected for a "standard candle" (object of fixed luminosity)
    + the same with correction for evolution (e-korrection) and spectrum redshift (k-correction)
    """
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    
    plot_2d_dist(x,y, xlim,ylim, nxbins, nybins, xlabel=xlabel, ylabel=ylabel, fig_setup=ax)
    
    # plot curve showing m-z relation for the constant luminosity ignoring K correction
    # set cosmology to the best values from 9-year WMAP data

    zd = np.linspace(x.min(), x.max(), 100)
    dlum = d_l(zd, Om0, OmL) * d_H / h
    
    # and k-correction using polynomial approximations of Chilingarian et al. 2010
    # see http://kcor.sai.msu.ru/getthecode/
    # this k-correction is not designed for z>0.6, so limit the z for correction calculation
    grzd = np.ones_like(zd); grzd = 0.8*grzd
    kcorr = calc_kcor('r', zd, 'g - r', grzd)

    mcandle = -23. + 5.*np.log10(dlum*1.e6/10.)
    mcandlevol = -23. + 5.*np.log10(dlum*1.e6/10.) - 1.3*zd
    mcandlevolkcorr = -23. + 5.*np.log10(dlum*1.e6/10.) - 1.3*zd + kcorr
    plt.plot(zd, mcandle, '--', c='r', lw=2., label=r'$\mathrm{st.\ candle}\ M_r=-23$')
    plt.plot(zd, mcandlevol, '--', c='darkorange', lw=1.5, label=r'$\mathrm{+evo\ correction}$')
    plt.plot(zd, mcandlevolkcorr, '--', c='y', lw=1., label=r'$\mathrm{+evo+k\ corrections}$')
    
    if savefig != None:
        plt.savefig(savefig,bbox_inches='tight')
        
    plt.legend(loc='lower right', fontsize=8, frameon=False)
    plt.show()
    return

def plot_image_spec_sdss_galaxy(sdss_obj, save_figure=False):
    #plot image and spectrum of a specified SDSS galaxy
    # input sdss_obj = individual SDSS main galaxy data base entry
    # save_figure specifies whether PDF of the figure will be saved in fig/ subdirectory for the record
    #
    # define plot with 2 horizonthal panels of appropriate size
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 2.))
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
        data_home = sdss_filter_dir()
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
    grcatcol = sdss_obj['modelMag_g']-sdss_obj['modelMag_r'] 
    print("computed g-r = %.2f"%(color[1]-color[2]))
    print("catalog g-r = %.2f"%grcatcol)

    ax1.plot(lam, spectrum, '-k', lw=0.5, label=r'$(g-r)=%.2f$'%grcatcol )
    ax1.set_xlabel(r'$\lambda {(\rm \AA)}$')
    ax1.set_ylabel(r'$\mathrm{norm.\ specific\ flux}\ \ \ f_{\lambda}$')
    #ax.set_title('Plate = %(plate)i, MJD = %(mjd)i, Fiber = %(fiber)i' % locals())
    #ax.set_title('%s' % objects[obj_]['name'])
    ax1.legend(frameon=False)

    if save_figure: 
        plt.savefig('fig/gal_img_spec'+'_'+str(sdss_obj['objID'])+'.pdf',bbox_inches='tight')

    plt.subplots_adjust(wspace = 0.2)
    plt.show()
    return
    
def plot_image_spec_sdss_galaxy_meert(sdss_obj, gr, save_figure=False):
    #plot image and spectrum of a specified SDSS galaxy
    # input sdss_obj = individual SDSS main galaxy data base entry
    # save_figure specifies whether PDF of the figure will be saved in fig/ subdirectory for the record
    #
    # define plot with 2 horizonthal panels of appropriate size
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 2.))
    # set an appropriate font size for labels
    plt.rc('font',size=8)

    #get RA and DEC of the galaxy and form the filename to save SDSS image
    RA = sdss_obj['ra']; DEC = sdss_obj['dec']; scale=0.25
    outfile = image_home_dir()+str(sdss_obj['objid'])+'.jpg'
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
    grcatcol = gr
    print("computed g-r = %.2f"%(color[1]-color[2]))
    print("catalog g-r = %.2f"%grcatcol)

    ax1.plot(lam, spectrum, '-k', lw=0.5, label=r'$(g-r)=%.2f$'%grcatcol )
    ax1.set_xlabel(r'$\lambda {(\rm \AA)}$')
    ax1.set_ylabel(r'$\mathrm{norm.\ specific\ flux}\ \ \ f_{\lambda}$')
    #ax.set_title('Plate = %(plate)i, MJD = %(mjd)i, Fiber = %(fiber)i' % locals())
    #ax.set_title('%s' % objects[obj_]['name'])
    ax1.legend(frameon=False)

    if save_figure: 
        plt.savefig('fig/gal_img_spec'+'_'+str(sdss_obj['objID'])+'.pdf',bbox_inches='tight')

    plt.subplots_adjust(wspace = 0.2)
    plt.show()
    return

if python_version >= 3:
    import urllib.request
else:
    import cStringIO, urllib
    
def fetch_sdss_image(outfile, RA, DEC, scale=0.2, width=400, height=400):
    """Fetch the image at the given RA, DEC from the SDSS server"""
    url = ("http://skyservice.pha.jhu.edu/DR8/ImgCutout/"
           "getjpeg.aspx?ra=%.8f&dec=%.8f&scale=%.2f&width=%i&height=%i"
           % (RA, DEC, scale, width, height))

    if python_version>=3: 
        urllib.request.urlretrieve(url, outfile)
    else:
        fd = cStringIO.StringIO(urllib.urlopen(url).read())
        im = Image.open(fd)
        im.save(outfile)

def sdss_img_collage(objs, ras, decs, nrow, ncol, npix, scale, savefig=None):
    from PIL import Image
    if python_version >=3: 
        from .fetch_sdss_image import fetch_sdss_image
        from .setup.setup import image_home_dir
    else:
        from fetch_sdss_image import fetch_sdss_image
        from setup.setup import image_home_dir
        
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol, nrow))

    # Check that PIL is installed for jpg support
    if 'jpg' not in fig.canvas.get_supported_filetypes():
        raise ValueError("PIL required to load SDSS jpeg images")

    for _obj, ra, dec, ax in zip(objs, ras, decs, axs.flatten()):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        outfile = image_home_dir()+str(_obj)+'.jpg'
        fetch_sdss_image(outfile, ra, dec, scale=scale, width=npix, height=npix)
        I = Image.open(outfile)
        ax.imshow(I,origin='lower')
        ax.set_aspect('auto')

    #plt.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0)
    if savefig != None:
        plt.savefig(savefig,bbox_inches='tight')
    plt.show()
        
    
def conf_interval(x, pdf, conf_level):
    return np.sum(pdf[pdf > x])-conf_level
    
def fetch_image(objid, ra, dec, scale, npix):
    from PIL import Image
    import sys
    python_version = sys.version_info[0]

    if python_version >=3: 
        from .fetch_sdss_image import fetch_sdss_image
        from .setup.setup import image_home_dir
    else:
        from fetch_sdss_image import fetch_sdss_image
        from setup.setup import image_home_dir

    outfile = image_home_dir()+str(objid)+str(np.round(scale,2))+'.jpg'
    if not os.path.isfile(outfile):
        fetch_sdss_image(outfile, ra, dec, scale=scale, width=npix, height=npix)
    return Image.open(outfile)

def plot_sdss_collage_with_2d_dist(objs=None, ras=None, decs=None, weights=None, 
                                   xs=None, ys=None, xlab='x', ylab='y', 
                                   xlims=None, ylims=None, nrows=3, ncols=3, npix = 150, 
                                   show_axis=False, show_xaxis=False, show_yaxis=False, facecolor='white',
                                   clevs = None, ncont_bins = None, 
                                   rnd_seed=None, dA = None, kpc_per_npix = 25, outfile=None):
                                   
    """
    Plot a collage of SDSS images (downloading them if needed) for a list of SDSS objects
    with declinations decs and right ascensions ras ordered on a grid by properties xs and ys (x and y-axis)
    + plot contours of object distribution on top of images if needed
    
    Parameters
    --------------------------------------------------------------------------------------------------------
    objs: array_like
           list of SDSS objIDs 
    ras:  array_like
           list of R.A.s of objects in objs of the same size as objs
    decs: array_like 
          list of DECs of objects in objs of the same size as objs
    show_axis: bool
          show axis with labels if True
    xs: array_like
        property of objects in objs to order along x
    ys: array_like
        property of objects in objs to order along x
    """

    arcsec_to_rad = np.pi/180./3600.
    samp_dist = 0.2
    #axes ranges and number of images along each axis
    if xlims is None:
        xmin = np.min(xs); xmax = np.max(xs)
        xlims = np.array([xmin,xmax])
    if ylims is None:
        ymin = 0.95*np.min(ys); ymax = 1.05*np.max(ys)
        ylims = np.array([ymin, ymax])
        
    dxh = 0.5*np.abs(xlims[1] - xlims[0])/ncols; dyh = 0.5*np.abs(ylims[1] - ylims[0])/nrows
    
    xgrid = np.linspace(xlims[0]+dxh, xlims[1]-dxh, ncols)
    ygrid = np.linspace(ylims[0]+dyh, ylims[1]-dyh, nrows)

    fig, ax = plt.subplots(1,1,figsize=(5, 5))    
    #fig.patch.set_facecolor('white')
    ax.patch.set_facecolor(facecolor)
    if facecolor == 'black' and show_axis == True:
        ecol = 'whitesmoke'
        ax.tick_params(color=ecol, labelcolor='black', direction='in')
        for spine in ax.spines.values():
            spine.set_edgecolor(ecol)

    ax.set_xlim(xlims[0], xlims[1]); ax.set_ylim(ylims[0], ylims[1])
    if xlims[1] < 0.: ax.invert_xaxis()
    #if ylims[1] < ylims[0]: ax.invert_yaxis()

    if not show_axis:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    else:
        if show_xaxis: 
            ax.set_xlabel(xlab)
            ax.xaxis.set_visible(True)
        else: ax.xaxis.set_visible(False)
        if show_yaxis: 
            ax.set_ylabel(ylab)
            ax.yaxis.set_visible(True)
        else: ax.yaxis.set_visible(False)
        
    from itertools import product
    # Check that PIL is installed for jpg support
    if 'jpg' not in fig.canvas.get_supported_filetypes():
        raise ValueError("PIL required to load SDSS jpeg images")
    
    np.random.seed(rnd_seed)
    for xi, yi in product(xgrid, ygrid):
        inds = ((xs > xi-samp_dist*dxh) & (xs < xi+samp_dist*dxh) &
                (ys > yi-samp_dist*dyh) & (ys < yi+samp_dist*dyh))
        _objs = objs[inds]; _ras = ras[inds]; _decs = decs[inds]
        lobjs = len(_objs)
        if lobjs == 0 : continue
        if lobjs == 1: 
            iran = 0
        else:    
            iran = np.random.randint(0,lobjs-1,1)
        if dA[0] != None: 
            _dA = dA[inds]
            dAi = _dA[iran]
            img_scale = kpc_per_npix/(dAi*1.e3*npix*arcsec_to_rad)
        else:
            img_scale = 0.2
        I = fetch_image(_objs[iran],_ras[iran],_decs[iran],img_scale, npix)
        ax.imshow(I, extent=[xi-dxh, xi+dxh, yi-dyh, yi+dyh])

    ax.set_aspect(dxh/dyh)
    
    # add contours if ncont_bins is specified on input
    if ncont_bins != None:
        if clevs is None:
            raise Exception('ncont_bin is specified but contour levels clevs is not!')
            
        contours_bins = np.linspace(xlims[0], xlims[1], ncont_bins), np.linspace(ylims[0], ylims[1], ncont_bins)

        if weights is None: weights = np.ones_like(xs)

        H, xbins, ybins = np.histogram2d(xs, ys, weights=weights, bins=contours_bins)
        H = np.rot90(H); H = np.flipud(H); Hmask = np.ma.masked_where(H==0,H)
        H = H/np.sum(H)        

        X,Y = np.meshgrid(xbins,ybins) 

        lvls = []
        for cld in clevs:  
            sig = opt.brentq( conf_interval, 0., 1., args=(H,cld) )   
            lvls.append(sig)
        ax.contour(H, linewidths=np.linspace(1,2,len(lvls))[::-1], 
                    colors='whitesmoke', alpha=0.4, levels = sorted(lvls), norm = LogNorm(), 
                    extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]], interpolation='bicubic')

    # save plot if file is specified 
    if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight')
            
    plt.show()



def sgolay2d ( z, window_size, order, derivative=None):
    """
    """
    from scipy.signal import fftconvolve
    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0
    
    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')
    
    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2
    
    # exponents of the polynomial. 
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ... 
    # this line gives a list of two item tuple. Each tuple contains 
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]
    
    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])
        
    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band ) 
    # left band
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
    # right band
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z
    
    # top left corner
    band = z[0,0]
    Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = z[-1,-1]
    Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band ) 
    
    # top right corner
    band = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band ) 
    # bottom left corner
    band = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band ) 
    
    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return fftconvolve(Z, -c, mode='valid')        
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return fftconvolve(Z, -r, mode='valid')        
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return fftconvolve(Z, -r, mode='valid'), fftconvolve(Z, -c, mode='valid')        

def compute_inverse_Vmax(mags, zs, m_min=None, m_max=None, cosmomodel='WMAP9'):
    """compute inverse Vmax for a given set of galaxy magnitudes and redshifts, given magnitude limits m_min and m_max"""
    #from code.calc_kcor import calc_kcor 
    #ihz = [z>0.9]; zk = z; zk[ihz] = 0.9
    #kcorr = calc_kcor('r', zk, 'g - r', grm)
    from colossus.cosmology import cosmology
    # set cosmology to the best values from 9-year WMAP data
    cosmo = cosmology.setCosmology(cosmomodel)

    # compute luminosity and angular distances
    d_L = cosmo.luminosityDistance(zs)/cosmo.h

    # absolute magnitude in the r-band corrected for extinction
    Mabs = mags - 5.0*np.log10(d_L/1e-5) #- extm + 1.3*zs - kcorr; 

    # the sample magnitude limit is defined using Petrosian magnitude, so this is what we need to use to compute Vmax
    # we need to compute at what distance this galaxy would have limiting magnitude mlim, 
    # then compute Vmax using this distance, assuming flat cosmology (not a big deal at these low z): Vmax=d_M^3(z); d_M=d_L/(1+z) 
    d_Mmax = 1.e-5*np.power(10.,0.2*(m_max-Mabs))/(1.+zs)
    d_Mmin = 1.e-5*np.power(10.,0.2*(m_min-Mabs))/(1.+zs)

    vmaxi = 1.0/(np.power(d_Mmax,3.0) - np.power(d_Mmin,3.0))
    return vmaxi
    
def plot_2d_dist(x,y, xlim,ylim, nxbins,nybins, cmin=1.e-4, cmax=1.0, log=False, weights=None, xlabel='x',ylabel='y', clevs=None, smooth=None, fig_setup=None, savefig=None):
    """
    log = specifies whether logged quantities are passed to be plotted on log-scale outside this routine
    """
    if fig_setup is None:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
    else:
        ax = fig_setup
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        #ax.set_xlim(xlim); ax.set_ylim(ylim)

    if xlim[1] < 0.: ax.invert_xaxis()

    if weights is None: weights = np.ones_like(x)
    H, xbins, ybins = np.histogram2d(x, y, weights=weights, bins=(np.linspace(xlim[0], xlim[1], nxbins),np.linspace(ylim[0], ylim[1], nybins)))
    
    H = np.rot90(H); H = np.flipud(H); 
             
    #X,Y = np.meshgrid(xbins,ybins) 
    X,Y = np.meshgrid(xbins[:-1],ybins[:-1]) 
    if smooth is not None:
        if ( np.size(smooth) < 2):
            raise Exception("smooth needs to be an array of size 2 containing 0=SG window size, 1=SG poly order");
        H = sgolay2d( H, window_size=smooth[0], order=smooth[1])

    H = H/np.sum(H)        
    Hmask = np.ma.masked_where(H==0,H)
    
    if log:
        X = np.power(10.,X); Y = np.power(10.,Y)

    pcol = ax.pcolormesh(X, Y,(Hmask), vmin=cmin*np.max(Hmask), vmax=cmax*np.max(Hmask), cmap=plt.cm.BuPu, norm = LogNorm(), linewidth=0., rasterized=True)
    pcol.set_edgecolor('face')

    if clevs is not None:
        lvls = []
        for cld in clevs:  
            sig = opt.brentq( conf_interval, 0., 1., args=(H,cld) )   
            lvls.append(sig)
        
        ax.contour(X, Y, H, linewidths=(1.0,0.75, 0.5, 0.25), colors='black', levels = sorted(lvls), 
                norm = LogNorm(), extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]])
    if savefig:
        plt.savefig(savefig,bbox_inches='tight')
    if fig_setup is None:
        plt.show()
    return


def plot_2d_dist2(x,y, xlim,ylim, nxbins,nybins, cmin=1.e-4, cmax=1.0, log=False, weights=None, xlabel='x',ylabel='y', clevs=None, smooth=None, fig_setup=None, savefig=None):
    """
    log = specifies whether logged quantities are passed to be plotted on log-scale outside this routine
    """
    if fig_setup is None:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
    else:
        ax = fig_setup

    if xlim[1] < 0.: ax.invert_xaxis()

    if weights is None: weights = np.ones_like(x)
    H, xbins, ybins = np.histogram2d(x, y, weights=weights, bins=(np.linspace(xlim[0], xlim[1], nxbins),np.linspace(ylim[0], ylim[1], nybins)))
    
    H = np.rot90(H); H = np.flipud(H); 
             
    #X,Y = np.meshgrid(xbins,ybins) 
    X,Y = np.meshgrid(xbins[:-1],ybins[:-1]) 
    if smooth is not None:
        if ( np.size(smooth) < 2):
            raise Exception("smooth needs to be an array of size 2 containing 0=SG window size, 1=SG poly order");
        H = sgolay2d( H, window_size=smooth[0], order=smooth[1])

    H = H/np.sum(H)        
    Hmask = np.ma.masked_where(H==0,H)
    
    if log:
        X = np.power(10.,X); Y = np.power(10.,Y)

    pcol = ax.pcolormesh(X, Y,(Hmask), vmin=cmin*np.max(Hmask), vmax=cmax*np.max(Hmask), cmap=plt.cm.BuPu, norm = LogNorm(), linewidth=0., rasterized=True)
    pcol.set_edgecolor('face')

    if clevs is not None:
        lvls = []
        for cld in clevs:  
            sig = opt.brentq( conf_interval, 0., 1., args=(H,cld) )   
            lvls.append(sig)
        
        ax.contour(X, Y, H, linewidths=(1.0,0.75, 0.5, 0.25), colors='black', levels = sorted(lvls), 
                norm = LogNorm(), extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]])
    if savefig:
        plt.savefig(savefig,bbox_inches='tight')
    if fig_setup is None:
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

    import py_compile
    from setup import setup
    py_compile.compile(os.path.join(setup.code_home_dir(),'fetch_sdss_image.py'))
    
    