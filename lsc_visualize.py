#
# read in data with positions of galaxies in the vicinity of the Milky Way
# and plot their spatial distribution in 3d
# 
# Andrey Kravtsov (July 2015)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def eq_to_3d(ra, dec):
    """
    convert equatorial to 3d coordinates
    """
    x = np.cos(ra) * np.cos(dec)
    y = np.sin(ra) * np.cos(dec)
    z = np.sin(dec)
    return x, y, z

def eq2000_to_supergal(x, y, z):
    """
    convert from the equatorial to supergalactic coordinates
    using rotation matrix multiplication. 
    for rotation matrices fror different transformation see:
        http://www.atnf.csiro.au/computing/software/gipsy/sub/skyco.c
    input: vectors of x,y,z coordinates obtained from equatorial ra, dec as
        x = np.cos(ra) * np.cos(dec)
        y = np.sin(ra) * np.cos(dec)
        z = np.sin(dec)
    return: s = array of triples of transformed coordinates
        multiply them by distance to get SGX, SGY, SGZ
    
    """
    Rot = [
      [0.3751891698,   0.3408758302,   0.8619957978],
      [-0.8982988298,  -0.0957026824,   0.4288358766],
      [0.2286750954,  -0.9352243929,   0.2703017493]]     
      
    r = np.array([x,y,z])
    s = np.zeros_like(r)
    s = np.dot(Rot,r)
    return s
    
import os
from astropy.io import fits

def read_McConnachie12(datafile):
    if not os.path.exists(datafile):
        print("***error! data file", datafile," does not exist!")
        return 0
    hdulist = fits.open(datafile)
    data = np.asarray(hdulist[1].data)
    mcgalname = data['Name']
    mstar = data['Mass']
    # ra and dec 
    RA = data['RAJ2000']; DEC = data['DEJ2000']
    # heliocentric distance in kpc
    D = data['D_MW_']
    d = D.astype(int)
    d[0] = 0; d = d.astype(float)/1000. # kpc-> Mpc
    vr = data['V_MW_']; vr[0] = 0.; 
    ra = np.zeros_like(d); dec = np.zeros_like(d)
    ms = np.zeros_like(d)
    for i in range(len(RA)):
        ras = np.fromstring(RA[i], sep=" ")
        decs = np.fromstring(DEC[i], sep=" ")
        ms[i] = 1.e6 * np.fromstring(mstar[i], sep=" ")
        if vr[i] == '    ':
            vr[i] = 0.
        else:
            vr[i] = vr[i].astype(float)

        ra[i] = (ras[0]*3600.*15.+ ras[1]*60. + ras[2])*np.pi/(180.*3600) # 360/24 = 15
        dec[i] = np.sign(decs[0])*(np.abs(decs[0])*3600.+ decs[1]*60. + decs[2])*np.pi/(180.*3600)
        #print "%3d %s %2.1f %2.1f"%(i, mcgalname[i], d[i], vr[i])
        ms[0] = 6e10 # MW
        ms[28] = 9e10 # Andromeda
        ms[79] = 1e6 # HIZSS (3A)
        ms[80] = 1e6

        #print i, mcgalname[i], d[i], vr[i], ms[i]
    vr = vr.astype(float)
    return ra, dec, d, ms

def read_Karachentsev13(datafile):
    data = np.loadtxt(datafile, skiprows=1, delimiter=',')
    # ra and dec 
    RA = data[:,0]; DEC = data[:,1]
    # heliocentric distance in kpc
    D = data[:,2]
    #VPEC = data['VPEC']
    #Bolometric Magnitude
    bm = data[:,3]
    L = [0]*len(bm)
    m = [0]*len(bm)
    #Mass calculation
    for i in range(len(bm)):
        # solar luminosity
        L[i] = pow(10, (5.48 - bm[i])/2.5) # http://www.ucolick.org/~cnaw/sun.html
        #if L[i] > 5e10:
        #    print D[i], bm[i], L[i]
    d = D.astype(float)
    ras = RA.astype(float); decs = DEC.astype(float)
    ra = ras * np.pi/180.; dec = decs * np.pi/180.
    
    #D Selection#
    dmax = 30 #MODIFY SELECTION HERE, MAX is 26.2
    Ra, Dec, Dd, M, Ll = zip(*((ra, dec, d, m, L) 
    for ra, dec, d, m, L in zip(ra, dec, d, m, L) if d<dmax))
    #print 'D Selection:', max(Dd), '<', dmax, 'Mpc'  #Sanity check
    return Ra, Dec, Dd, np.array(Ll)
    
data_home_dir = r'c:\Users\h2_sf\ocuments\Classes\a304s18\data\\'
ram, decm, dm, m = read_McConnachie12(r"c:\Users\h2_sf\Documents\Classes\a304s18\data\mcconnachie12.fits")
ram, decm, dm, m = read_Karachentsev13(r"c:\Users\h2_sf\Documents\Classes\a304s18\data\karachentsev13_BMag.txt")
xm, ym, zm = eq_to_3d(ram, decm)

#m /= np.sum(m)
m = 0.1*np.square(np.log10(m))
#m = m/m
#print m.min(), m.max()

SGXm, SGYm, SGZm = dm*eq2000_to_supergal(xm, ym, zm)
#SGXm -= np.average(SGXm, weights=1./m); SGYm -= np.average(SGYm, weights=1./m); SGZm -= np.average(SGZm, weights=1./m)
Rm = np.sqrt(SGXm**2 + SGYm**2 + SGZm**2)
Rmax = 20
isel = (Rm<Rmax)
SGXm = SGXm[isel]; SGYm = SGYm[isel]; SGZm = SGZm[isel]
m = m[isel]

# plot
from plot_utils import plot_pretty
plot_pretty()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel(r'SGZ ($h^{-1}\ Mpc$)'); ax.set_ylabel(r'SGZ ($h^{-1}\ Mpc$)'); 
ax.set_zlabel(r'SGZ ($h^{-1}\ Mpc)$')
ax.set_xlim(-Rmax/2,Rmax/2); ax.set_ylim(-Rmax/2,Rmax/2); ax.set_zlim(-Rmax/2,Rmax/2)

plt.hold
ax.scatter(SGXm,SGYm,SGZm, s=m)

plt.show()
#plt.close(fig)
#del fig
