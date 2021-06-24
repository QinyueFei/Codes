import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colorbar as colorbar
from matplotlib.transforms import Affine2D
from astropy.visualization.wcsaxes import WCSAxes
import astropy.units as u
from spectral_cube import SpectralCube
from astropy.modeling import models, fitting
from astropy.stats import sigma_clipped_stats

plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'
plt.rcParams['font.size']=20
plt.rcParams['font.family']='serif'
plt.rcParams['xtick.minor.visible']=True
plt.rcParams['ytick.minor.visible']=True
plt.rcParams['xtick.top']=True
plt.rcParams['ytick.right']=True

def beam(HDU, XPOS, YPOS, col, cellsize):
    hdu = HDU
    xpos, ypos = XPOS, YPOS
    c = col
    cell = cellsize
    bmaj, bmin, bpa = hdu.header['BMAJ'], hdu.header['BMIN'], hdu.header['BPA']
    Bmaj = bmaj*u.Unit('deg').to('arcsec')/cell
    Bmin = bmin*u.Unit('deg').to('arcsec')/cell
    Beam = matplotlib.patches.Ellipse((xpos,ypos), Bmaj, Bmin, 90+bpa, 
                                      edgecolor='k', facecolor='gray', fill=True, zorder=3)
    return Beam, Bmaj, Bmin

###############
## Load data ##
###############
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/working/"
file = "PG0050_CO21.combine.all.line.10km.mom0.rms.fits"
hdu = fits.open(path+file)[0]
mom0_rms = hdu.data[0][0]

# build the mask
pos_cen = [745, 749]
yy,xx = np.indices([1500,1500],dtype='float')
radius = ((yy-pos_cen[1])**2+(xx-pos_cen[0])**2)**0.5
mask = abs(mom0_rms)>=0

# load datacube and calculate the spectrum in km/s
CO21_cube = SpectralCube.read(path+'PG0050_CO21.combine.all.line.10km.fits')
CO21_cube_with_mask = CO21_cube.with_mask(mask)
masked_CO21_cube = CO21_cube_with_mask.with_spectral_unit(unit='km/s',
                                                         rest_value=217.232*u.GHz,
                                                         velocity_convention='radio')

# calculate the spectrum with mask in Jy/beam

hdu = fits.open('PG0050_CO21.combine.all.line.10km.fits')[0]
CO21_cube = hdu.data[0]
spectrum = CO21_cube*mask

# calculate the beam area

bmaj = hdu.header['BMAJ']
bmin = hdu.header['BMIN']
delt = hdu.header['CDELT1']
beam_area = np.pi*bmaj*bmin/delt**2/4/np.log(2)

# calculate the total flux, y-axis
flux = np.nansum(spectrum,axis=(1,2))/beam_area*1e3*u.Unit('mJy')

# calculate the x-axis
velo = masked_CO21_cube.spectral_axis

sigma = sigma_clipped_stats(flux)
sigma_p = sigma
sigma_m = -sigma

#####################
## plot the figure ##
#####################

plt.figure(figsize=(8,12))
grid=plt.GridSpec(5,1,wspace=0,hspace=0)
ax1=plt.subplot(grid[0:3])
ax1.step(velo, flux,'k',label='comb_all')

ax1.hlines(0,-1000,1000,'k','--')
ax1.set_xlim(-1000,550)
ax1.fill_between(velo, sigma_p, sigma_m, facecolor='k',hatch='/',linestyle=':',alpha=0.5)
ax1.set_xlabel('Velocity [km/s]')
ax1.set_ylabel('Flux density [mJy]')
ax1.legend()
plt.savefig('/home/qyfei/Desktop/Codes/Result/spectrum.pdf', bbox_inches='tight', dpi=300)