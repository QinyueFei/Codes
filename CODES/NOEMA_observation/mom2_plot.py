# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.modeling.projections import Projection
from astropy.units import si
from astropy.wcs import WCS
import matplotlib.colorbar as colorbar
from matplotlib.transforms import Affine2D
from astropy.visualization.wcsaxes import WCSAxes
import astropy.units as u
from spectral_cube import SpectralCube
from astropy.modeling import models, fitting
from astropy.stats import sigma_clipped_stats
import astropy.constants as c
from skimage.feature import peak_local_max

plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)
plt.rcParams['xtick.major.bottom'] = True
plt.rcParams['ytick.major.left'] = True

def beam(HDU, XPOS, YPOS, col, cellsize):
    hdu = HDU
    xpos, ypos = XPOS, YPOS
    c = col
    cell = cellsize
    bmaj, bmin, bpa = hdu.header['BMAJ'], hdu.header['BMIN'], hdu.header['BPA']
    Bmaj = bmaj*u.Unit('deg').to('arcsec')/cell
    Bmin = bmin*u.Unit('deg').to('arcsec')/cell
    Beam = matplotlib.patches.Ellipse((xpos,ypos), Bmaj, Bmin, 90+bpa, edgecolor='k', facecolor='gray', fill=True, zorder=3)
    return Beam, Bmaj, Bmin

# %%
## Load data ##
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/NOEMA/w20cf/w20cf_taper/w20cf001/"
name = "F08238+0752"
file = name + "_CO32_taper.fits"
file_mom0 = name + "_CO32_taper_mom0.fits"
file_mom1 = name + "_CO32_taper_mom1.fits"
file_mom2 = name + "_CO32_taper_mom2.fits"


hdu = fits.open(path+file)[0]

bmaj = hdu.header['BMAJ']
bmin = hdu.header['BMIN']
delt = hdu.header['CDELT1']
pix_size = delt*u.deg.to('arcsec')
beam_area = np.pi*bmaj*bmin/delt**2/4/np.log(2)

wcs = WCS(hdu.header)
# %%
# find source, estimate spectral resolution
cube_data = hdu.data
hdr = hdu.header
CO32_data = SpectralCube.read(path+file)
velo = CO32_data.spectral_axis
restfreq = hdr['RESTFREQ']/1e9
freq = (velo/c.c).si*restfreq+restfreq

hdu = fits.open(path+file_mom0)[0]
mom0 = hdu.data[0]
coordinates = peak_local_max(mom0, min_distance=100)
mom1 = fits.open(path+file_mom1)[0].data[0]
mom2 = fits.open(path+file_mom2)[0].data[0]

print(coordinates)
print(velo[1]-velo[2])

# %%
## Plot moment 0
pos_cen = coordinates[0]
yy, xx = np.indices([hdr['NAXIS1'], hdr['NAXIS2']],dtype='float')
radius = ((yy-pos_cen[0])**2+(xx-pos_cen[1])**2)**0.5
rad = 1.0
ring = abs(rad/pix_size)

size = abs(4/pix_size)
mom0_rms = sigma_clipped_stats(mom0)[-1]
print("The rms noise of moment 0 is:", mom0_rms, "Jy/beam km/s")
mom0_level = np.array([-1,1,2,4,8,16,32])*2*mom0_rms
mom2_level = np.linspace(0, 80, 5)
fig = plt.figure(figsize=(8,10))
ax = plt.subplot(projection=wcs[0])

im = ax.imshow(mom2, vmin=0, vmax=85, cmap='jet', origin='lower')
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label(name+' velocity dispersion [$\mathrm{km\,s^{-1}}$]')
#ax.contour(mom2, mom2_level, colors=['k'])

circ = matplotlib.patches.Circle((pos_cen[1], pos_cen[0]), ring, fill=False, edgecolor='m')
ax.add_artist(circ)

rec_size = abs(bmaj/delt)*1.25
rec = matplotlib.patches.Rectangle((pos_cen[1]-size, pos_cen[0]-size), rec_size, rec_size, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax.add_artist(rec)
Beam = beam(hdu, pos_cen[1]-size+rec_size/2,pos_cen[0]-size+rec_size/2, 'k', pix_size)
ax.add_artist(Beam[0])
ax.set_xlim(pos_cen[1]-size, pos_cen[1]+size)
ax.set_ylim(pos_cen[0]-size, pos_cen[0]+size)
ax.set_xlabel("R.A. (J2000)", labelpad=0.5)
ax.set_ylabel("Dec (J2000)", labelpad=-1.0)

ax.scatter(128, 128, marker="+", color='w', s=200, zorder=3)
ax.scatter(pos_cen[1], pos_cen[0], marker='*', color='k', zorder=3)

#plt.savefig("/home/qyfei/Desktop/Results/NOEMA_detection/taper/"+name+"_mom2.pdf", bbox_inches="tight", dpi=300)

# %%
