# %%
from matplotlib.colors import LogNorm
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
import astropy.constants as c
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
###############
## Load data ##
###############
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
file = "PG0050_CO21-combine-line-10km-mosaic-pbc-minorpv.fits"
file = "../Barolo_fit/output/PG0050+124/pvs/PG0050+124_pv_b.fits"
mod_file = "../Barolo_fit/output/PG0050+124/pvs/PG0050+124mod_pv_b_local.fits"
hdu = fits.open(path+file)[0]
PV_maj = hdu.data

from astropy.stats import sigma_clipped_stats
sigma_PV = sigma_clipped_stats(PV_maj, sigma = 3)[-1]
PV_level = np.array([-2,2,np.sqrt(8),4, np.sqrt(32),8,np.sqrt(128),16,np.sqrt(512),32,np.sqrt(2048),64])*sigma_PV

pix_size=0.05

hdu = fits.open(path+mod_file)[0]
PV_maj_mod = hdu.data

# %%
transform = Affine2D()
transform.scale(pix_size, 10.78065)
#transform.translate(-150*pix_size, -37*10.78065)
transform.translate(-395*pix_size, -115*10.78065)

transform.rotate(0.)  # radians

# Set up metadata dictionary
coord_meta = {}
coord_meta['name'] = 'Offset [\"]', '$\Delta$V [$\mathrm{km\,s^{-1}}$]'
coord_meta['type'] = 'longitude', 'scalar'
coord_meta['wrap'] = 180, None
coord_meta['unit'] = u.arcsec, u.Unit("")
coord_meta['format_unit'] = None, None

fig = plt.figure(figsize=(10,10))
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True

ax = WCSAxes(fig, [0.1,0.1,0.8,0.8], aspect='equal',
             transform=transform, coord_meta=coord_meta)
fig.add_axes(ax)
ax.imshow(PV_maj, vmin=-3*sigma_PV, vmax=8*np.sqrt(2)*sigma_PV, cmap='jet')
ax.hlines(37, 0, 1000, 'k', lw=0.5)
ax.vlines(150, 0, 1000, 'k', lw=0.5)

#ax.set_xticks(offset.value)
ax.contour(PV_maj_mod, PV_level, linewidths=1, colors='k')
#ax.contour(PV_mod_maj, PV_level, linewidths=1, colors='r')

#ax.set_xlim(40, 260)
ax.set_xlim(350, 450)
ax.set_ylim(80, 150)
#ax.set_ylim(0, 80)

plt.savefig('/home/qyfei/Desktop/Codes/Result/PG0050/PV_minor_comp.pdf', bbox_inches='tight')

# %%
