# %%
## Load module

import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import JSON
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colorbar as colorbar
from matplotlib.transforms import Affine2D
from astropy.visualization.wcsaxes import WCSAxes
from spectral_cube import SpectralCube
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from matplotlib.colors import LogNorm
import matplotlib

plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

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
# %%
###############
## Load data ##
###############
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
file = "PG0050_CO21.combine.all.line.10km.mom2.fits"

hdu = fits.open(path+file)[0]
mom2 = hdu.data[0][0]
mom2_level = np.linspace(0, 100, 11)
#rms = sigma_clipped_stats(mom0)[-1]
#mom0_level = np.array([-1,1,2,4,8,16,32,64])*2*rms

path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
file_mom0 = "PG0050_CO21.combine.all.line.10km.mom0.4sigma.rms.fits"
hdu = fits.open(path+file_mom0)[0]
mom0_rms = hdu.data[0][0]


file_mom0_peak1 = "PG0050_CO21.combine.all.line.10km.image.test0.mom0.fits"
hdu = fits.open(path+file_mom0_peak1)[0]
mom0_peak1 = hdu.data[0][0]
rms_peak1 = sigma_clipped_stats(mom0_peak1)[-1]
peak1_level = np.array([-2,2,4,8,16,32])*2*rms_peak1

file_mom0_peak2 = "PG0050_CO21.combine.all.line.10km.image.test1.mom0.fits"
hdu = fits.open(path+file_mom0_peak2)[0]
mom0_peak2 = hdu.data[0][0]
rms_peak2 = sigma_clipped_stats(mom0_peak2)[-1]
peak2_level = np.array([-2,2,4,8,16,32])*2*rms_peak2

# %%
## Overplot
pos_cen = [749, 745]
pix_size = 0.05
size = 100
#############################
## plot the momentum 0 map ##
#############################
xpos, ypos = 749, 745
x = np.linspace(749-200, 749+200)
k = np.tan((90+np.mean(pa_fit))*u.deg)
y = k*(x-xpos)+ypos
y_perp =  np.tan((np.mean(pa_fit))*u.deg)*(x-xpos)+ypos

# %%
import matplotlib
from matplotlib.colors import PowerNorm as normalize

transform = Affine2D()
transform.scale(pix_size, pix_size)
transform.translate(-pos_cen[0]*pix_size, -pos_cen[1]*pix_size)
transform.rotate(0.)  # radians

# Set up metadata dictionary
coord_meta = {}
coord_meta['name'] = 'RA (J2000)', 'DEC (J2000)'
coord_meta['type'] = 'longitude', 'latitude'
coord_meta['wrap'] = 180, None
coord_meta['unit'] = u.arcsec, u.arcsec
coord_meta['format_unit'] = None, None

## plot the figure
fig = plt.figure(figsize=(8,10))

ax = WCSAxes(fig, [0.1,0.1,0.8,0.8], aspect='equal',
             transform=transform, coord_meta=coord_meta)
fig.add_axes(ax)
im = ax.imshow(mom2, cmap='Greys', origin='lower', norm=LogNorm())
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label('$\sigma_v$ [km/s]')
ax.contour(mom2, mom2_level, colors=['k'])

## Overplot the double peak
ax.contour(mom0_peak1, peak1_level, lw=0.5, colors=['cyan'])
ax.contour(mom0_peak2, peak2_level, lw=0.5, colors=['r'])

## Plot the representing bar
xbar = np.linspace(pos_cen[0]+110, pos_cen[0]+110+82.05) 
ybar = (pos_cen[1]-195)*xbar/xbar
#ax.plot(xbar, ybar, 'k', lw=2)
#ax.text(xbar[0]+20,ybar[0]+5,'5kpc',size=20, color='k')

ax.scatter(xpos, ypos, c='w', edgecolor='k', zorder=3, linewidth=0.9, s=100, marker='*')
ax.plot(x, y,'--',linewidth=1.5, c='k', alpha=0.8)
ax.plot(x, y_perp,':',linewidth=1.5, c='k', alpha=0.8)

rec = matplotlib.patches.Rectangle((pos_cen[0]-size, pos_cen[1]-size), 10, 10, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax.add_artist(rec)
Beam = beam(hdu, pos_cen[0]-size+5,pos_cen[1]-size+5, 'k', pix_size)
ax.add_artist(Beam[0])

ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)

plt.savefig('/home/qyfei/Desktop/Codes/Result/mom0_overplot_red.pdf', bbox_inches='tight', dpi=300) #savefig

# %%
