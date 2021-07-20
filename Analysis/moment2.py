import numpy as np
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

###############
## Load data ##
###############
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/working/"
file = "PG0050_CO21.combine.all.line.10km.mom2.fits"

hdu = fits.open(path+file)[0]
pos_cen = [745, 749]
mom1 = hdu.data[0][0] #- hdu.data[0][0][745, 749]
size = 200
pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')

#############################
## plot the momentum 2 map ##
#############################
import matplotlib

level = np.linspace(0,100,6)

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

fig = plt.figure(figsize=(8,10))

ax = WCSAxes(fig, [0.1,0.1,0.8,0.8], aspect='equal',
             transform=transform, coord_meta=coord_meta)
fig.add_axes(ax)
im = ax.imshow(mom1, cmap='jet', origin='lower')
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label(r'$\sigma_\mathrm{gas}$ [km/s]')
ax.contour(mom1, level, colors=['k'])


rec = matplotlib.patches.Rectangle((pos_cen[0]-size, pos_cen[1]-size), 10, 11,
                                   angle=0.0,fill=True, edgecolor='k', facecolor='w')
ax.add_artist(rec)
Beam = beam(hdu, pos_cen[0]-size+5,pos_cen[1]-size+5, 'k', pix_size)
ax.add_artist(Beam[0])

ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)

plt.savefig('/home/qyfei/Desktop/Codes/Result/mom2.pdf', bbox_inches='tight', dpi=300) #savefig