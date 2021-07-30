# %%
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma import count
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colorbar as colorbar
from matplotlib.transforms import Affine2D
from astropy.visualization.wcsaxes import WCSAxes
import astropy.units as u
from spectral_cube import SpectralCube
from astropy.modeling import models, fitting
from astropy.stats import sigma_clipped_stats

#%matplotlib inline
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
    Beam = matplotlib.patches.Ellipse((xpos,ypos), Bmaj, Bmin, 90+bpa, edgecolor='k', facecolor='gray', fill=True, zorder=3)
    return Beam, Bmaj, Bmin

# %%
###############
## Load data ##
###############
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/cont/"
file = "PG0050-cont-mosaic.fits"
file_pbc = "PG0050-cont-mosaic-pbc.fits"

hdu = fits.open(path+file)[0]
pos_cen = [399, 395]
cont = hdu.data[0][0]*1e3
size = 100
pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
hdu = fits.open(path+file_pbc)[0]
cont_pbc = hdu.data[0][0]*1e3

#############################
## plot the momentum 0 map ##
#############################
import matplotlib
rms = 1.3e-02#sigma_clipped_stats(cont, sigma=3)[-1]

cont_level = np.array([-1,1,2,4,8,16,32])*3*rms

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
im = ax.imshow(cont_pbc, cmap='jet', origin='lower')
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label(r'cont [mJy$\cdot$beam$^{-1}$]')
ax.contour(cont_pbc, cont_level, colors=['k'])


rec = matplotlib.patches.Rectangle((pos_cen[0]-size, pos_cen[1]-size), 10, 10, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2.5)
ax.add_artist(rec)
Beam = beam(hdu, pos_cen[0]-size+5,pos_cen[1]-size+5, 'k', pix_size)
ax.add_artist(Beam[0])

circ = matplotlib.patches.Circle((pos_cen[0], pos_cen[1]), 15, ls = '--', fill = False, edgecolor='m', facecolor='w', zorder=3)
ax.add_artist(circ)

ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)

#plt.savefig('/home/qyfei/Desktop/Codes/Result/PG0050/PG0050p0124_cont.pdf', bbox_inches='tight', dpi=300) 

# %%

bmaj = hdu.header['BMAJ']
bmin = hdu.header['BMIN']
delt = hdu.header['CDELT1']
beam_area = np.pi*bmaj*bmin/delt**2/4/np.log(2)

pos_cen = [399, 395]
yy,xx = np.indices([800, 800],dtype='float')
radius = ((yy-pos_cen[1])**2+(xx-pos_cen[0])**2)**0.5
#mask = (abs(cont_pbc)>=2.*rms) & (radius<=15)
ape = 10/0.05
mask = radius<=ape
np.nansum(cont*mask)/beam_area
# %%

n_circ = 50
r_circ = np.random.uniform(2*ape, 400-ape, n_circ)
theta_circ = np.random.uniform(0, 2*np.pi, n_circ)
pos_cen_x_circ = 400 + r_circ*np.cos(theta_circ)
pos_cen_y_circ = 400 + r_circ*np.sin(theta_circ)

plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
for i in range(len(r_circ)):
    circ = matplotlib.patches.Circle((pos_cen_x_circ[i], pos_cen_y_circ[i]), ape, ls = '--', fill = False, edgecolor='m', facecolor='w', zorder=3)
    ax.add_artist(circ)

circ_in = matplotlib.patches.Circle((400, 400), ape, ls = '--', fill = False, edgecolor='r', facecolor='w', zorder=3)
circ_out = matplotlib.patches.Circle((400, 400), 400, ls = '--', fill = False, edgecolor='b', facecolor='w', zorder=3)
ax.add_artist(circ_in)
ax.add_artist(circ_out)
ax.imshow(cont, origin='lower', cmap='jet')
ax.set_xlim(0, 800)
ax.set_ylim(0, 800)
# %%
noise = np.zeros(n_circ)

for i in range(n_circ):
    radius = ((yy-pos_cen_y_circ[i])**2+(xx-pos_cen_x_circ[i])**2)**0.5
    mask = radius<=ape#(abs(cont_pbc)>=2.*rms) & 
    noise[i] = np.nansum(cont*mask)/beam_area

# %%
plt.figure(figsize=(8,8))
ax = plt.subplot(111)
ax.hist(noise, bins=10, orientation='vertical', color='r', alpha=0.5, density=True)
ax.vlines(sigma_clipped_stats(noise)[-1], 0, 10)
print(sigma_clipped_stats(noise))

# %%
