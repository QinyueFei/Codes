# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy.ma import masked_greater
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
from skimage.feature import peak, peak_local_max

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
# Load data 
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/NOEMA/w20cf/w20cf_origin/w20cf006/"
name = "F13403-0038"
file = name + "_cont.fits"
file_mom0 = name + "_CO32_mom0.fits"

hdu = fits.open(path+file)[0]
cont = hdu.data[0]*1e3 #convert to mJy/beam
hdr = hdu.header

bmaj = hdu.header['BMAJ']
bmin = hdu.header['BMIN']
delt = hdu.header['CDELT1']
pix_size = delt*u.deg.to('arcsec')
beam_area = np.pi*bmaj*bmin/delt**2/4/np.log(2)

wcs = WCS(hdu.header)
hdu_mom0 = fits.open(path+file_mom0)[0]
mom0 = hdu_mom0.data[0]
coordinates = peak_local_max(mom0, min_distance = 100)
print(coordinates)

# %%
## Plot continuum map
pos_cen = coordinates[0]
yy, xx = np.indices([hdr['NAXIS1'], hdr['NAXIS2']],dtype='float')
radius = ((yy-pos_cen[0])**2+(xx-pos_cen[1])**2)**0.5
rad = 1.0
ring = abs(rad/pix_size)

size = abs(4/pix_size)
cont_rms = sigma_clipped_stats(cont)[-1]
print("The rms noise of continuum is:", cont_rms, "mJy/beam")
cont_level = np.array([-2,2,3,4,5,6,7,8])*cont_rms

fig = plt.figure(figsize=(8,10))
ax = plt.subplot(projection=wcs[0])

im = ax.imshow(cont, vmin=-0.5, vmax=0.6, cmap='jet', origin='lower')
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label(name+' continuum [mJy/beam]')
ax.contour(cont, cont_level, colors=['k'])

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
ax.scatter(512, 512, marker="+", color='w', s=200, zorder=3)
ax.scatter(pos_cen[1], pos_cen[0], marker='*', color='k', zorder=4)

#plt.savefig("/home/qyfei/Desktop/Results/Result/NOEMA_detection/origin/"+name+"_cont.pdf", bbox_inches="tight", dpi=300)

# %%
## Esimate the flux: aperture
mask = (radius<=ring) & (cont>=2*cont_rms)
cont_mask = cont*mask

print("The aperture flux is:", np.nansum(cont_mask)/beam_area)

## Monte-Carlo sampling

radius = ((yy-pos_cen[0])**2+(xx-pos_cen[1])**2)**0.5
rad = 1.0
ring = abs(rad/pix_size)
# %%
ape = ring

n_circ = 50
r_circ = np.random.uniform(2*ape, 400-ape, n_circ)
theta_circ = np.random.uniform(0, 2*np.pi, n_circ)
pos_cen_x_circ = pos_cen[1] + r_circ*np.cos(theta_circ)
pos_cen_y_circ = pos_cen[0] + r_circ*np.sin(theta_circ)

plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
for i in range(len(r_circ)):
    circ = matplotlib.patches.Circle((pos_cen_x_circ[i], pos_cen_y_circ[i]), ape, ls = '--', fill = False, edgecolor='m', facecolor='w', zorder=3)
    ax.add_artist(circ)

circ_in = matplotlib.patches.Circle((pos_cen[1], pos_cen[0]), ape, ls = '--', fill = False, edgecolor='r', facecolor='w', zorder=3)
circ_out = matplotlib.patches.Circle((512, 512), 512, ls = '--', fill = False, edgecolor='b', facecolor='w', zorder=3)
ax.add_artist(circ_in)
ax.add_artist(circ_out)
ax.imshow(cont, origin='lower', cmap='jet')
ax.set_xlim(0, 1024)
ax.set_ylim(0, 1024)

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
