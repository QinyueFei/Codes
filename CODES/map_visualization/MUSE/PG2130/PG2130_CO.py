# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sympy import approximants
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colorbar as colorbar
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from map_visualization.maps import beam
from astropy.cosmology import Planck15

#%matplotlib inline
plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

# %%
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG0923/Barolo_fit/output/PG0923+129_well"
mod_file = "/maps/PG0923+129_local_1mom.fits"
dat_file = "/maps/PG0923+129_1mom.fits"

# %%
## model and data
mod_hdu = fits.open(path+mod_file)
mod_map = mod_hdu[0].data

dat_hdu = fits.open(path+dat_file)
dat_map = dat_hdu[0].data

res_map = mod_map - dat_map

# %%
def iso_rad(sizex_, sizey_, pos_cen_, pix_size_, PA_, inc_):
    sizex, sizey, pos_cen, pix_size, PA, inc = sizex_, sizey_, pos_cen_, pix_size_, PA_, inc_
    # This function calculate the radius between each pixel and the kinematic center
    # size, pos_cen, pix_size are adopted from observation, which are map size, coordinates of galaxy center and size of each pixel
    # PA, inc are position angle and inclination angle
    z = 0.029
    yy,xx = np.indices([sizey, sizex],dtype='float')
    coordinates_xx = (xx-pos_cen[0])*np.cos(PA*u.deg).value + (yy-pos_cen[1])*np.sin(PA*u.deg).value
    coordinates_yy = -(xx-pos_cen[0])*np.sin(PA*u.deg).value + (yy-pos_cen[1])*np.cos(PA*u.deg).value
    Radius_pixel = np.sqrt(coordinates_xx**2 + coordinates_yy**2/(np.cos(inc*u.deg).value)**2)
    Radius = (Radius_pixel * pix_size)*u.arcsec#/Planck15.arcsec_per_kpc_proper(z)).to('kpc')
    return Radius.value

# %%
radius_CO = iso_rad(185, 185, (88.4, 89.66), 0.18, 261.29+90, 40)

ring_bound_CO = np.arange(0, 6.25, 0.5)

ring_res_CO = np.zeros(len(ring_bound_CO) - 1)
for i in range(len(ring_bound_CO)-1):
    N = np.where((radius_CO>=ring_bound_CO[i]) & (radius_CO<ring_bound_CO[i+1]))
    ring_res_CO[i] = np.sqrt(np.nansum(res_map[N]**2)/len(N[0]))
