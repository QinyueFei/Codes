# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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


# def load_mom1(path, file):
#     # Using procedure
#     # mom1, wcs, size, pix_size, hdu, pos_cen = load_mom1(path, file)
#     hdu = fits.open(path+file)[0]
#     wcs = WCS(hdu.header)
#     pos_cen = [540, 540]
#     mom1 = hdu.data[0][0] - hdu.data[0][0][540, 540]
#     pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
#     return mom1, wcs, size, pix_size, hdu, pos_cen
    
# def plot_mom1(path, file):
    # mom1, wcs, size, pix_size, hdu, pos_cen = load_mom1(path, file)


path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG2130"
file = "/PG2130+099_Ha_SN3lim_fullmaps.fits"

hdu = fits.open(path + file)
vlosHa = hdu[1].data
vlosHa = vlosHa - np.nanmean(vlosHa)

pos_cen = np.array([len(vlosHa)/2, len(vlosHa[1])/2])
# %%
## plot the line-of-sight velocity map

size = 90
fov = size #define the size of the map
r = 0.20
pix_size = 0.20

level = np.arange(-300, 320, 20)
print(level)
fig = plt.figure(figsize=(8,10))
ax = plt.subplot(111)#projection=wcs[0,0])
im = ax.imshow(vlosHa, vmin=level[0]+5, vmax=level[-1]-5, cmap='coolwarm', origin='lower')
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label(r'$V_\mathrm{los}$ [$\mathrm{km\,s^{-1}}$]')
ax.contour(vlosHa, level, colors=['k'])

bmaj, bmin, bpa, delt = 1.0*u.arcsec.to('deg'), 1.0*u.arcsec.to('deg'), 90.0, 0.20*u.arcsec.to('deg')
rec_size = abs(bmaj/delt)*4.0


rec = matplotlib.patches.Rectangle((pos_cen[0]-fov, pos_cen[1]-fov), rec_size, rec_size, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax.add_artist(rec)
# Beam = beam(hdu, pos_cen[0]-size+rec_size/2,pos_cen[1]-size+rec_size/2, 'k', pix_size)
Beam = matplotlib.patches.Ellipse((pos_cen[0]-fov+rec_size/2, pos_cen[1]-fov+rec_size/2), bmaj/delt, bmin/delt, angle=0.0,fill=True, edgecolor='k', facecolor='k', zorder=3)
ax.add_artist(Beam)

redshift = 0.048
scale_bar_length = 5.0*u.kpc*Planck15.arcsec_per_kpc_proper(redshift)
bar_size = abs(scale_bar_length.to('deg').value/delt)*1.5
rec = matplotlib.patches.Rectangle((pos_cen[0]+fov-bar_size, pos_cen[1]-fov), bar_size, rec_size, angle=0.0, fill=True, edgecolor='k', facecolor='w', zorder=2)
ax.add_artist(rec)
bar_length = abs(scale_bar_length.to('deg').value/delt)
scale_bar = matplotlib.patches.Rectangle((pos_cen[0]+fov-(bar_length+bar_size)/2, pos_cen[1]-fov+rec_size*3/5), bar_length, 2, angle=0.0, fill=True, edgecolor='k', facecolor='k', zorder=3)
ax.add_artist(scale_bar)

ax.text(pos_cen[0]+fov-bar_size/2, pos_cen[1]-fov+rec_size/6, '5kpc', ha='center')
    
ax.text(pos_cen[0]+fov-rec_size/6, pos_cen[1]+fov-rec_size/6, 'PG\,2130+099', color='k', ha='right', va='top', fontsize=30)

ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)
# ax.set_xlabel("R.A. (J2000)", labelpad=0.5, fontsize=20)
# ax.set_ylabel("Dec (J2000)", labelpad=-1.0, fontsize=20)
ax.set_xticks([])
ax.set_yticks([])

# plt.savefig("/home/qyfei/Desktop/Results/map_visualization/MUSE/PG2130/PG2130_vlosHa.pdf", bbox_inches="tight", dpi=300)

# %%


hdu[1].header['CTYPE1'] = 'RA---SIN'
hdu[1].header['CRVAL1'] = 3.231159030000E+02
hdu[1].header['CDELT1'] = -0.2*u.arcsec.to('deg')
hdu[1].header['CRPIX1'] = 162
hdu[1].header['CUNIT1'] = 'deg     '

hdu[1].header['CTYPE2'] = 'DEC---SIN'
hdu[1].header['CRVAL2'] = 1.013868800000E+01 
hdu[1].header['CDELT2'] = 0.2*u.arcsec.to('deg')
hdu[1].header['CRPIX2'] = 158
hdu[1].header['CUNIT2'] = 'deg     '

hdu[1].header['BMAJ'] = 1.*u.arcsec.to('deg')
hdu[1].header['BMIN'] = 1.*u.arcsec.to('deg')
hdu[1].header['BPA'] = 0

fits.writeto("/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG2130/Barolo_fit/MUSE_Ha/PG2130_VlosHa.fits", vlosHa, hdu[1].header, overwrite=True)

# %%
mask = 1 - np.isnan(vlosHa)
plt.imshow(mask)
plt.colorbar()
# %%
fits.writeto("/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG0923/Barolo_fit/MUSE_Ha/PG0923_mask.fits", mask, hdu[1].header, overwrite=True)

# %%
evlosHa = hdu[4].data
level = np.arange(-20, 22, 2)
print(level)
fig = plt.figure(figsize=(8,10))
ax = plt.subplot(111)#projection=wcs[0,0])
im = ax.imshow(evlosHa, vmin=level[0]+5, vmax=level[-1]-5, cmap='coolwarm', origin='lower')
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label(r'$\sigma_{V_\mathrm{los}}$ [$\mathrm{km\,s^{-1}}$]')
# ax.contour(evlosHa, level, colors=['k'])

bmaj, bmin, bpa, delt = 1.0*u.arcsec.to('deg'), 1.0*u.arcsec.to('deg'), 90.0, 0.20*u.arcsec.to('deg')
rec_size = abs(bmaj/delt)*5.0


rec = matplotlib.patches.Rectangle((pos_cen[0]-fov, pos_cen[1]-fov), rec_size, rec_size, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax.add_artist(rec)
# Beam = beam(hdu, pos_cen[0]-size+rec_size/2,pos_cen[1]-size+rec_size/2, 'k', pix_size)
Beam = matplotlib.patches.Ellipse((pos_cen[0]-fov+rec_size/2, pos_cen[1]-fov+rec_size/2), bmaj/delt, bmin/delt, angle=0.0,fill=True, edgecolor='k', facecolor='k', zorder=3)
ax.add_artist(Beam)

redshift = 0.029
scale_bar_length = 5.0*u.kpc*Planck15.arcsec_per_kpc_proper(redshift)
bar_size = abs(scale_bar_length.to('deg').value/delt)*1.5
rec = matplotlib.patches.Rectangle((pos_cen[0]+fov-bar_size, pos_cen[1]-fov), bar_size, rec_size, angle=0.0, fill=True, edgecolor='k', facecolor='w', zorder=2)
ax.add_artist(rec)
bar_length = abs(scale_bar_length.to('deg').value/delt)
scale_bar = matplotlib.patches.Rectangle((pos_cen[0]+fov-(bar_length+bar_size)/2, pos_cen[1]-fov+rec_size*3/5), bar_length, 2, angle=0.0, fill=True, edgecolor='k', facecolor='k', zorder=3)
ax.add_artist(scale_bar)

ax.text(pos_cen[0]+fov-bar_size/2, pos_cen[1]-fov+rec_size/6, '5kpc', ha='center')
    
ax.text(pos_cen[0]+fov-rec_size/6, pos_cen[1]+fov-rec_size/6, 'PG\,0923+129', color='k', ha='right', va='top', fontsize=30)

ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)
# ax.set_xlabel("R.A. (J2000)", labelpad=0.5, fontsize=20)
# ax.set_ylabel("Dec (J2000)", labelpad=-1.0, fontsize=20)
ax.set_xticks([])
ax.set_yticks([])

