# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colorbar as colorbar
from matplotlib.colors import LogNorm
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from map_visualization.maps import beam
from matplotlib import colors
from astropy.cosmology import Planck15

#%matplotlib inline
plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)


# def load_mom0(path, file):
#     # Using
#     # mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file)
#     hdu = fits.open(path+file)[0]
#     mom0 = hdu.data[0][0]
#     wcs = WCS(hdu.header)
#     pos_cen = np.where(mom0 == np.nanmax(mom0))
#     pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
#     r = sigma_clipped_stats(mom0)[-1]
#     # size = 150
#     # Output map data, coordinate system, galaxy center, size of each pixel, noise level and beam shape
#     return mom0, wcs, pos_cen, size, pix_size, r, hdu

#############################
## plot the momentum 0 map ##
#############################
# def plot_mom0(path, file):
# mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file)
    # pos_cen = [89, 89]

path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG0923"
file = "/PG0923 129_Ha_SN3lim_fullmaps.fits"

hdu = fits.open(path + file)
intHa = hdu[0].data

pos_cen = np.array([len(intHa)/2, len(intHa[1])/2])

size = 100
fov = size #define the size of the map
r = 0.20
pix_size = 0.20
mom0_level = np.array([-1,1,2,4,8,16,32,64,128,256,512,1024])*2*r
fig = plt.figure(figsize=(8,10))
ax = plt.subplot(111)#projection=wcs[0,0])#, when moment maps is directly obtained from CASA
im = ax.imshow(intHa, cmap='magma', origin='lower', norm=LogNorm())
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label(r'H$\alpha$')#[$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]
ax.contour(intHa, mom0_level, colors=['k'])

# ax.scatter(544, 539, marker="x", color='k', s=100, zorder=3)
# ax.scatter(544, 538, marker="+", color='w', s=100)
# ax.scatter(540, 540, marker="o", color='b', s=10)
    ## plot the figure
    # rec = matplotlib.patches.Rectangle((pos_cen[0]+100, pos_cen[1]-200), 200, 30, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
    # ax.add_artist(rec)
    # xbar = np.linspace(pos_cen[0]+110, pos_cen[0]+110+82.05) 
    # ybar = (pos_cen[1]-195)*xbar/xbar
    # ax.plot(xbar, ybar, 'k', lw=2, zorder=2.5) #plot the representer bar
    # ax.text(xbar[0]+20,ybar[0]+5,'5kpc',size=20, color='k', zorder=2.5)

# bmaj, bmin, bpa, delt = hdu.header['BMAJ'], hdu.header['BMIN'], hdu.header['BPA'], hdu.header['CDELT1']
# rec_size = abs(bmaj/delt)*1.5

bmaj, bmin, bpa, delt = 1.0*u.arcsec.to('deg'), 1.0*u.arcsec.to('deg'), 90.0, 0.20*u.arcsec.to('deg')
rec_size = abs(bmaj/delt)*4.0


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

# plt.show()
# plt.savefig("/home/qyfei/Desktop/Results/map_visualization/MUSE/PG0923/PG0923_intHa.pdf", bbox_inches="tight", dpi=300)

# %%
hdu = fits.open(path + file)
eintHa = hdu[3].data

pos_cen = np.array([len(intHa)/2, len(intHa[1])/2])

fov = size #define the size of the map
r = 0.20
pix_size = 0.20
mom0_level = np.array([-1,1,2,4,8,16,32,64,128,256,512,1024])*2*r
fig = plt.figure(figsize=(8,10))
ax = plt.subplot(111)#projection=wcs[0,0])#, when moment maps is directly obtained from CASA
im = ax.imshow(eintHa, cmap='magma', origin='lower', norm=LogNorm())
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label(r'int H$\alpha$ err')#[$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]
ax.contour(eintHa, mom0_level, colors=['k'])

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
ax.text(pos_cen[0]+fov-rec_size/6, pos_cen[1]+fov-rec_size/6, 'PG0923+129', color='k', ha='right', va='top', fontsize=30)
ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)
ax.set_xticks([])
ax.set_yticks([])

# plt.show()
# plt.savefig("/home/qyfei/Desktop/Results/map_visualization/MUSE/PG0923/PG0923_intHa.pdf", bbox_inches="tight", dpi=300)
