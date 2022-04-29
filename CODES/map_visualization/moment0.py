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

size = 50

def load_mom0(path, file):
    # Using
    # mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file)
    hdu = fits.open(path+file)[0]
    mom0 = hdu.data[0][0]
    wcs = WCS(hdu.header)
    pos_cen = np.where(mom0 == np.nanmax(mom0))
    pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    r = sigma_clipped_stats(mom0)[-1]
    # size = 150
    # Output map data, coordinate system, galaxy center, size of each pixel, noise level and beam shape
    return mom0, wcs, pos_cen, size, pix_size, r, hdu

#############################
## plot the momentum 0 map ##
#############################
def plot_mom0(path, file):
    mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file)
    # pos_cen = [89, 89]
    fov = size #define the size of the map
    r = 0.065
    mom0_level = np.array([-1,1,2,4,8,16,32,64,128,256,512,1024])*2*r
    fig = plt.figure(figsize=(8,10))
    ax = plt.subplot(projection=wcs[0,0])#, when moment maps is directly obtained from CASA
    im = ax.imshow(mom0, cmap='magma', origin='lower')#, norm=colors.PowerNorm(0.5))
    cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
    cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
    cb.set_label('CO(3-2) [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]')
    ax.contour(mom0, mom0_level, colors=['k'])

    ax.scatter(544, 539, marker="x", color='k', s=100, zorder=3)
    ax.scatter(544, 538, marker="+", color='w', s=100)
    ax.scatter(540, 540, marker="o", color='b', s=10)
    ## plot the figure
    # rec = matplotlib.patches.Rectangle((pos_cen[0]+100, pos_cen[1]-200), 200, 30, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
    # ax.add_artist(rec)
    # xbar = np.linspace(pos_cen[0]+110, pos_cen[0]+110+82.05) 
    # ybar = (pos_cen[1]-195)*xbar/xbar
    # ax.plot(xbar, ybar, 'k', lw=2, zorder=2.5) #plot the representer bar
    # ax.text(xbar[0]+20,ybar[0]+5,'5kpc',size=20, color='k', zorder=2.5)

    bmaj, bmin, bpa, delt = hdu.header['BMAJ'], hdu.header['BMIN'], hdu.header['BPA'], hdu.header['CDELT1']
    rec_size = abs(bmaj/delt)*1.5

    rec = matplotlib.patches.Rectangle((pos_cen[0]-fov, pos_cen[1]-fov), rec_size, rec_size, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
    ax.add_artist(rec)
    Beam = beam(hdu, pos_cen[0]-size+rec_size/2,pos_cen[1]-size+rec_size/2, 'k', pix_size)
    ax.add_artist(Beam[0])

    redshift = 0.048
    scale_bar_length = 0.4*u.kpc*Planck15.arcsec_per_kpc_proper(redshift)
    bar_size = abs(scale_bar_length.to('deg').value/delt)*1.5
    rec = matplotlib.patches.Rectangle((pos_cen[0]+fov-bar_size, pos_cen[1]-fov), bar_size, rec_size, angle=0.0, fill=True, edgecolor='k', facecolor='w', zorder=2)
    ax.add_artist(rec)
    bar_length = abs(scale_bar_length.to('deg').value/delt)
    scale_bar = matplotlib.patches.Rectangle((pos_cen[0]+fov-(bar_length+bar_size)/2, pos_cen[1]-fov+rec_size*3/5), bar_length, 2, angle=0.0, fill=True, edgecolor='k', facecolor='k', zorder=3)
    ax.add_artist(scale_bar)
    ax.text(pos_cen[0]+fov-bar_size/2, pos_cen[1]-fov+rec_size/6, '400pc', ha='center')
    
    ax.text(pos_cen[0]+fov-rec_size/6, pos_cen[1]+fov-rec_size/6, 'PG1244+026', color='w', ha='right', va='top', fontsize=30)

    ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
    ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)
    ax.set_xlabel("R.A. (J2000)", labelpad=0.5, fontsize=20)
    ax.set_ylabel("Dec (J2000)", labelpad=-1.0, fontsize=20)
    plt.show()

    # plt.savefig('/home/qyfei/Desktop/Results/map_visualization/Image/PG_quasars/moment0/PG1244_CO32_mom0_present.pdf', bbox_inches='tight', dpi=300)

# %%
# path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
# file = "PG0050_CO21-combine-line-10km-mosaic-mom0-pbcor.fits"

path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG1244/CO32/Re_Re_calibration/product"
file = "/PG1244+026_CO32_10km_clean_mom0.fits"

fits.open(path+file)[0].header
# CO21file = "/PG1244+026_CO21.final.mom0.regrid.fits"
# CO21_data = fits.open(path+CO21file)[0].data[0][0]


# path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG0923"
# file = "/PG0923 129_Ha_SN3lim_fullmaps.fits"
# plot_mom0(path, file)

# object = "PG0923"
# path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/"+object+"/data"
# file = "/"+object+".dilmsk.mom0.fits.gz"
# # mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file)
# # print(pos_cen)
# plot_mom0(path, file)

# # %%
# ## Dilated Moment 0 mapes
# path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/"+object+"/data"
# file = "/"+object+".dilmsk.mom0.fits.gz"

# hdumom0 = fits.open(path+file)[0]
# hdrmom0 = hdumom0.header
# mom0_dilated = hdumom0.data

# ## Build the mask
# mask = np.ones((len(mom0_dilated), len(mom0_dilated)))
# mask[np.isnan(mom0_dilated)] = 0

# beam2pixels = np.pi*hdrmom0['BMAJ']*hdrmom0['BMIN']/(4*np.log(2))/hdrmom0['CDELT1']**2
# mom0flux_dilated = np.nansum(mom0_dilated)/beam2pixels
# rms = np.sqrt(len(np.where(mask)[0])*0.049/beam2pixels**2)

# print("The flux of dilated map is:", mom0flux_dilated, "Jy km/s")

# # %%
# ## Flux in mask region of moment 0 maps
# object = "PG1126"
# path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/"+object+"/data"
# file = "/PG1126-041_CO21_final_image_mom0.fits"

# plot_mom0(path, file)

# hdumom0 = fits.open(path+file)[0]
# mom0 = hdumom0.data[0][0]
# mom0flux = np.nansum(mom0)/beam2pixels
# rms = np.sqrt(len(np.where(mask)[0])/beam2pixels*0.074**2)

# print("The flux of moment map is:", mom0flux, r"$\pm$", rms, "Jy km/s")

# # %%
# ## Extract the spectrum of same region
# path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/"+object+"/data"
# file = "/PG2130+099_CO21.final.image.fits"
# hducube = fits.open(path+file)[0]
# cube = hducube.data[0]
# mask_spectrum = np.nansum(cube*mask, axis=(1,2))/beam2pixels*10.15755
# spectrumflux = np.nansum(mask_spectrum)

# print("The flux extracted from spectrum is:", spectrumflux, "Jy km/s")

# # %%
# maskregion = np.where(mom0_dilated>=0)

# plt.figure(figsize=(8, 8))
# ax = plt.subplot(111)
# ax.scatter(mom0_dilated[maskregion], mom0[maskregion], color='b', marker='o', alpha=0.1)
# ax.plot([1e-03, 1e3], [1e-03,1e3], 'k-.', zorder=3)
# ax.vlines(2*0.074, 1e-03, 1e3, colors='k', ls='--')
# ax.hlines(2*0.074, 1e-03, 1e3, colors='k', ls='--')

# ax.loglog()
# ax.set_xlim(1e-2,10)
# ax.set_ylim(1e-2,10)
# ax.set_xlabel("Dilated Flux [Jy/beam km/s]")
# ax.set_ylabel("Flux [Jy/beam km/s]")

# plt.savefig('/home/qyfei/Desktop/Results/map_visualization/Image/PG_quasars/moment0/PG2130_flux_comparison.pdf', bbox_inches='tight', dpi=300)

# %%
# CO21mom0, wcs, pos_cen, size, pix_size, r, CO21hdu = load_mom0(path, CO21file)

# mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file)

# size = 150
# # pos_cen = [89, 89]
# fov = size #define the size of the map
# r = 0.065
# mom0_level = np.array([-1,1,2,4,8,16,32,64,128,256,512,1024])*2*r
# fig = plt.figure(figsize=(8,10))
# ax = plt.subplot(projection=wcs[0,0])#, when moment maps is directly obtained from CASA
# im = ax.imshow(mom0, cmap='jet', origin='lower')#, norm=colors.PowerNorm(0.5))
# cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
# cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
# cb.set_label('CO(3-2) [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]')
# ax.contour(mom0, mom0_level, colors=['w'])


# CO21level = np.array([-1,1,2,4,8,16])*2*3.6e-02
# ax.contour(CO21mom0, CO21level, colors=['k'])

# ax.scatter(544, 539, marker="x", color='k', s=100, zorder=3)
# ax.scatter(544, 538, marker="+", color='w', s=100)
# ax.scatter(540, 540, marker="o", color='b', s=10)

# bmaj, bmin, bpa, delt = CO21hdu.header['BMAJ'], CO21hdu.header['BMIN'], CO21hdu.header['BPA'], CO21hdu.header['CDELT1']
# rec_size = abs(bmaj/delt)*1.5
# rec = matplotlib.patches.Rectangle((pos_cen[0]-fov, pos_cen[1]-fov), rec_size, rec_size, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
# ax.add_artist(rec)
# CO21Beam = beam(CO21hdu, pos_cen[0]-size+rec_size/2,pos_cen[1]-size+rec_size/2, 'gray', pix_size)
# ax.add_artist(CO21Beam[0])

# bmaj, bmin, bpa, delt = hdu.header['BMAJ'], hdu.header['BMIN'], hdu.header['BPA'], hdu.header['CDELT1']
# rec_size = abs(bmaj/delt)*1.5
# rec = matplotlib.patches.Rectangle((pos_cen[0]-fov, pos_cen[1]-fov), rec_size, rec_size, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
# ax.add_artist(rec)
# Beam = beam(hdu, pos_cen[0]-size+rec_size/2,pos_cen[1]-size+rec_size/2, 'gray', pix_size)
# ax.add_artist(Beam[0])

# ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
# ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)
# ax.set_xlabel("R.A. (J2000)", labelpad=0.5, fontsize=20)
# ax.set_ylabel("Dec (J2000)", labelpad=-1.0, fontsize=20)
# plt.show()
# plt.savefig('/home/qyfei/Desktop/Results/map_visualization/Image/PG_quasars/moment0/PG1244_CO32_mom0_with_overplotted_CO21.pdf', bbox_inches='tight', dpi=300)


# %%
