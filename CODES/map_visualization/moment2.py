import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colorbar as colorbar
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from map_visualization.maps import beam
import matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

###############
## Load data ##
###############
size = 50                      ## field of view
def load_mom2(path, file):
    # Using procedure
    # mom2, wcs, size, pix_size, hdu, pos_cen = load_mom2(path, file)
    hdu = fits.open(path+file)[0]   ## header of observation data
    wcs = WCS(hdu.header)           ## coordinates system
    pos_cen = [540, 540]            ## galaxy center
    mom2 = hdu.data[0][0]           ## moment 2 data
    
    pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    return mom2, wcs, size, pix_size, hdu, pos_cen
def plot_mom2(path, file):
    mom2, wcs, size, pix_size, hdu, pos_cen = load_mom2(path, file)
    level = np.linspace(5,30,6)
    print(level)
    fig = plt.figure(figsize=(8,10))
    ax = plt.subplot(projection=wcs[0,0])
    im = ax.imshow(mom2, vmin=5, vmax=level[-1]+1, cmap='jet', origin='lower')
    cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
    cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
    cb.set_label(r'$\sigma_\mathrm{gas}$ [$\mathrm{km\,s^{-1}}$]')
    ax.contour(mom2, level, colors=['k'])

    bmaj, bmin, bpa, delt = hdu.header['BMAJ'], hdu.header['BMIN'], hdu.header['BPA'], hdu.header['CDELT1']
    rec_size = abs(bmaj/delt)*1.5
    rec = matplotlib.patches.Rectangle((pos_cen[0]-size, pos_cen[1]-size), rec_size, rec_size, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2.5)
    ax.add_artist(rec)
    Beam = beam(hdu, pos_cen[0]-size+5,pos_cen[1]-size+5, 'gray', pix_size)
    ax.add_artist(Beam[0])

    ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
    ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)
    ax.set_xlabel("R.A. (J2000)", labelpad=0.5, fontsize=20)
    ax.set_ylabel("Dec (J2000)", labelpad=-1.0, fontsize=20)
    # plt.show()
    # plt.savefig('/home/qyfei/Desktop/Results/map_visualization/Image/PG_quasars/moment2/PG1244_CO32_10km_resolution_mom2.pdf', bbox_inches='tight', dpi=300) #savefig
# path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
# file = "PG0050_CO21-combine-line-10km-mosaic-mom2.fits"
# file_pbc = "PG0050_CO21-combine-line-10km-mosaic-pbc-mom2.fits"
# object = "PG0923"
# path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/"+object
# file = "/"+object+".dilmsk.mom2.fits.gz"
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG1244/CO32/Re_Re_calibration/product"
file = "/PG1244+026_CO32_10km_clean_mom2.fits"

plot_mom2(path, file)