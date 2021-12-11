import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colorbar as colorbar
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from map_visualization.maps import beam

#%matplotlib inline
plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=20)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)


def load_mom0(path, file):
    # Using
    # mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file)
    hdu = fits.open(path+file)[0]
    mom0 = hdu.data[0][0]
    wcs = WCS(hdu.header)
    pos_cen = np.where(mom0 == np.nanmax(mom0))
    pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    r = sigma_clipped_stats(mom0)[-1]
    size = 200
    # Output map data, coordinate system, galaxy center, size of each pixel, noise level and beam shape
    return mom0, wcs, pos_cen, size, pix_size, r, hdu

#############################
## plot the momentum 0 map ##
#############################
def plot_mom0(path, file):
    mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file)
    fov = 200 #define the size of the map
    r = 0.043
    mom0_level = np.array([-1,1,2,4,8,16,32])*3*r
    fig = plt.figure(figsize=(8,10))
    ax = plt.subplot(projection=wcs[0,0])
    im = ax.imshow(mom0, cmap='jet', origin='lower')
    cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
    cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
    cb.set_label('CO(2-1) [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]')
    ax.contour(mom0, mom0_level, colors=['k'])

    ## plot the figure
    rec = matplotlib.patches.Rectangle((pos_cen[0]+100, pos_cen[1]-200), 200, 30, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
    ax.add_artist(rec)
    xbar = np.linspace(pos_cen[0]+110, pos_cen[0]+110+82.05) 
    ybar = (pos_cen[1]-195)*xbar/xbar
    ax.plot(xbar, ybar, 'k', lw=2, zorder=2.5) #plot the representer bar
    ax.text(xbar[0]+20,ybar[0]+5,'5kpc',size=20, color='k', zorder=2.5)

    rec_size = 10
    rec = matplotlib.patches.Rectangle((pos_cen[0]-fov, pos_cen[1]-fov), rec_size, rec_size, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
    ax.add_artist(rec)
    Beam = beam(hdu, pos_cen[0]-size+rec_size/2,pos_cen[1]-size+rec_size/2, 'k', pix_size)
    ax.add_artist(Beam[0])

    ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
    ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)
    ax.set_xlabel("R.A. (J2000)", labelpad=0.5, fontsize=20)
    ax.set_ylabel("Dec (J2000)", labelpad=-1.0, fontsize=20)
    #plt.show()
    plt.savefig('/home/qyfei/Desktop/Codes/Result/PG0050/PG0050p0124_mom0.pdf', bbox_inches='tight')

#path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
#file = "PG0050_CO21-combine-line-10km-mosaic-mom0-pbcor.fits"
#plot_mom0(path, file)