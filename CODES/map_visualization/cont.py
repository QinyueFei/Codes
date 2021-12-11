import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy.ma import count
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

def load_cont(path, file):
    # Using procedure
    # cont, wcs, size, pix_size, hdu, pos_cen = load_cont(path, file)
    hdu = fits.open(path+file)[0]
    pos_cen = [399, 395]
    cont = hdu.data[0][0]*1e3
    size = 200
    pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    #hdu = fits.open(path+file_pbc)[0]
    #cont_pbc = hdu.data[0][0]*1e3
    wcs = WCS(hdu.header)
    return cont, wcs, size, pix_size, hdu, pos_cen

#############################
## plot the momentum 0 map ##
#############################
def plot_cont(path, file):
    cont, wcs, size, pix_size, hdu, pos_cen = load_cont(path, file)
    rms = 1.3e-02#sigma_clipped_stats(cont, sigma=3)[-1]
    cont_level = np.array([-1,1,2,4,8,16,32])*3*rms
    fig = plt.figure(figsize=(8,10))
    ax = plt.subplot(projection=wcs[0,0])
    im = ax.imshow(cont, cmap='jet', origin='lower')
    cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
    cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
    cb.set_label(r'cont [mJy$\cdot$beam$^{-1}$]')
    ax.contour(cont, cont_level, colors=['k'], linewidths=0.5)

    rec = matplotlib.patches.Rectangle((pos_cen[0]-size, pos_cen[1]-size), 10, 10, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2.5)
    ax.add_artist(rec)
    Beam = beam(hdu, pos_cen[0]-size+5,pos_cen[1]-size+5, 'k', pix_size)
    ax.add_artist(Beam[0])

    ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
    ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)
    ax.set_xlabel("R.A. (J2000)", labelpad=0.5, fontsize=20)
    ax.set_ylabel("Dec (J2000)", labelpad=-1.0, fontsize=20)

    #plt.show()
    plt.savefig('/home/qyfei/Desktop/Codes/Result/PG0050/PG0050p0124_cont.pdf', bbox_inches='tight', dpi=300)

path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/cont/"
file = "PG0050-cont-mosaic-pbc.fits"
plot_cont(path, file)