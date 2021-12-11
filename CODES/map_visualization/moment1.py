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
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)


path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
file = "PG0050_CO21-combine-line-10km-mosaic-mom1.fits"
file_pbc = "PG0050_CO21-combine-line-10km-mosaic-pbc-mom1.fits"

def load_mom1(path, file):
    # Using procedure
    # mom1, wcs, size, pix_size, hdu, pos_cen = load_mom1(path, file)
    hdu = fits.open(path+file)[0]
    wcs = WCS(hdu.header)
    pos_cen = [395, 399]
    mom1 = hdu.data[0][0] - hdu.data[0][0][395, 399]
    size = 200
    pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    return mom1, wcs, size, pix_size, hdu, pos_cen
def plot_mom1(path, file):
    mom1, wcs, size, pix_size, hdu, pos_cen = load_mom1(path, file)

    level = np.linspace(-220,220,12)
    print(level)
    fig = plt.figure(figsize=(8,10))
    ax = plt.subplot(projection=wcs[0,0])
    im = ax.imshow(mom1, vmin=-220, vmax=220, cmap=plt.cm.get_cmap('coolwarm', 11), origin='lower')
    cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
    cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
    cb.set_label(r'$V_\mathrm{los}$ [$\mathrm{km\,s^{-1}}$]')
    ax.contour(mom1, level, colors=['k'])

    rec = matplotlib.patches.Rectangle((pos_cen[0]-size, pos_cen[1]-size), 10, 10, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2.5)
    ax.add_artist(rec)
    Beam = beam(hdu, pos_cen[0]-size+5,pos_cen[1]-size+5, 'k', pix_size)
    ax.add_artist(Beam[0])

    ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
    ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)
    ax.set_xlabel("R.A. (J2000)", labelpad=0.5, fontsize=20)
    ax.set_ylabel("Dec (J2000)", labelpad=-1.0, fontsize=20)
    #plt.show()
#plt.savefig('/home/qyfei/Desktop/Codes/Result/PG0050/PG0050p0124_mom1.pdf', bbox_inches='tight', dpi=300) 

#plot_mom1(path, file)