from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from map_visualization.maps import *
from skimage.feature import peak_local_max
from matplotlib.colors import LogNorm

path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
file_mom0 = "PG0050_CO21-combine-line-10km-mosaic-mom0-rms.fits"

hdu = fits.open('/home/qyfei/Desktop/Codes/CODES/Physical_values/clouds/PG0050_props.fits')
props = hdu[1].data
xctr = props['XCTR_PIX']
yctr = props['YCTR_PIX']
mom2maj= abs(props['MOMMAJPIX'])
mom2min = abs(props['MOMMINPIX'])
FWHMmaj = np.sqrt(8*np.log(2))*mom2maj
FWHMmin = np.sqrt(8*np.log(2))*mom2min


posang = props['POSANG']*u.rad.to('deg')

from map_visualization.maps import *
mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file_mom0)
mom0_level = np.array([-1,1,2,4,8,16,32,64])*r

plt.figure(figsize=(8,10))
ax = plt.subplot(projection=wcs[0,0])
im = ax.imshow(mom0, cmap='jet', origin='lower', norm=LogNorm())
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
    ## Add beam and annotated information
rec_size = 10
fov = 200
rec = matplotlib.patches.Rectangle((pos_cen[0]-fov, pos_cen[1]-fov), rec_size, rec_size, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax.add_artist(rec)
Beam = beam(hdu, pos_cen[0]-size+rec_size/2,pos_cen[1]-size+rec_size/2, 'k', pix_size)
ax.add_artist(Beam[0])
    ## Set the limit of x- and y-axis
ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)
ax.set_xlabel("R.A. (J2000)", labelpad=0.5, fontsize=20)
ax.set_ylabel("Dec (J2000)", labelpad=-1.0, fontsize=20)

for i in range(len(xctr)):
    ellipse = matplotlib.patches.Ellipse((xctr[i], yctr[i]), FWHMmaj[i], FWHMmin[i], posang[i], fill=False, edgecolor='red', zorder=2)
    ax.add_artist(ellipse)
    ax.text(xctr[i], yctr[i], "%i"%(i+1), color='C1', fontsize=8)

plt.savefig("/home/qyfei/Desktop/Codes/CODES/Physical_values/clouds/PG0050_identify.pdf", bbox_inches="tight", dpi=300)