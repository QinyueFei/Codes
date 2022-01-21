# %%
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import matplotlib
from map_visualization.maps import load_mom1
from scipy import integrate
from matplotlib import colorbar
from map_visualization.maps import beam
from astropy.cosmology import Planck15
plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

# %%
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
file = "PG0050_CO21-combine-line-10km-mosaic-mom1.fits"
file_pbc = "PG0050_CO21-combine-line-10km-mosaic-pbc-mom1.fits"

mom1, wcs, size, pix_size, hdu, pos_cen = load_mom1(path, file)
# %%
size = 100
xslit = np.linspace(pos_cen[1]-40, pos_cen[1]+40, 20)
yslit = np.tan(np.radians(127.59-90))*(xslit-pos_cen[1])+pos_cen[0]

yy, xx = np.mgrid[:800, :800]
radius = np.sqrt((xx-xslit[5])**2 + (yy-yslit[5])**2)
mask = radius<=7

level = np.linspace(-220,220,12)
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

#size = 100
ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)
ax.set_xlabel("R.A. (J2000)", labelpad=0.5, fontsize=20)
ax.set_ylabel("Dec (J2000)", labelpad=-1.0, fontsize=20)

ax.plot(xslit, yslit, "k--", lw=3)
#plt.savefig(output_dir+"CO_vlos.pdf", bbox_inches="tight", dpi=300)

# %%
vrot = np.zeros(len(xslit))
evrot = np.zeros(len(xslit))
for i in range(len(xslit)):
    radius = np.sqrt((xx-xslit[i])**2 + (yy-yslit[i])**2)
    mask = radius<4*np.exp(abs(i-10)/5)
    N = np.where(mask)
    vrot[i] = np.nanmean(mom1[N])
    evrot[i] = np.nanstd(mom1[N])

position = (xslit-pos_cen[1])*0.05/np.cos(np.radians(130))
distance = position/Planck15.arcsec_per_kpc_proper(0.061).value

r_CO = distance[:10]

# %%
plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
ax.errorbar(distance, vrot/np.sin(np.radians(39)), yerr=evrot, fmt='ko', mfc='k', ms=8, mew=1, elinewidth=1, capsize=4)
ax.errorbar(-distance, -vrot/np.sin(np.radians(39)), yerr=evrot, fmt='ks', mfc='none', ms=8, mew=1, elinewidth=1, capsize=4)

#ax.errorbar(r_fit, vrot_fit, yerr=[-evrot1_fit, evrot2_fit], fmt='bo', mfc='none', ms=8, mew=1, elinewidth=1, capsize=4)
#ax.errorbar(-r_fit, -vrot_fit, yerr=[-evrot1_fit, evrot2_fit], fmt='bo', mfc='none', ms=8, mew=1, elinewidth=1, capsize=4)

ax.set_xlabel("Radius [kpc]")
ax.set_ylabel("$V_\mathrm{rot}$ [$\mathrm{km\,s^{-1}}$]")

#plt.savefig(output_dir+"CO_vrot_dbl.pdf", bbox_inches="tight", dpi=300)

# %%
