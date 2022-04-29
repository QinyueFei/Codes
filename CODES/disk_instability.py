# %%
from matplotlib.colors import LogNorm
from Physical_values.surface_density import surface_density, iso_rad, surface_density_mom0
from map_visualization.maps import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
file = "PG0050_CO21-combine-line-10km-mosaic-mom0-pbcor.fits"
file_rms = "PG0050_CO21-combine-line-10km-mosaic-mom0-pbcor.fits"
file_mom2 = "PG0050_CO21-combine-line-10km-mosaic-mom2.fits"
radius = iso_rad(800, 800, np.array([395, 399]), 0.05, 35, 41)

from Physical_values.stellar_properties import Sigma_disk

Sigma, re = np.power(10, 10.64)/40.965, 10.97
Sigma_s = Sigma_disk(Sigma, radius, re)
disp_s = 60
Sigma_H2 = surface_density(path, file_rms, 0.8)
mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file_rms)
mom0_level = np.array([-1,1,2,4,8,16,32,64])*3*r
mom2, wcs, size, pix_size, hdu, pos_cen = load_mom2(path, file_mom2)
disp_CO = np.sqrt(mom2**2 - 9.156**2)
Sigma_tot = Sigma_H2[0] + Sigma_s*2/(1+60**2/mom2**2)

kappa = np.sqrt(2*270**2/radius**2)
center_regions = np.where(radius < 0.8)
kappa[center_regions] = np.sqrt(4*(150/0.8)**2)
#kappa[center_regions] = np.sqrt(270**2/radius[center_regions]**2)
G = 4.302e-06
Q_H2 = mom2*kappa/(np.pi*G*Sigma_H2[0]*1e6)

fig = plt.figure(figsize=(8,10))
ax = plt.subplot(projection=wcs[0,0])
im = ax.imshow(Q_H2, vmin=0, vmax=4, cmap='jet', origin='lower')
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label(r'$Q_\mathrm{tot}$')
#ax.contour(mom0_pbc, mom0_level, linewidths=1, colors=['k'])
ax.contour(Q_H2, [1,1.5,2,2.5,3], colors=['k'], linewidths=[0.5])
ax.set_xlim(pos_cen[0]-size, pos_cen[0]+size)
ax.set_ylim(pos_cen[1]-size, pos_cen[1]+size)
ax.set_xlabel("R.A. (J2000)", labelpad=0.5, fontsize=20)
ax.set_ylabel("Dec (J2000)", labelpad=-1.0, fontsize=20)
ax.contour(radius, [1.0, 2.4],linestyles=['--'],linewidths=2, colors=['w'])
plt.show()

#plt.savefig('/home/qyfei/Desktop/Codes/CODES/Physical_values/clouds/Toomre_Q_tot.pdf', bbox_inches='tight', dpi=300)

# %%

size = 100
fig = plt.figure(figsize=(8,10))
ax = plt.subplot(projection=wcs[0,0])
im = ax.imshow(mom0, cmap='jet', vmin=0.086, vmax=9.0, origin='lower', norm=LogNorm())
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label(r'FLUX')
#ax.contour(mom0_pbc, mom0_level, linewidths=1, colors=['k'])
ax.contour(radius, [0.4, 0.8, 2.1], colors=['k'], linewidths=[0.5])
ax.set_xlim(pos_cen[0]-size, pos_cen[0]+size)
ax.set_ylim(pos_cen[1]-size, pos_cen[1]+size)
ax.set_xlabel("R.A. (J2000)", labelpad=0.5, fontsize=20)
ax.set_ylabel("Dec (J2000)", labelpad=-1.0, fontsize=20)
# ax.contour(radius, [1.0, 2.4],linestyles=['--'],linewidths=2, colors=['w'])
# plt.savefig("/home/qyfei/Desktop/Results/Analysis/radius.pdf", bbox_inches="tight", dpi=300)

# %%
