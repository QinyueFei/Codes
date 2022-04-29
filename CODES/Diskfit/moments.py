# %%
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import Planck15
# %%
path = "/home/qyfei/Desktop/Results/Diskfit/PG_quasars/PG0050"
file = "/OUT/bi.mod.fits"
bihdu = fits.open(path+file)[0]

resfile = "/OUT/bi.res.fits"
reshdu = fits.open(path+resfile)[0]

obsfile = "/PG0050_CO21_velocity.fits"
obshdu = fits.open(path+obsfile)[0]

bidata = bihdu.data/1e3
resdata = reshdu.data/1e3
obsdata = obshdu.data[0][0]/1e3

# %%
VSYS = 17267.22
XPOS, YPOS = 399.14, 396.81
PA = -57.19
x = np.linspace(0,1000) 
k = np.tan(np.deg2rad(90+PA))
y = k*(x-XPOS)+YPOS
y_perp = np.tan(np.deg2rad(PA))*(x-XPOS)+YPOS

size = 100
los_level = np.arange(-1000, 1000, 40)
res_level = np.arange(-500, 500, 10)
fig, axes = plt.subplots(figsize=(18, 7), nrows=1, ncols=3)
plt.subplots_adjust(wspace=0)
ax0, ax1, ax2 = axes
im0 = ax0.imshow(obsdata-VSYS, vmin=-260, vmax=260, cmap="coolwarm", origin="lower")
ax0.contour(obsdata-VSYS, los_level, colors=["k"], linewidths=0.5)
ax0.text(XPOS-size+5, YPOS-size+5, "DATA")
im1 = ax1.imshow(bidata-VSYS, vmin=-260, vmax=260, cmap="coolwarm", origin="lower")
ax1.contour(bidata-VSYS, los_level, colors=["k"], linewidths=0.5)
ax1.text(XPOS-size+5, YPOS-size+5, "MODEL")
im2 = ax2.imshow(resdata, vmin=-60, vmax=60, cmap="coolwarm", origin="lower")
ax2.contour(resdata, res_level, colors=["k"], linewidths=0.5)

for ax in axes[:]:
    ax.set_xlim(XPOS-size,XPOS+size)
    ax.set_ylim(YPOS-size,YPOS+size)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

for ax in axes[:2]:
    ax.scatter(XPOS, YPOS, c='w', edgecolor='k', zorder=3, linewidth=0.9, s=100, marker='*')
    ax.plot(x, y,'-.',linewidth=1.5, c='k', alpha=0.8)
    ax.plot(x, y_perp,':',linewidth=1.5, c='k', alpha=0.8)

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.125, 0.115, 0.517, 0.05])
cb_ax = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
cb_ax.set_label(r"$V_\mathrm{LOS}$ [$\mathrm{km\,s^{-1}}$]")

cbar_res = fig.add_axes([0.643, 0.115, 0.257, 0.05])
cb_res = fig.colorbar(im2, cax=cbar_res, orientation='horizontal')
cb_res.set_label(r"$V_\mathrm{res}$ [$\mathrm{km\,s^{-1}}$]")

# plt.savefig(path+"/bimodel.pdf", bbox_inches="tight", dpi=300)
# %%
