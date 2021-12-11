# %%
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from Barolo_analyse.parameters import load_parameters

obs = 'line/'
path = '/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/'
folder = "Barolo_fit/output/PG0050+124_best/"
file = "ringlog2.txt"
parameters = load_parameters(path, folder, file)
r_fit, rad_fit, vrot_fit, evrot1_fit, evrot2_fit, vdisp_fit, edisp1_fit, edisp2_fit, inc_fit, pa_fit, vrad_fit, vsys_fit, vrad, xpos,ypos, x, y, y_perp, vcirc_fit, einc1_fit, einc2_fit, epa1_fit, epa2_fit = parameters

# %%
def mom_compare(path, folder, file):
    obs_moms = []
    mod_moms = []
    res_moms = []
    for i in ["0", "1", "2"]:
        obs_mom, mod_mom = fits.open(path+folder+file+'_'+i+'mom.fits')[0].data, fits.open(path+folder+file+'_local_'+i+'mom.fits')[0].data
        obs_moms.append(obs_mom)
        mod_moms.append(mod_mom)
        res_moms.append(obs_mom- mod_mom)

    hdu = fits.open(path+folder+'PG0050+124_0mom.fits')[0]
    hdr = hdu.header
    return obs_moms, mod_moms, res_moms

path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/"
folder = "Barolo_fit/output/PG0050+124_best/maps/"
file = "PG0050+124"
obs_mom0, obs_mom1, obs_mom2 = mom_compare(path, folder, file)[0]
mod_mom0, mod_mom1, mod_mom2 = mom_compare(path, folder, file)[1]
res_mom0, res_mom1, res_mom2 = mom_compare(path, folder, file)[2]

# %%
## Plot the velocity
size = 100
los_level = np.linspace(-240, 240, 13)
res_level = np.linspace(-60, 60, 13)
fig, axes = plt.subplots(figsize=(18, 7), nrows=1, ncols=3)
plt.subplots_adjust(wspace=0)
ax0, ax1, ax2 = axes
im0 = ax0.imshow(obs_mom1, vmin=-240, vmax=240, cmap=plt.cm.get_cmap('coolwarm', 12), origin='lower')
ax0.contour(obs_mom1, los_level, colors=['k'], linewidths=0.5)
ax0.text(310, 310, "DATA")
im1 = ax1.imshow(mod_mom1, vmin=-240, vmax=240, cmap=plt.cm.get_cmap('coolwarm', 12), origin='lower')
ax1.contour(mod_mom1, los_level, colors=['k'], linewidths=0.5)
ax1.text(310, 310, "MODEL")
im2 = ax2.imshow(res_mom1, vmin=-60, vmax=60, cmap=plt.cm.get_cmap('coolwarm', 10), origin='lower')
ax2.contour(res_mom1, res_level, colors=['k'], linewidths=0.5)

for ax in axes[:]:
    ax.set_xlim(xpos-size,xpos+size)
    ax.set_ylim(ypos-size,ypos+size)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

for ax in axes[:2]:
    ax.scatter(xpos, ypos, c='w', edgecolor='k', zorder=3, linewidth=0.9, s=100, marker='*')
    ax.plot(x, y,'-.',linewidth=1.5, c='k', alpha=0.8)
    ax.plot(x, y_perp,':',linewidth=1.5, c='k', alpha=0.8)

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.125, 0.115, 0.517, 0.05])
cb_ax = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
cb_ax.set_label(r"$V_\mathrm{LOS}$ [$\mathrm{km\,s^{-1}}$]")

cbar_res = fig.add_axes([0.643, 0.115, 0.257, 0.05])
cb_res = fig.colorbar(im2, cax=cbar_res, orientation='horizontal')
cb_res.set_label(r"$V_\mathrm{res}$ [$\mathrm{km\,s^{-1}}$]")

#plt.show()
#plt.savefig('/home/qyfei/Desktop/Results/Barolo/PG0050/fit_mom1.pdf', bbox_inches='tight', dpi=300)

# %%
