# %%
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import Planck15
from Physical_values.surface_density import iso_rad
# %%
path = "/home/qyfei/Desktop/Results/Barolo/PG0050/Ha/output/NONE/"
file = "NONE_2dtrm.txt"

FIT2D = np.loadtxt(path+file)

# %%

rad2D = FIT2D[:,1]
rfit2D = rad2D*(u.arcsec/Planck15.arcsec_per_kpc_proper(0.061)).value
VSYS2D = FIT2D[:,2]
Vrot2D = FIT2D[:,3]
eVrot2D = 20*Vrot2D/Vrot2D
PA2D = FIT2D[:,5]
INC2D = FIT2D[:,6]
Xcen2D = FIT2D[:,7]
Ycen2D = FIT2D[:,8]

# %%
file = "NONE_2d_mod.fits"
model2d = fits.open(path+file)[0].data

# %%
res2d = vlosHa - model2d
# %%
Vsys = VSYS2D[0]

obs_mom1 = vlosHa-Vsys
mod_mom1 = model2d - Vsys
res_mom1 = res2d
xpos, ypos = 165, 164

size = 100
los_level = np.linspace(-240, 240, 13)
res_level = np.linspace(-60, 60, 13)
fig, axes = plt.subplots(figsize=(18, 7), nrows=1, ncols=3)
plt.subplots_adjust(wspace=0)
ax0, ax1, ax2 = axes
im0 = ax0.imshow(obs_mom1, vmin=-240, vmax=240, cmap=plt.cm.get_cmap('coolwarm', 12), origin='lower')
ax0.contour(obs_mom1, los_level, colors=['k'], linewidths=0.5)
ax0.text(75, 75, "DATA")
im1 = ax1.imshow(mod_mom1, vmin=-240, vmax=240, cmap=plt.cm.get_cmap('coolwarm', 12), origin='lower')
ax1.contour(mod_mom1, los_level, colors=['k'], linewidths=0.5)
#ax1.contour(Ha_radius, [0, 2,4,6,8,10,12,14], colors=['k'], linewidths=0.5)
ax1.text(75, 75, "MODEL")
im2 = ax2.imshow(res_mom1, vmin=-60, vmax=60, cmap=plt.cm.get_cmap('coolwarm', 10), origin='lower')
ax2.contour(res_mom1, res_level, colors=['k'], linewidths=0.5)

for ax in axes[:]:
    ax.set_xlim(xpos-size,xpos+size)
    ax.set_ylim(ypos-size,ypos+size)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

for ax in axes[:]:
    ax.scatter(xpos, ypos, c='r', edgecolor='k', zorder=3, linewidth=0.9, s=150, marker='*')
    #ax.plot(x, y,'-.',linewidth=1.5, c='k', alpha=0.8)
    #ax.plot(x, y_perp,':',linewidth=1.5, c='k', alpha=0.8)

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.125, 0.115, 0.517, 0.05])
cb_ax = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
cb_ax.set_label(r"$V_\mathrm{LOS}$ [$\mathrm{km\,s^{-1}}$]")

cbar_res = fig.add_axes([0.643, 0.115, 0.257, 0.05])
cb_res = fig.colorbar(im2, cax=cbar_res, orientation='horizontal')
cb_res.set_label(r"$V_\mathrm{res}$ [$\mathrm{km\,s^{-1}}$]")
#plt.savefig(output_dir+"Ha_2DFIT.pdf", bbox_inches="tight", dpi=300)
# %%
Ha_radius = iso_rad(331, 324, np.array([164, 165]), 0.2, 35, 41)
plt.figure(figsize=(10, 8))
plt.imshow(Ha_radius, cmap='jet', origin='lower')
plt.colorbar()
# %%
evrot2D = np.zeros(len(rfit2D))
for i in range(len(rfit2D)):
    mask = (Ha_radius>=rfit2D[i]-0.5) & (Ha_radius<rfit2D[i]+0.5)
    N = np.where(mask)
    evrot2D[i] = np.nanmean(evlosHa[N])*3

# %%
output_dir = "/home/qyfei/Desktop/Results/Barolo/PG0050/Ha/"
r_fit_tot = np.array(list(r_fit[7:])+list(rfit2D[5:]))
vrot_tot = np.array(list(vrot_fit)[7:]+list(Vrot2D[5:]))
evrot_tot = np.array(list(evrot2_fit)[7:]+list(evrot2D[5:]))

plt.figure(figsize=(8, 8)),
ax = plt.subplot(111)
ax.errorbar(rfit2D[5:], Vrot2D[5:], yerr=evrot2D[5:], fmt='ro', mfc='r', ms=8, mew=1, elinewidth=1, capsize=4, label=r'$\mathrm{H}\alpha$')
ax.errorbar(r_fit, vrot_fit, yerr=[-evrot1_fit, evrot2_fit], fmt='bo', mfc='b', ms=8, mew=1, elinewidth=1, capsize=4, label='CO'),

ax.errorbar(r_fit_tot, vrot_tot, yerr=evrot_tot, fmt='ks', mfc='none', ms=8, mew=1, elinewidth=1, capsize=4)

ax.plot(x_test, v_b_test, "red", lw=2, label="Bulge")
ax.plot(x_test, v_d_test, "blue", lw=2, label="Disk")
ax.plot(x_test, v_dh_test, "yellow", lw=2, label="DM")
ax.plot(x_test, v_g_test*np.sqrt(alpha_CO), "green", lw=2, label="Gas")
#v_tot_test = np.sqrt(v_b_test**2 + v_d_test**2 + v_dh_test**2 + v_g_test**2*alpha_CO)
ax.plot(x_test, v_tot_test, "Grey", lw=5, zorder=3, label="Total")

ax.set_xlabel("Radius [kpc]")
ax.set_ylabel("$V_\mathrm{rot}$ [$\mathrm{km\,s^{-1}}$]")
ax.set_ylim(150, 350)
ax.set_xlim(0.5, 15)
ax.vlines(0.86, 0, 400, color='k', ls=':')
ax.semilogx()
plt.legend()
#plt.savefig(output_dir+"vrot.pdf", bbox_inches="tight", dpi=300)

# %%
