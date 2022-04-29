# %%
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
# from Barolo_analyse.parameters import load_parameters
import astropy.units as u
# parameters = load_parameters(path, folder, file)
# r_fit, rad_fit, vrot_fit, evrot1_fit, evrot2_fit, vdisp_fit, edisp1_fit, edisp2_fit, inc_fit, pa_fit, vrad_fit, vsys_fit, vrad, xpos,YPOS, x, y, y_perp, vcirc_fit, einc1_fit, einc2_fit, epa1_fit, epa2_fit = parameters

# %%
# obs = 'line/'
# path = '/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/'
# folder = "Barolo_fit/output/PG0050+124_best/"
# object = "PG1011"
name = "PG1244+026"

path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/" + object
folder = "/Barolo_fit/CO32/output/" + name
file = "/ringlog2.txt"

RAD_kpc, RAD_arcs, VROT, DISP, INC, PA, Z0, Z0_arcs, SIG, XPOS, YPOS, VSYS, VRAD, E_VROT1, E_VROT2, E_DISP1, E_DISP2  = np.genfromtxt(path+folder+file,skip_header=1,unpack=True) #, E_VRAD1, E_VRAD2
XPOS, YPOS = np.nanmean(XPOS), np.nanmean(YPOS)
# E_INC1, E_INC2, E_PA1, E_PA2, E_XPOS1, E_XPOS2, E_YPOS1, E_YPOS2, E_VSYS1, E_VSYS2
x = np.linspace(0,1000) 
k = np.tan(np.deg2rad(90+np.mean(PA)))
y = k*(x-XPOS)+YPOS
y_perp = np.tan(np.deg2rad(np.mean(PA)))*(x-XPOS)+YPOS

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

    hdu = fits.open(path+folder+name+'_0mom.fits')[0]
    hdr = hdu.header
    return obs_moms, mod_moms, res_moms

# path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/"
# folder = "Barolo_fit/output/PG0050+124_best/maps/"
# file = "PG0050+124"
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/"+object
folder = "/Barolo_fit/CO32/output/"+name+"/maps/"
file = name

obs_mom0, obs_mom1, obs_mom2 = mom_compare(path, folder, file)[0]
mod_mom0, mod_mom1, mod_mom2 = mom_compare(path, folder, file)[1]
res_mom0, res_mom1, res_mom2 = mom_compare(path, folder, file)[2]

# %%
## Plot the velocity

size = 50
los_level = np.arange(-1000, 1000, 40)
res_level = np.arange(-500, 500, 10)
fig, axes = plt.subplots(figsize=(18, 7), nrows=1, ncols=3)
plt.subplots_adjust(wspace=0)
ax0, ax1, ax2 = axes
im0 = ax0.imshow(obs_mom1-VSYS[0], vmin=-50, vmax=50, cmap="coolwarm", origin="lower")
ax0.contour(obs_mom1-VSYS[0], los_level, colors=["k"], linewidths=0.5)
ax0.text(XPOS-size+5, YPOS-size+5, "DATA")
im1 = ax1.imshow(mod_mom1-VSYS[0], vmin=-50, vmax=50, cmap="coolwarm", origin="lower")
ax1.contour(mod_mom1-VSYS[0], los_level, colors=["k"], linewidths=0.5)
ax1.text(XPOS-size+5, YPOS-size+5, "MODEL")
im2 = ax2.imshow(res_mom1, vmin=-20, vmax=20, cmap="coolwarm", origin="lower")
ax2.contour(res_mom1, res_level, colors=["k"], linewidths=0.5)

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

#plt.show()
# plt.savefig('/home/qyfei/Desktop/Results/Barolo/PG_quasars/'+object+'/fit_mom1_CO32.pdf', bbox_inches='tight', dpi=300)

# %%
size = 50
disp_level = np.arange(5, 1005, 20)
res_level = np.arange(-400, 400, 10)
fig, axes = plt.subplots(figsize=(18, 7), nrows=1, ncols=3)
plt.subplots_adjust(wspace=0)
ax0, ax1, ax2 = axes
im0 = ax0.imshow(obs_mom2, vmin=5, vmax=35, cmap="jet", origin='lower')
ax0.contour(obs_mom2, disp_level, colors=['k'], linewidths=0.5)
ax0.text(XPOS-size+5, YPOS-size+5, "DATA")
im1 = ax1.imshow(mod_mom2, vmin=5, vmax=35, cmap="jet", origin='lower')
ax1.contour(mod_mom2, disp_level, colors=['k'], linewidths=0.5)
ax1.text(XPOS-size+5, YPOS-size+5, "MODEL")
im2 = ax2.imshow(res_mom2, vmin=-20, vmax=20, cmap="jet", origin='lower')
ax2.contour(res_mom2, res_level, colors=['k'], linewidths=0.5)

for ax in axes[:]:
    ax.set_xlim(XPOS-size,XPOS+size)
    ax.set_ylim(YPOS-size,YPOS+size)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.125, 0.115, 0.517, 0.05])
cb_ax = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
cb_ax.set_label(r"$\sigma$ [$\mathrm{km\,s^{-1}}$]")

cbar_res = fig.add_axes([0.643, 0.115, 0.257, 0.05])
cb_res = fig.colorbar(im2, cax=cbar_res, orientation='horizontal')
cb_res.set_label(r"$\sigma_{res}$ [$\mathrm{km\,s^{-1}}$]")

# plt.savefig('/home/qyfei/Desktop/Results/Barolo/PG_quasars/'+object+'/fit_mom2_CO32.pdf', bbox_inches='tight', dpi=300)

# %%
plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
ax.imshow(obs_mom0, cmap='coolwarm', origin='lower')
ax.contour(res_mom1, level = res_level, colors='k')
ax.set_xlim(XPOS-size,XPOS+size)
ax.set_ylim(YPOS-size,YPOS+size)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])

# %%
