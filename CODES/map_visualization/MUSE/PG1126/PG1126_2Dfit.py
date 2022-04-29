# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sympy import approximants
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colorbar as colorbar
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from map_visualization.maps import beam
from astropy.cosmology import Planck15

#%matplotlib inline
plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

# %%
## laod the data
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG1126/Barolo_fit/MUSE_Ha/output/NONE"
ring_file = "/NONE_2dtrm.txt"
Radius, Radius, Vsys, Vrot, Vexp, PA, Incl, Xcenter, Ycenter = np.genfromtxt(path+ring_file, skip_header=2,usecols=(0,1,2,3,4,5,6,7,8),unpack=True)

# %%
## find the galaxy center
plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
ax.scatter(Radius, Xcenter, marker = 'o', edgecolors='k', color='r', s=50, label='X')
ax.scatter(Radius, Ycenter, marker = 'o', edgecolors='k', color='b', s=50, label='Y')
ax.set_xlabel('Radius [arcsec]')
ax.set_ylabel('Pixels')
ax.vlines(8, 0, 180, color='k', ls='--')
ax.vlines(1.5, 0, 180, color='k', ls='--')
ax.set_ylim(120, 180)
ax.set_xlim(0, 10)
N = np.where((Radius>1.5) & (Radius<8))
ax.hlines(np.nanmean(Xcenter[N]), 0, 20, color='r', ls='-.')
ax.hlines(np.nanmean(Ycenter[N]), 0, 20, color='b', ls='-.')
ax.text(1.5, np.nanmean(Xcenter[N]), "%.2f"%np.nanmean(Xcenter[N]), color='r', ha='left', va='top', fontsize=40)
ax.text(1.5, np.nanmean(Ycenter[N]), "%.2f"%np.nanmean(Ycenter[N]), color='b', ha='left', va='bottom', fontsize=40)
plt.legend()
# plt.savefig("/home/qyfei/Desktop/Results/Barolo/PG_quasars/PG1126/MUSE/pos_prof.pdf", bbox_inches="tight", dpi=300)

# %%
## represent the position angle
plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
ax.scatter(Radius, PA, marker = 'o', edgecolors='k', color='r', s=50, label='PA')
ax.set_xlabel('Radius [arcsec]')
ax.set_ylabel('PA [deg]')
ax.vlines(8, 0, 1000, color='k', ls='--')
ax.vlines(1.5, 0, 1000, color='k', ls='--')
ax.set_ylim(120, 180)
ax.set_xlim(0, 10)
N = np.where((Radius>1.5) & (Radius<8))
ax.hlines(np.nanmean(PA[N]), 0, 20, color='r', ls='-.')
ax.text(1.5, np.nanmean(PA[N]), "%.2f"%np.nanmean(PA[N]), color='r', fontsize=40)
plt.legend()
# plt.savefig("/home/qyfei/Desktop/Results/Barolo/PG_quasars/PG1126/MUSE/pa_prof.pdf", bbox_inches="tight", dpi=300)

# %%
## find the best systemic velocity
plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
ax.scatter(Radius, Vsys, marker = 'o', edgecolors='k', color='r', s=50, label='$V_\mathrm{sys}$')
ax.set_xlabel('Radius [arcsec]')
ax.set_ylabel('$V_\mathrm{sys}$ [$\mathrm{km\,s^{-1}}$]')
ax.vlines(8, -1000, 1000, color='k', ls='--')
ax.vlines(1.5, -1000, 1000, color='k', ls='--')
ax.set_ylim(-35, 35)
ax.set_xlim(0, 10)
N = np.where((Radius>1.5) & (Radius<8))
ax.hlines(np.nanmean(Vsys[N]), 0, 1000, color='r', ls='-.')
ax.text(1.5, np.nanmean(Vsys[N]), "%.2f"%np.nanmean(Vsys[N]), color='r', fontsize=40)
plt.legend()
# plt.savefig("/home/qyfei/Desktop/Results/Barolo/PG_quasars/PG1126/MUSE/vsys_prof.pdf", bbox_inches="tight", dpi=300)

# %%
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG1126/Barolo_fit/MUSE_Ha/output/NONE"
mod_file = "/NONE_2d_mod.fits"
dat_file = "/../../PG1126_VlosHa.fits"

# %%
## model and data
mod_hdu = fits.open(path+mod_file)
mod_map = mod_hdu[0].data

dat_hdu = fits.open(path+dat_file)
dat_map = dat_hdu[0].data

res_map = mod_map - dat_map

# %%
## plot the line-of-sight velocity map of model

size = 60
fov = size #define the size of the map
r = 0.20
pix_size = 0.20
# pos_cen = [159, 157.5]
phase_cen = [158, 161]
pos_cen = [152.13, 160.30]

rad = np.array([4.0, 10])*u.kpc*Planck15.arcsec_per_kpc_proper(0.060)

level = np.arange(-200, 220, 20)
print(level)
fig = plt.figure(figsize=(8,10))
ax = plt.subplot(111)#projection=wcs[0,0])
im = ax.imshow(mod_map, vmin=level[0]+5, vmax=level[-1]-5, cmap='coolwarm', origin='lower')
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label(r'$V_\mathrm{los}$ [$\mathrm{km\,s^{-1}}$]')
# ax.contour(mod_map, level, linewidths=0.5, colors=['k'])
ax.contour(dat_map, level, linewidths=0.5, colors=['k'])
ax.contour(radius, levels = rad.value, linewidths=3.5, colors=['b','k'], linestyles=['--','-'])
ax.contour(vsigHa, level = np.arange(0, 160, 20), linewidths=1.8, colors=['magenta'])


# ax.scatter(phase_cen[0], phase_cen[1], marker='+', s=100, color='b')
# ax.scatter(164.77, 156.33, marker='x', s=100, color='k')
ax.scatter(pos_cen[0], pos_cen[1], marker='x', s=250, color='k')
bmaj, bmin, bpa, delt = 1.0*u.arcsec.to('deg'), 1.0*u.arcsec.to('deg'), 90.0, 0.20*u.arcsec.to('deg')
rec_size = abs(bmaj/delt)*3.0


rec = matplotlib.patches.Rectangle((pos_cen[0]-fov, pos_cen[1]-fov), rec_size, rec_size, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax.add_artist(rec)
# Beam = beam(hdu, pos_cen[0]-size+rec_size/2,pos_cen[1]-size+rec_size/2, 'k', pix_size)
Beam = matplotlib.patches.Ellipse((pos_cen[0]-fov+rec_size/2, pos_cen[1]-fov+rec_size/2), bmaj/delt, bmin/delt, angle=0.0,fill=True, edgecolor='k', facecolor='k', zorder=3)
ax.add_artist(Beam)

redshift = 0.060
scale_bar_length = 5.0*u.kpc*Planck15.arcsec_per_kpc_proper(redshift)
bar_size = abs(scale_bar_length.to('deg').value/delt)*1.5
rec = matplotlib.patches.Rectangle((pos_cen[0]+fov-bar_size, pos_cen[1]-fov), bar_size, rec_size, angle=0.0, fill=True, edgecolor='k', facecolor='w', zorder=2)
ax.add_artist(rec)
bar_length = abs(scale_bar_length.to('deg').value/delt)
scale_bar = matplotlib.patches.Rectangle((pos_cen[0]+fov-(bar_length+bar_size)/2, pos_cen[1]-fov+rec_size*3/5), bar_length, 2, angle=0.0, fill=True, edgecolor='k', facecolor='k', zorder=3)
ax.add_artist(scale_bar)

ax.text(pos_cen[0]+fov-bar_size/2, pos_cen[1]-fov+rec_size/6, '5kpc', ha='center')
    
ax.text(pos_cen[0]+fov-rec_size/6, pos_cen[1]+fov-rec_size/6, 'PG\,1126-041', color='k', ha='right', va='top', fontsize=30)

ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)
ax.set_xticks([])
ax.set_yticks([])
# plt.savefig("/home/qyfei/Desktop/Results/Barolo/PG_quasars/PG1126/MUSE/vlos_present.pdf", bbox_inches="tight", dpi=300)

# %%
## plot the residual of line-of-sight velocity
size = 60
fov = size #define the size of the map
r = 0.20
pix_size = 0.20
# pos_cen = [159, 157.5]

level = np.arange(-60, 80, 20)
print(level)
fig = plt.figure(figsize=(8,10))
ax = plt.subplot(111)#projection=wcs[0,0])
im = ax.imshow(res_map, vmin=level[0]+5, vmax=level[-1]-5, cmap='coolwarm', origin='lower')
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label(r'$V_\mathrm{los}$ [$\mathrm{km\,s^{-1}}$]')
ax.contour(res_map, level, linewidths=0.5, colors=['k'])
# ax.contour(vsigHa, level = np.arange(0, 160, 20), linewidths=1.2, colors=['k'])
ax.contour(radius, levels = rad.value, linewidths=3.5, colors=['b','k'], linestyles=['--','-'])
ax.contour(vsigHa, level = np.arange(0, 160, 20), linewidths=1.8, colors=['magenta'])

ax.scatter(pos_cen[0], pos_cen[1], marker='x', s=250, color='k')
bmaj, bmin, bpa, delt = 1.0*u.arcsec.to('deg'), 1.0*u.arcsec.to('deg'), 90.0, 0.20*u.arcsec.to('deg')
rec_size = abs(bmaj/delt)*3.0


rec = matplotlib.patches.Rectangle((pos_cen[0]-fov, pos_cen[1]-fov), rec_size, rec_size, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax.add_artist(rec)
# Beam = beam(hdu, pos_cen[0]-size+rec_size/2,pos_cen[1]-size+rec_size/2, 'k', pix_size)
Beam = matplotlib.patches.Ellipse((pos_cen[0]-fov+rec_size/2, pos_cen[1]-fov+rec_size/2), bmaj/delt, bmin/delt, angle=0.0,fill=True, edgecolor='k', facecolor='k', zorder=3)
ax.add_artist(Beam)

redshift = 0.060
scale_bar_length = 5.0*u.kpc*Planck15.arcsec_per_kpc_proper(redshift)
bar_size = abs(scale_bar_length.to('deg').value/delt)*1.5
rec = matplotlib.patches.Rectangle((pos_cen[0]+fov-bar_size, pos_cen[1]-fov), bar_size, rec_size, angle=0.0, fill=True, edgecolor='k', facecolor='w', zorder=2)
ax.add_artist(rec)
bar_length = abs(scale_bar_length.to('deg').value/delt)
scale_bar = matplotlib.patches.Rectangle((pos_cen[0]+fov-(bar_length+bar_size)/2, pos_cen[1]-fov+rec_size*3/5), bar_length, 2, angle=0.0, fill=True, edgecolor='k', facecolor='k', zorder=3)
ax.add_artist(scale_bar)

ax.text(pos_cen[0]+fov-bar_size/2, pos_cen[1]-fov+rec_size/6, '5kpc', ha='center')
    
ax.text(pos_cen[0]+fov-rec_size/6, pos_cen[1]+fov-rec_size/6, 'PG\,1126-041', color='k', ha='right', va='top', fontsize=30)

ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)
# ax.set_xlabel("R.A. (J2000)", labelpad=0.5, fontsize=20)
# ax.set_ylabel("Dec (J2000)", labelpad=-1.0, fontsize=20)
ax.set_xticks([])
ax.set_yticks([])

# plt.savefig("/home/qyfei/Desktop/Results/Barolo/PG_quasars/PG1126/MUSE/res_map.pdf", bbox_inches="tight", dpi=300)

# %%
def iso_rad(sizex_, sizey_, pos_cen_, pix_size_, PA_, inc_):
    sizex, sizey, pos_cen, pix_size, PA, inc = sizex_, sizey_, pos_cen_, pix_size_, PA_, inc_
    # This function calculate the radius between each pixel and the kinematic center
    # size, pos_cen, pix_size are adopted from observation, which are map size, coordinates of galaxy center and size of each pixel
    # PA, inc are position angle and inclination angle
    z = 0.060
    yy,xx = np.indices([sizey, sizex],dtype='float')
    coordinates_xx = (xx-pos_cen[0])*np.cos(PA*u.deg).value + (yy-pos_cen[1])*np.sin(PA*u.deg).value
    coordinates_yy = -(xx-pos_cen[0])*np.sin(PA*u.deg).value + (yy-pos_cen[1])*np.cos(PA*u.deg).value
    Radius_pixel = np.sqrt(coordinates_xx**2 + coordinates_yy**2/(np.cos(inc*u.deg).value)**2)
    Radius = (Radius_pixel * pix_size)*u.arcsec#/Planck15.arcsec_per_kpc_proper(z)).to('kpc')
    return Radius.value

# %%
radius = iso_rad(316, 322, (pos_cen[0], pos_cen[1]), 0.20, 145.31+90, 71)
ring_bound = np.arange(0, 9.75, 0.5)

ring_res = np.zeros(len(ring_bound) - 1)
for i in range(len(ring_bound)-1):
    N = np.where((radius>=ring_bound[i]) & (radius<ring_bound[i+1]))
    ring_res[i] = np.sqrt(np.nansum(res_map[N]**2)/len(N[0]))
## save the radius in units of arcsec, rotation velocity and 
# np.savetxt("/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG1126/Barolo_fit/MUSE_Ha/output/NONE/vel_prof.txt", np.array([Radius, Vrot, ring_res]).T)

# %%
norm = 1*u.arcsec/Planck15.arcsec_per_kpc_proper(0.060)
plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
# ax.scatter(Radius * norm.value, Vrot, marker = 'o', edgecolors='k', color='r', s=50, label='$V_\mathrm{rot}$')
ax.errorbar(Radius * norm.value, Vrot, yerr = ring_res, fmt='ro', mfc='r', ms=10, mew=1, elinewidth=1, capsize=5., label='$V_\mathrm{rot}$')
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel(r'$V_\mathrm{rot}$ [$\mathrm{km\,s^{-1}}$]')
ax.vlines(15 * norm.value, 0, 1000, color='k', ls='--')
ax.vlines(1.5 * norm.value, 0, 1000, color='k', ls='--')

ax.set_ylim(0, 350)
ax.set_xlim(0, 12.5)
# N = np.where((Radius>1.5) & (Radius<15))
# ax.hlines(np.nanmean(PA[N]), 0, 20, color='r', ls='-.')
# ax.hlines(np.nanmean(Ycenter[N]), 0, 20, color='b', ls='-.')
# ax.text(1.5, np.nanmean(PA[N]), "%.2f"%np.nanmean(PA[N]), color='k')
# ax.text(1.5, np.nanmean(Ycenter[N]), "%.2f"%np.nanmean(Ycenter[N]), color='k', ha='left', va='top')
plt.legend()

# %%
## Load 2D fit of Ha
Radius, Vrot, ring_res = np.genfromtxt("/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG1126/Barolo_fit/MUSE_Ha/output/NONE/vel_prof.txt", skip_header=0, unpack=True)

## Load 3D fit of CO(2-1)
object = "PG1126"
name = "PG1126-041"
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/"+object
folder = "/Barolo_fit/output/"+name#+"_well"

RAD_kpc, RAD_arcs, VROT, DISP, INC, PA, Z0, Z0_arcs, SIG, XPOS, YPOS, VSYS, VRAD, E_VROT1, E_VROT2, E_DISP1, E_DISP2, E_INC1, E_INC2, E_PA1, E_PA2  = np.genfromtxt(path+folder+"/ringlog1.txt",skip_header=1,unpack=True)

RAD_kpc, RAD_arcs, VROT, DISP, INC, PA, Z0, Z0_arcs, SIG, XPOS, YPOS, VSYS, VRAD, E_VROT1, E_VROT2, E_DISP1, E_DISP2 = np.genfromtxt(path+folder+"/ringlog2.txt",skip_header=1,unpack=True)

# %%
plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

fig, axes = plt.subplots(figsize=(12, 12), nrows=1, ncols=1)
ax0 = axes
msize=5
capsize = 5

ax0.errorbar(RAD_kpc, VROT, yerr=[-E_VROT1, E_VROT2], fmt='bo', mfc='b', ms=msize, mew=1, elinewidth=1, capsize=capsize, label="CO(2-1)", zorder=3)
# ax0.scatter(RAD_kpc, VROT, marker='o', color='b', s=10)
# ax0.fill_between(RAD_kpc, VROT+E_VROT1, VROT+E_VROT2, color='b', alpha=0.3, zorder=3)
# ax0.errorbar(RAD_kpc, VROT, yerr=ring_res_CO, fmt='ks', mfc='none', ms=msize, mew=1, elinewidth=1, capsize=5, label="$V_\mathrm{rot}$", zorder=3)

ax0.errorbar(Radius * norm.value, Vrot, yerr = ring_res, fmt='ro', mfc='r', ms=msize, mew=1, elinewidth=1, capsize=capsize, label=r'$\mathrm{H \alpha}$')
# ax0.scatter(Radius * norm.value, Vrot, marker='o', color='r', s=10)
# ax0.fill_between(Radius * norm.value, Vrot-ring_res, Vrot+ring_res, color='r', alpha=0.3, zorder=3)

ax0.set_xlabel('Radius [kpc]')
ax0.set_ylabel(r'$V_\mathrm{rot}$ [$\mathrm{km\,s^{-1}}$]')
# ax0.vlines(4, 0, 1000, color='k', ls='-.', lw=0.7)
ax0.vlines(10, 0, 1000, color='k', ls='-.', lw=0.7)
ax0.fill_between(x=(.0, 4.), y1=0, y2=1000, color='yellow', alpha=0.2, zorder=1)

ax0.set_xlim(1e-2, 11.5)
ax0.set_ylim(0, 365)
ax0.grid()
ax0.fill_between([0, 2*(RAD_kpc[1]-RAD_kpc[0])], [-500], [500], color='k', hatch='/', alpha=0.3, zorder=3)
plt.legend()
ax0.text(0.5, 350, "PG\,1126-040", ha='left', va='top', fontsize=40)

plt.rc('font', family='dejavuserif', size=15)
# left, bottom, width, height = 0.3, 0.16, 0.35, 0.35
# ax1 = plt.axes([left, bottom, width, height])
# ax1.errorbar(RAD_kpc, VROT, yerr=[-E_VROT1, E_VROT2], fmt='bo', mfc='b', ms=msize, mew=1, elinewidth=1, capsize=capsize, zorder=3)# %%
# ax1.errorbar(Radius * norm.value, Vrot, yerr = ring_res, fmt='ro', mfc='r', ms=msize, mew=1, elinewidth=1, capsize=capsize)
# ax1.set_xlim(0, 3.99)
# ax1.set_ylim(180, 320)
# plt.savefig("/home/qyfei/Desktop/Results/Barolo/PG_quasars/PG1126/MUSE/rot_cur_tot.pdf", bbox_inches="tight", dpi=300)

# %%
