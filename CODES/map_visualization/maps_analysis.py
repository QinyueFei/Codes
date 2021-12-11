# %%
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
from reproject.interpolation.high_level import reproject_interp
#from Dynamics.models import f
#from Dynamics.models import Rs
import astropy.units as u
import matplotlib
from scipy import integrate
import scipy
from scipy.ndimage import interpolation

plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
file = "PG0050_CO21-combine-line-10km-mosaic-mom0.fits"
#name_pbc = "PG0050_CO21-combine-line-10km-mosaic-pbc-mom0-rms.fits"

from map_visualization.moment0 import load_mom0
from map_visualization.maps import beam
from map_visualization.fitting.module import Disk2D, Gauss2D
mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file)

# %%
# azimuthal averaging, f vs. radius
from Physical_values.surface_density import *
radius = iso_rad(800, np.array([395, 399]), 0.05, 35, 41)
radius_model = iso_rad(200, np.array([100, 100]), 0.05, 35, 41)
output_dir = "/home/qyfei/Desktop/Results/map_visualization/Non_para/PG0050/"
distance = np.linspace(0, 3., 21)
f_r = []
ef_r = []
for i in range(len(distance)-1):
    mask = (radius>=distance[i]) & (radius<distance[i+1])
    N = np.where((mask > 0) * (mom0>2*r))
    f_r.append(np.nanmean(mom0[N]))
    ef_r.append(np.nanstd(mom0[N]))
'''
f_rm = []
ef_rm = []
for i in range(len(distance)-1):
    mask = (radius_model>=distance[i]) & (radius_model<distance[i+1])
    N = np.where((mask > 0) * (f_model>2*r))
    f_rm.append(np.nanmean(f_model[N]))
    ef_rm.append(np.nanstd(f_model[N]))
'''
from scipy.special import gammaincinv
def Sersic(r, para):
    I0, Re, n = para
    bn = gammaincinv(2*n, 0.5)
    return I0*np.exp(-bn*((r/Re)**(1/n)-1))

#f_out = Sersic(distance, para_out[2:5])
#f_in = Sersic(distance, para_out[7:10])

plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
ax.errorbar(distance[:-1], f_r, fmt='bo', mfc='none')
#ax.errorbar(distance[:-1], f_rm, fmt='rs', mfc='none')

#ax.plot(distance, f_in, 'k--')
#ax.plot(distance, f_out, 'k:')
#ax.plot(distance, f_in + f_out, 'k')

ax.set_xlim(-0.1, 3.1)
ax.set_ylim(1e-1, 1e1)
ax.set_xlabel("radius [kpc]")
ax.set_ylabel("Mean Flux [Jy/beam km/s]")
ax.semilogy()

#plt.savefig(output_dir+"fvsr_m.pdf", bbox_inches="tight", dpi=300)

# %%
# calculate the position angle of each pixel
yy, xx = np.mgrid[:800, :800]
pos_cen = [398.5, 394.1]
dxx, dyy = xx-pos_cen[0], yy-pos_cen[1]
comp = -1j*dxx + dyy
phase = np.zeros([800, 800])

import cmath
for i in range(len(dxx)):
    for j in range(len(dyy)):
        phase[i,j] = cmath.phase(comp[i,j])

phase[phase<0] += np.pi*2
plt.figure(figsize=(10, 8))
ax = plt.subplot(111)
im = ax.imshow(phase*u.rad.to('deg'), origin='lower')
plt.colorbar(im)
ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)

#%%
phi = np.linspace(0, 2*np.pi, len(distance))
f_phi = []
ef_phi = []
for i in range(len(phi)-1):
    mask = (phase>=phi[i]) & (phase<phi[i+1])
    N = np.where((mask > 0) & (mom0>=2*r) & (radius<=3.))
    f_phi.append(np.nanmean(mom0[N]))
    ef_phi.append(np.nanstd(mom0[N]))
plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
ax.errorbar(phi[:-1]*u.rad.to('deg'), f_phi, fmt='ko', mfc='none')
ax.set_xlabel("$\phi$ [deg]")
ax.set_ylabel("Mean Flux [Jy/beam km/s]")
ax.set_xlim(0, 360)
ax.set_ylim(0, 1.5)
#plt.savefig(output_dir+"f_phi.pdf", bbox_inches="tight", dpi=300)

# %%
f = np.zeros([len(distance), len(phi)])
for i in range(len(distance)-1):
    for j in range(len(phi)-1):
        mask = ((phase>=phi[j]) & (phase<=phi[j+1]) & (radius>=distance[i]) & (radius<=distance[i+1]))
        N = np.where((mask > 0) & (mom0>2*r))
        f[i, j] = np.nanmean(mom0[N])
        if np.nanmean(mom0[N]) == 'nan':
            print(i,j)

# %%
#mom0_contour = np.array([-1,1,2,4,8,16,32])*3*r
plt.figure(figsize=(8, 10))
ax = plt.subplot(111)
im = ax.imshow(f, cmap='jet', origin='lower', norm=LogNorm())
#ax.errorbar(phi[:-1]*u.rad.to('deg')/12, np.array(f_phi)*15, fmt='ko', mfc='none')

#im = ax.tricontour(phi, distance, f)
plt.xticks(np.arange(len(phi))[::5], np.round(phi[::5]*u.rad.to('deg')))
plt.yticks(np.arange(len(distance))[::5], distance[::5])

cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label('Mean Flux [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]')

#ax.contour(f, mom0_contour, colors=['k'])
ax.set_xlim(0, len(phi)-1.5)
ax.set_ylim(0, len(distance)-1.5)
ax.set_xlabel("$\phi$ [deg]")
ax.set_ylabel("Radius [kpc]")
#ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
#plt.savefig("/home/qyfei/Desktop/Codes/CODES/map_visualization/Analysis/PG0050/R_phi_flux.pdf", bbox_inches="tight", dpi=300)

# %%
mom0_level = np.array([-1,1,2,4,8,16,32,64,128])*2*r
fov = 100
size = fov
fig = plt.figure(figsize=(8,10))
ax = plt.subplot(projection=wcs[0,0])
im = ax.imshow(mom0, cmap='jet', origin='lower', vmin=r, vmax=128*r, norm=LogNorm())
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label('CO(2-1) [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]')
ax.contour(mom0, mom0_level, colors=['k'], linewidths=1.5)
    ## plot the figure
ax.contour(radius, np.linspace(0, 2.5, 5), colors=['m'], linewidths=2)
ax.hlines(pos_cen[1], 0, 1000, color='m', ls="--", lw=2)
ax.vlines(pos_cen[0], 0, 1000, color='m', ls="--", lw=2)

#ax.contour(phase, np.linspace(np.pi/4, np.pi*7/4, 7), colors=['w'], linewidths=1, alpha=0.7)

#rec = matplotlib.patches.Rectangle((pos_cen[0]+100, pos_cen[1]-200), 200, 30, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
#ax.add_artist(rec)
xbar = np.linspace(pos_cen[0]+110, pos_cen[0]+110+82.05) 
ybar = (pos_cen[1]-195)*xbar/xbar
#ax.plot(xbar, ybar, 'k', lw=2, zorder=2.5) #plot the representer bar
#ax.text(xbar[0]+20,ybar[0]+5,'5kpc',size=20, color='k', zorder=2.5)
    ## Add beam and annotated information
rec_size = 10
rec = matplotlib.patches.Rectangle((pos_cen[0]-fov, pos_cen[1]-fov), rec_size, rec_size, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax.add_artist(rec)
Beam = beam(hdu, pos_cen[0]-size+rec_size/2,pos_cen[1]-size+rec_size/2, 'k', pix_size)
ax.add_artist(Beam[0])
    ## Set the limit of x- and y-axis
ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)
ax.set_xlabel("R.A. (J2000)", labelpad=0.5, fontsize=20)
ax.set_ylabel("Dec (J2000)", labelpad=-1.0, fontsize=20)
#plt.savefig(output_dir+"bins.pdf", bbox_inches="tight", dpi=300)

# %%
size = 100
x_test = np.array([0, 1000])
def line(deg):
    return pos_cen[1] + np.tan(np.radians(deg))*(x_test - pos_cen[0])
y0 = line(90+15)            # Molina+ 21 phot
y1 = line(90+304)           # Molina+ 21 kinetic
y2 = line(90+141.879)       # Bulge
y3 = line(90+199.257)       # Disk
y4 = line(90+33.74)         # Bar
y5 = line(130+90)           # Kinetic

fig = plt.figure(figsize=(8,10))
ax = plt.subplot(projection=wcs[0,0])
im = ax.imshow(mom0, cmap='Greys', origin='lower', vmin=r, vmax=128*r, norm=LogNorm())

ax.plot(x_test, y0, 'C0', ls="--", lw=3, label="Molina+ 21 Phot") ##Juan's PA
ax.plot(x_test, y1, 'C1', ls="--", lw=3, label="Molina+ 21 Kine") ##Juan's PA
ax.plot(x_test, y2, 'C2', ls="--", lw=3, label="Bulge")
ax.plot(x_test, y3, 'C3', ls="--", lw=3, label="Disk")
ax.plot(x_test, y4, 'C4', ls="--", lw=3, label="Bar")
ax.plot(x_test, y5, 'C5', ls="--", lw=3, label="Kine")
ax.legend(loc="upper left", fontsize=20)


cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label('CO(2-1) [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]')
ax.contour(mom0, mom0_level, colors=['k'], linewidths=1.5)
    ## plot the figure
rec_size = 10
rec = matplotlib.patches.Rectangle((pos_cen[0]-fov, pos_cen[1]-fov), rec_size, rec_size, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax.add_artist(rec)
Beam = beam(hdu, pos_cen[0]-size+rec_size/2,pos_cen[1]-size+rec_size/2, 'k', pix_size)
ax.add_artist(Beam[0])
    ## Set the limit of x- and y-axis
ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)
ax.set_xlabel("R.A. (J2000)", labelpad=0.5, fontsize=20)
ax.set_ylabel("Dec (J2000)", labelpad=-1.0, fontsize=20)

#plt.savefig(output_dir+"PAs.pdf", bbox_inches="tight", dpi=300)
# %%
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
file = "PG0050_CO21-combine-line-10km-mosaic-mom0-pbcor.fits"
hdu2 = fits.open(path+file)[0]
mom0 = hdu2.data[0][0]
hdr2 = hdu2.header

dir = "/home/qyfei/Desktop/Codes/CODES/map_visualization/optical/PG0050/"
file = "PG0050+124_sci_F438W.fits"
#file = "PG0050+124_gal_nfn1cpcp.fits"
hdu1 = fits.open(dir+file)[0]
hdr1 = hdu1.header
hst_image = hdu1.data
hst_rms = sigma_clipped_stats(hst_image)[-1]

# %%
input_array = hdu1.data
input_wcs = WCS(hdr1)
output_wcs = WCS(hdr2)[0, 0]
test, _ = reproject_interp((input_array, input_wcs), output_wcs, shape_out=(800, 800))

# %%
size = 100
pos_cen = [395, 399]
plot_CO = mom0[pos_cen[0]-size:pos_cen[0]+size, pos_cen[1]-size:pos_cen[1]+size]
plot_hst = test[pos_cen[0]+43-size:pos_cen[0]+43+size, pos_cen[1]+16-size:pos_cen[1]+16+size]
vmin, vmax = np.percentile(hst_image, [43, 99.95])

fig = plt.figure(figsize=(8,10))
ax = plt.subplot(projection=output_wcs)
im = ax.imshow(plot_hst, cmap='Greys', origin='lower', vmin=1e-1, vmax=vmax, norm=LogNorm())
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label('HST image')
ax.contour(plot_CO, mom0_level, colors=['r'], linewidths=0.5)

#ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
#ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)

ax.set_xlabel("R.A. (J2000)", labelpad=0.5, fontsize=20)
ax.set_ylabel("Dec (J2000)", labelpad=-1.0, fontsize=20)
#plt.savefig(output_dir+"overplot.pdf", bbox_inches="tight", dpi=300)

# %%

hdr2

# %%
