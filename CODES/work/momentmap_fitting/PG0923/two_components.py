# %%
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
#from Dynamics.models import Rs
import astropy.units as u
import matplotlib
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats

plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)


path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG0923/data/"
# file = "PG0923+129_CO21_final_image_mom0.fits"
file = "PG0923.dilmsk.mom0.fits.gz"
efile = "PG0923.dilsmomsk.emom0.fits.gz"
# freefile = "PG0923+129_CO21_final_image_mom0_free.fits"

from map_visualization.maps import beam
from map_visualization.fitting.module import Disk2D, Gauss2D, kernel, truncated_Disk2D_ring, truncated_Disk2D_ring_out, Sersic_ring
# freemom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, freefile)
# mean, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, meanfile)

hdu = fits.open(path+file)[0]
mom0 = hdu.data#[0][0]
wcs = WCS(hdu.header)
pos_cen = [92, 92]#np.where(mom0 == np.nanmax(mom0))
pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
size = 80

emom0 = fits.open(path+efile)[0].data

mask = np.where((1 - np.isnan(mom0))*(1 - np.isnan(emom0)))
# free_mom0 = fits.open(path+freefile)[0].data[0][0]
# r = np.nanstd(free_mom0)

# %%
fit_results = [-0.08, 0.048, 1.16, 2.51, 0.46, 0.31, 61]#fit_mini(f_mom0)
x, y = np.mgrid[:185, :185]

size = 75                  ## The size of map
xpos, ypos = 88, 90#pos_cen[0][0], pos_cen[1][0]       ## The position of center
f_mom0 = mom0#[xpos-size:xpos+size, ypos-size:ypos+size] ## The moment map that we want to fit
f_err = emom0                   ## The rms noise of the moment0 map
# mom0_level = np.array([-1,1,2,4,8,16,32])*2*f_err

# %%

def log_likelihood(para, x, y, z, zerr):
    x0, y0, Ie, Re, n, e, t = para[:7]
    rbin, rsin, qin, PAin, rbout, rsout, qout, PAout = para[7:15]
    Ie1, Re1, n1, e1, t1 = para[15:]
    
    model_origin = Sersic_ring(hdu, x, y, x0, y0, Ie, Re, n, e, t, rbin, rsin, qin, PAin, rbout, rsout, qout, PAout) + Disk2D(hdu, x, y, x0, y0, Ie1, Re1, n1, e1, t1)

    model = model_origin[mask]

    sigma2 = zerr ** 2
    return -0.5 * np.sum((z - model) ** 2 / sigma2 + np.log(2*np.pi*sigma2))

# %%
## Fit the observation with double Sersic profile and one Gaussian profile

def log_prior(para):
    x0, y0, Ie, Re, n, e, t = para[:7]
    rbin, rsin, qin, PAin, rbout, rsout, qout, PAout = para[7:15]
    Ie1, Re1, n1, e1, t1 = para[15:]

    if not (70<x0<100 and 70<y0<100 and 0.0<rbin and 0.0<rsin and 0<qin<=1 and -360<PAin<=360 and rbin<rbout and 0.0<rsout and 0<qout<=1 and -360<PAout<=360 and 1e-5<Ie<1e5 and 0.<Re<=20. and 0.0<n<10. and 0.0<e<=1 and -360<t<=360 and 1e-5<Ie1<1e5 and 0.<Re1<=20 and 0.01<n1<10. and 0.01<e1<=1 and -360<t1<=360):

        return -np.inf
    lp = 0
    return lp

def log_probability(para, x, y, f, ferr):
    lp = log_prior(para)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(para, x, y, f, ferr)

# %%
## Load the observation
size = 75                  ## The size of map
x, y = np.mgrid[:185, :185]
xpos, ypos = 88, 90#pos_cen[0][0], pos_cen[1][0]       ## The position of center
f_mom0 = mom0[mask]#[xpos-size:xpos+size, ypos-size:ypos+size] ## The moment map that we want to fit
f_err = emom0[mask]                   ## The rms noise of the moment0 map
# mom0_level = np.array([-1,1,2,4,8,16,32])*2*f_err

# %%
from scipy.optimize import minimize
para_ini = [0.25, -0.41,  1.68,  2.30,  1.10, 0.31, 65.62,
            10.89,  2.00,  0.41, 61.12, 
            1.0, 0.3, 0.5, 0.3, 60]

para_ini = np.array([0.30, -0.43, 4.66, 1.26, 1.56, 0.31, 64.80,
                    1.23, 0.13, 0.70, 55.48, 2.00, 0.20, 0.70, 60.00,
                    1.0, 2.5, 1.0, 0.4, 60])

ini_guess = np.array([89.58, 88.10, 10.9, 1.1, 1.0, 0.1, 218,
                    1.3, 0.1, 0.7, 55, 2.0, 0.2, 0.5, 60,
                    0.1, 4.4, 1.0, 0.2, 60])


np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = ini_guess + 0.1 * np.random.randn(20)
soln = minimize(nll, initial, args=(x, y, f_mom0, f_err), method="Nelder-Mead")
para_mini = soln.x

print("Maximum likelihood estimates:")
print("dx0 = {0:.3f}".format(para_mini[0]))
print("dy0 = {0:.3f}".format(para_mini[1]))
print("Ie = {0:.3f}".format(para_mini[2]))
print("Re = {0:.3f}".format(para_mini[3]))
print("n = {0:.3f}".format(para_mini[4]))
print("e = {0:.3f}".format(para_mini[5]))
print("t = {0:.3f}".format(para_mini[6]))
print("rbin = {0:.3f}".format(para_mini[7]))
print("rsin = {0:.3f}".format(para_mini[8]))
print("qin = {0:.3f}".format(para_mini[9]))
print("PAin = {0:.3f}".format(para_mini[10]))
print("rbout = {0:.3f}".format(para_mini[11]))
print("rsout = {0:.3f}".format(para_mini[12]))
print("qout = {0:.3f}".format(para_mini[13]))
print("PAout = {0:.3f}".format(para_mini[14]))

print("Ie1 = {0:.3f}".format(para_mini[15]))
print("Re1 = {0:.3f}".format(para_mini[16]))
print("n1 = {0:.3f}".format(para_mini[17]))
print("e1 = {0:.3f}".format(para_mini[18]))
print("t1 = {0:.3f}".format(para_mini[19]))

# %%
output_dir = "/home/qyfei/Desktop/Results/map_visualization/fitting/Results/PG0923/EDGE/two_components/"
print("Begin mcmc fitting:")
from multiprocessing import Pool
import emcee

fit_results = np.array([0.28, -0.40, 8.55, 1.11, 0.88, 0.12, 217.20,
                        1.71, 0.01, 0.74, 56.37, 3.85, 0.02, 0.38, 73.94,
                        0.20, 4.51, 1.68, 0.26, 60.79])

ini_guess = np.array([89.58, 88.10, 9.78, 1.67, 0.15, 0.37, 57.37, 
                    3.31, 2.18, 0.55, 61.37, 3.81, 0.01, 0.37, 74.00, 
                    0.02, 15.22, 3.59, 0.13, 41.53])

pos = ini_guess + 1e-4 * np.random.randn(400, 20)
backname = "tutorial.h5"
nwalkers, ndim = pos.shape
backend = emcee.backends.HDFBackend(output_dir+backname)
backend.reset(nwalkers, ndim)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, f_mom0, f_err), pool=pool, backend=backend)
    sampler.run_mcmc(pos, 2000, progress=True)

# %%
fig, axes = plt.subplots(20, figsize=(10, 20), sharex=True)
samples = sampler.get_chain()
labels = ["$x_0$", "$y_0$", "Ie", "Re", "n", "e", "t", "rin", "rsin", "qin", "PAin", "rout", "rsout", "qout", "PAout", "Ie1", "Re1", "n1", "e1", "t1"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import corner

output_dir = "/home/qyfei/Desktop/Results/map_visualization/fitting/Results/PG0923/EDGE/two_components/"

with h5py.File(output_dir+"tutorial.h5", "r") as f:
    print(list(f.keys()))
f = h5py.File(output_dir+"tutorial.h5", "r")
accepted = f['mcmc']['accepted']
chain = f['mcmc']['chain']
log_prob = f['mcmc']['log_prob']
get_chain = np.reshape(chain[1800:], (400*200, 20))

# %%
labels = ["$x_0$", "$y_0$", "Ie", "Re", "n", "e", "t", "rin", "rsin", "qin", "PAin", "rout", "rsout", "qout", "PAout", "Ie1", "Re1", "n1", "e1", "t1"]
from IPython.display import display, Math
para_out = []
for i in range(len(get_chain[1])):
    para_out.append(np.percentile(get_chain[:,i], [50])[0])
    mcmc = (np.percentile(get_chain[:,i], [16, 50, 84]))
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))

# %%
#flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
#print(flat_samples.shape)
fig = corner.corner(
        get_chain, labels=labels, truths=[para_out[0],para_out[1],para_out[2],para_out[3],para_out[4],para_out[5],para_out[6],para_out[7],para_out[8],para_out[9],para_out[10],para_out[11], para_out[12], para_out[13], para_out[14], para_out[15], para_out[16], para_out[17], para_out[18], para_out[19]], show_titles=True
    )#
# plt.savefig(output_dir+"corner.png", bbox_inches="tight", dpi=300)

# %%
from matplotlib import colors
# para_out = [0, 0, -0.5, 1.0, 0.8, 60,
            # 0.4/2, 5, 0.5, 0.2, 60]#,
# para_out = para_mini
# para_out = np.array([-0.60, -0.90, 10.89, 1.02, 1.02, 0.12, 218.22, 
#                     1.71, 0.01, 0.74, 56.34, 3.86, 0.01, 0.38, 73.90, 
#                     0.20, 4.56, 1.70, 0.26, 23.18])
cmap = "jet"
# x, y = np.mgrid[:185,:185]#[:2*size, :2*size]

# para_out = soln.x

f_ring = Sersic_ring(hdu, x, y, para_out[0],para_out[1],para_out[2],para_out[3],para_out[4],para_out[5],para_out[6],para_out[7],para_out[8],para_out[9],para_out[10], para_out[11], para_out[12], para_out[13], para_out[14])
f_disk = Disk2D(hdu, x, y, para_out[0], para_out[1], para_out[15], para_out[16], para_out[17], para_out[18], para_out[19])

f_model = f_ring + f_disk

f_total_res = mom0 - f_model

mom0_level = np.array([-1,1,2,4,8,16,32,64,128])*2*0.055#r_test
# vmin, vmax = 0.25, np.percentile(f_mom0, [99.999])
vmin, vmax = 0.25, 2.3

fig, axes = plt.subplots(figsize=(18, 7), nrows=1, ncols=3)
plt.subplots_adjust(wspace=0)
ax0, ax1, ax2 = axes
im0 = ax0.imshow(mom0, vmin=-vmin, vmax=vmax, cmap=cmap, origin='lower', norm=colors.PowerNorm(gamma=0.525))
ax0.contour(mom0, mom0_level, colors=["k"], linewidths=1.)
ax0.text(10, 10, "DATA", color="w")
im1 = ax1.imshow(f_model, vmin=-vmin, vmax=vmax, cmap=cmap, origin='lower', norm=colors.PowerNorm(gamma=0.525))
ax1.contour(f_model, mom0_level, colors=["k"], linewidths=1.)
ax1.text(10, 10, "MODEL", color="w")
im2 = ax2.imshow(f_total_res, vmin=-0.45, vmax=0.45, cmap=cmap, origin='lower')
ax2.contour(f_total_res, mom0_level, colors=["k"], linewidths=1.)

for ax in axes[:]:
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.125, 0.115, 0.517, 0.05])
cb_ax = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
#cb_ax.set_label(r"$I_\mathrm{CO(2-1)}$ [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]")
cb_ax.set_label(r"$I_\mathrm{CO(2-1)}$ [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]")

cbar_res = fig.add_axes([0.643, 0.115, 0.257, 0.05])
cb_res = fig.colorbar(im2, cax=cbar_res, orientation='horizontal')
#cb_res.set_label(r"$I_\mathrm{res}$ [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]")
cb_res.set_label(r"$I_\mathrm{res}$ [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]")

rec = matplotlib.patches.Rectangle((0, 0), 10, 10,
angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax0.add_artist(rec)
Beam = beam(hdu, 5., 5., 'w', pix_size)
ax0.add_artist(Beam[0])

# plt.savefig(output_dir+"PG0923_CO21_EDGE_fit.pdf", bbox_inches="tight", dpi=300)
# %%
fig, axes = plt.subplots(figsize=(18, 7), nrows=1, ncols=3)
plt.subplots_adjust(wspace=0)
ax0, ax1, ax2 = axes
im0 = ax0.imshow(f_ring, vmin=-vmin, vmax=vmax, cmap=cmap, origin='lower', norm=colors.PowerNorm(0.5))
ax0.contour(f_ring, mom0_level, colors=['k'], linewidths=1)
ax0.text(10, 10, "RING", color="k")
im1 = ax1.imshow(f_disk, vmin=-vmin, vmax=vmax, cmap=cmap, origin='lower', norm=colors.PowerNorm(0.5))
ax1.contour(f_disk, mom0_level, colors=['k'], linewidths=1)
ax1.text(10, 10, "DISK", color="k")
im2 = ax2.imshow(f_model, vmin=-vmin, vmax=vmax, cmap=cmap, origin='lower', norm=colors.PowerNorm(0.5))
ax2.contour(f_model, mom0_level, colors=['k'], linewidths=1)
ax2.text(10, 10, "TOTAL", color="k")
# im3 = ax3.imshow(f_model, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower', norm=LogNorm())
# ax3.contour(f_model, mom0_level, colors=['k'], linewidths=1)
# ax3.text(10, 10, "TOTAL", color="k")
for ax in axes[:]:
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.125, 0.1, 0.775, 0.05])
cb_ax = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
cb_ax.set_label(r"FLUX [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]")
# plt.savefig(output_dir+"PG0923_EDGE_components.pdf", bbox_inches="tight", dpi=300)

# %%
