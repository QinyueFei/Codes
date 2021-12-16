# %%
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats.sigma_clipping import sigma_clipped_stats
import astropy.units as u
import matplotlib
from scipy import integrate
from matplotlib import colorbar

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

## Build the model

def log_likelihood(para, x, y, z, zerr):
    x0, y0, Ie, Re, n, e, t = para[:7]
    Ie1, Re1, n1, e1, t1 = para[7:12]
    Ie2, Re2, n2, e2, t2 = para[12:]
    model0 = Disk2D(hdu, x, y, x0, y0, Ie, Re, n, e, t)         # The bulge component
    model1 = Disk2D(hdu, x, y, x0, y0, Ie1, Re1, n1, e1, t1)    # The disk component
    model2 = Disk2D(hdu, x, y, x0, y0, Ie2, Re2, n2, e2, t2)    # The bar component
    model = model0 + model1 + model2
    sigma2 = zerr ** 2
    return -0.5 * np.sum((z - model) ** 2 / sigma2 + np.log(2*np.pi*sigma2))

# %%
mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file)
size = 100                  ## The size of map
xpos, ypos = pos_cen[0][0], pos_cen[1][0]       ## The position of center
f_mom0 = mom0[xpos-size:xpos+size, ypos-size:ypos+size] ## The moment map that we want to fit
f_err = r                   ## The rms noise of the moment0 map
mom0_level = np.array([-1,1,2,4,8,16,32])*3*f_err

# %%
## convert into K km/s
from astropy.cosmology import Planck15
z = 0.061
DL = Planck15.luminosity_distance(z)
nu_obs = 230.58/(1+z)#230.58 # Convert to CO(1-0)

beam_area = 0.18899035
mom0_test = 3.25e7*mom0*DL.value**2/(1+z)**3/nu_obs**2/beam_area/1e6
r_test = sigma_clipped_stats(mom0_test)[-1]
f_mom0_test = mom0_test[xpos-size:xpos+size, ypos-size:ypos+size]

plt.imshow(f_mom0_test, norm=LogNorm())
plt.colorbar()

# %%
## Fit the observation with double Sersic profile

def log_prior(para):
    x0, y0, Ie, Re, n, e, t = para[:7]
    Ie1, Re1, n1, e1, t1 = para[7:12]
    Ie2, Re2, n2, e2, t2 = para[12:]

    if not (-10<x0<10 and -10<y0<10 and 1e-5<Ie<20 and 0.1<Re<10. and 0.1<n<10.0 and 0.01<e<=1 and 0<t<=360 and 1e-5<Ie1<50 and 0.<Re1<=10. and 0.01<n1<10. and 0.01<e1<=1 and 0<t1<=360 and 1e-5<Ie2<20 and 0.1<Re2<=10. and 0.1<n2<10. and 0.01<e2<=1. and 0<t2<=360):
        return -np.inf
    #lp = 0
    n1mu = 4.00
    n1sigma = 0.10
    lpn1 = np.log(1./(np.sqrt(2*np.pi)*n1sigma)) - 0.5*(n1 - n1mu)**2/n1sigma**2-np.log(n1mu)
    lp = lpn1
    return lp

def log_probability(para, x, y, f, ferr):
    lp = log_prior(para)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(para, x, y, f, ferr)

#def fit_mcmc(f_mom0_, f_err_):
#f_mom0, f_err = f_mom0_, f_err_
    # Ie    Re    n   e   t  
comp_0 = [-0.015, 0.015, 0.436, 1.296, 0.478, 0.201, 141.729]
comp_1 = [4.000, 0.108, 4.000, 0.392, 190.225]
comp_2 = [2.550, 0.479, 0.353, 0.689, 33.512]
fit_results = comp_0 + comp_1 + comp_2
x, y = np.mgrid[:2*size, :2*size]
output_dir = "/home/qyfei/Desktop/Results/map_visualization/fitting/Results/PG0050/triple_sersic/"
print("Begin mcmc fitting:")
from multiprocessing import Pool
import emcee
pos = fit_results + 1e-4 * np.random.randn(400, 17)
backname = "tutorial.h5"
nwalkers, ndim = pos.shape
backend = emcee.backends.HDFBackend(output_dir+backname)
backend.reset(nwalkers, ndim)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, f_mom0, f_err), pool=pool, backend=backend)
    sampler.run_mcmc(pos, 1500, progress=True)

# %%
fig, axes = plt.subplots(17, figsize=(10, 17), sharex=True)
samples = sampler.get_chain()
labels = ["$x_0$", "$y_0$", "Ie", "Re", "n", "e", "t", "Ie1", "Re1", "n1", "e1", "t1", "Ie2", "Re2", "n2", "e2", "t2"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
#plt.savefig(output_dir+"step.pdf", bbox_inches="tight", dpi=300)

# %%
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)

para_out = np.zeros(ndim)
para_out_m = np.zeros(ndim)
para_out_p = np.zeros(ndim)
for i in range(ndim):
    para_out[i] = np.percentile(flat_samples[:, i], [50])
    para_out_m[i] = np.percentile(flat_samples[:, i], [16])
    para_out_p[i] = np.percentile(flat_samples[:, i], [84])
        
import corner
fig = corner.corner(
    flat_samples, labels=labels, truths=[para_out[0],para_out[1],para_out[2],para_out[3],para_out[4],para_out[5],para_out[6],para_out[7],para_out[8],para_out[9],para_out[10],para_out[11], para_out[12], para_out[13], para_out[14], para_out[15], para_out[16]]
    )
#np.savetxt(output_dir+'mom0_fit_para.txt', np.array([para_out, para_out_m, para_out_p]).T)
    #return para_out, para_out_m, para_out_p


# %%
output_dir = "/home/qyfei/Desktop/Results/map_visualization/fitting/Results/PG0050/triple_sersic/"

# %%
#para_out, para_out_m, para_out_p = fit_mcmc(f_mom0, f_err)

# %%

import h5py
import numpy as np
import matplotlib.pyplot as plt
import corner
output_dir = "/home/qyfei/Desktop/Results/map_visualization/fitting/Results/PG0050/triple_sersic/"

with h5py.File(output_dir+"tutorial.h5", "r") as f:
    print(list(f.keys()))
f = h5py.File(output_dir+"tutorial.h5", "r")
accepted = f['mcmc']['accepted']
chain = f['mcmc']['chain']
log_prob = f['mcmc']['log_prob']
get_chain = np.reshape(chain[1000:], (400*500, 17))

# %%
labels = ["$x_0$", "$y_0$", "Ie", "Re", "n", "e", "t", "Ie1", "Re1", "n1", "e1", "t1", "Ie2", "Re2", "n2", "e2", "t2"]
from IPython.display import display, Math
para_out = []
for i in range(len(get_chain[1])):
    para_out.append(np.percentile(get_chain[:,i], [50])[0])
    mcmc = (np.percentile(get_chain[:,i], [16, 50, 84]))
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))

# %%
#flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
#print(flat_samples.shape)
fig = corner.corner(
        get_chain, labels=labels, truths=[para_out[0],para_out[1],para_out[2],para_out[3],para_out[4],para_out[5],para_out[6],para_out[7],para_out[8],para_out[9],para_out[10],para_out[11], para_out[12], para_out[13], para_out[14], para_out[15], para_out[16]]
    )
#plt.savefig(output_dir+"corner.pdf", bbox_inches="tight")

# %%

cmap = "Greys"
x, y = np.mgrid[:2*size, :2*size]
f_bulge = Disk2D(hdu, x, y, para_out[0], para_out[1], para_out[2], para_out[3], para_out[4], para_out[5], para_out[6])
f_disk = Disk2D(hdu, x, y, para_out[0], para_out[1], para_out[7], para_out[8], para_out[9], para_out[10], para_out[11])
f_bar = Disk2D(hdu, x, y, para_out[0], para_out[1], para_out[12], para_out[13], para_out[14], para_out[15], para_out[16])

f_model = f_bulge + f_disk + f_bar
f_mom0 = f_mom0
f_total_res = f_mom0 - f_model
mom0_level = np.array([-1,1,2,4,8,16,32,64,128])*2*f_err
vmin, vmax = f_err, np.percentile(f_mom0, [99.9])

fig, axes = plt.subplots(figsize=(18, 7), nrows=1, ncols=3)
plt.subplots_adjust(wspace=0)
ax0, ax1, ax2 = axes
im0 = ax0.imshow(f_mom0, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower', norm=LogNorm())
ax0.contour(f_mom0, mom0_level, colors=["k"], linewidths=1.)
ax0.text(10, 10, "DATA", color="k")
im1 = ax1.imshow(f_model, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower', norm=LogNorm())
ax1.contour(f_model, mom0_level, colors=["k"], linewidths=1.)
ax1.text(10, 10, "MODEL", color="k")
im2 = ax2.imshow(f_total_res, vmin=-0.35, vmax=0.35, cmap=cmap, origin='lower')
ax2.contour(f_total_res, mom0_level, colors=["k"], linewidths=1.)

for ax in axes[:]:
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.125, 0.115, 0.517, 0.05])
cb_ax = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
cb_ax.set_label(r"$I_\mathrm{CO(2-1)}$ [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]")

cbar_res = fig.add_axes([0.643, 0.115, 0.257, 0.05])
cb_res = fig.colorbar(im2, cax=cbar_res, orientation='horizontal')
cb_res.set_label(r"$I_\mathrm{res}$ [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]")

rec = matplotlib.patches.Rectangle((0, 0), 10, 10,
angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax0.add_artist(rec)
Beam = beam(hdu, 5., 5., 'w', pix_size)
ax0.add_artist(Beam[0])

#plt.savefig(output_dir+"CO21_mom0_fit_rep.pdf", bbox_inches="tight", dpi=300)

# %%
## Show the region in which residual is large
plt.figure(figsize=(8, 10))
ax = plt.subplot(111)
im = ax.imshow(f_mom0, vmin=vmin, vmax=vmax, cmap='Greys', origin='lower', norm=LogNorm())
ax.contour(f_mom0, mom0_level, colors=["b"], linewidths=1.)
ax.contour(f_total_res, mom0_level, colors=["r"], linewidths=1.)

cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label("FLUX")
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
#plt.savefig(output_dir+"model_present.pdf", bbox_inches="tight", dpi=300)

# %%
cmap = "Greys"
fig, axes = plt.subplots(figsize=(24, 7), nrows=1, ncols=4)
plt.subplots_adjust(wspace=0)
ax0, ax1, ax2, ax3 = axes
im0 = ax0.imshow(f_bulge, vmin=0.086, vmax=8.5, cmap=cmap, origin='lower', norm=LogNorm())
ax0.contour(f_bulge, mom0_level, colors=['k'], linewidths=1)
ax0.text(10, 10, "DISK", color="k")
im1 = ax1.imshow(f_disk, vmin=0.086, vmax=8.5, cmap=cmap, origin='lower', norm=LogNorm())
ax1.contour(f_disk, mom0_level, colors=['k'], linewidths=1)
ax1.text(10, 10, "CORE", color="k")
im2 = ax2.imshow(f_bar, vmin=0.086, vmax=8.5, cmap=cmap, origin='lower', norm=LogNorm())
ax2.contour(f_bar, mom0_level, colors=['k'], linewidths=1)
ax2.text(10, 10, "BAR", color="k")
im3 = ax3.imshow(f_model, vmin=0.086, vmax=8.5, cmap=cmap, origin='lower', norm=LogNorm())
ax3.contour(f_model, mom0_level, colors=['k'], linewidths=1)
ax3.text(10, 10, "TOTAL", color="k")
for ax in axes[:]:
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.125, 0.1, 0.775, 0.05])
cb_ax = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
cb_ax.set_label(r"FLUX [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]")
#plt.savefig(output_dir+"components.pdf", bbox_inches="tight", dpi=300)

# %%
