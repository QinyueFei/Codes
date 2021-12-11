# %%
import numpy as np
import matplotlib.pyplot as plt
from Dynamics.models import Rs
import astropy.units as u
import matplotlib
from scipy import integrate

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
mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file)
size = 100                  ## The size of map
xpos, ypos = pos_cen[0][0], pos_cen[1][0]       ## The position of center
f_mom0 = mom0[xpos-size:xpos+size, ypos-size:ypos+size] ## The moment map that we want to fit
f_err = r                   ## The rms noise of the moment0 map
mom0_level = np.array([-1,1,2,4,8,16,32])*3*f_err

# %%
## Fit with single sersic profile
def log_likelihood(para, x, y, z, zerr):
    x0, y0, Ie, Re, n, e, t = para
    model = Disk2D(hdu, x, y, x0, y0, Ie, Re, n, e, t)         # The outer disk
    sigma2 = zerr ** 2
    return -0.5 * np.sum((z - model) ** 2 / sigma2 + np.log(2*np.pi*sigma2))

def log_prior(para):
    x0, y0, Ie, Re, n, e, t = para[:7]
    if not (-1<x0<1 and -1<y0<1 and 1e-5<Ie<20. and 0.<Re<3. and 0.0<n<10.0 and 0<t<2*np.pi and 0<e<=1):
        return -np.inf
    t_mu = 0.722
    t_sigma = 0.007
    lp = np.log(1./(np.sqrt(2*np.pi)*t_sigma)) - 0.5*(t - t_mu)**2/t_sigma**2-np.log(t_mu)
    return lp

def log_probability(para, x, y, f, ferr):
    lp = log_prior(para)
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + log_likelihood(para, x, y, f, ferr)

def fit_mcmc(f_mom0):
    fit_results = [-0.017, 0.014, 0.168, 1.3, 6.0, 0.242, 0.702]#fit_mini(f_mom0)
    x, y = np.mgrid[:2*size, :2*size]
    output_dir = "/home/qyfei/Desktop/Codes/CODES/map_visualization/fitting/Results/PG0050/sersic/"
    print("Begin mcmc fitting:")
    from multiprocessing import Pool
    import emcee
    pos = fit_results + 1e-4 * np.random.randn(200, 7)
    backname = "tutorial.h5"
    nwalkers, ndim = pos.shape
    backend = emcee.backends.HDFBackend(output_dir+backname)
    backend.reset(nwalkers, ndim)

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, f_mom0, f_err), pool=pool, backend=backend)
        sampler.run_mcmc(pos, 1000, progress=True)

    fig, axes = plt.subplots(7, figsize=(10, 12), sharex=True)
    samples = sampler.get_chain()
    labels = ["$x_0$", "$y_0$", "Ie", "Re", "n", "e", "t"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    #plt.savefig("Result/step.png", bbox_inches='tight', dpi=300)


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
        flat_samples, labels=labels, truths=[para_out[0],para_out[1],para_out[2],para_out[3],para_out[4],para_out[5],para_out[6]]
    )
    np.savetxt(output_dir+'mom0_fit_para.txt', np.array([para_out, para_out_m, para_out_p]).T)
    return para_out, para_out_m, para_out_p

# %%
output_dir = "/home/qyfei/Desktop/Codes/CODES/map_visualization/fitting/Results/PG0050/sersic/"
para_out, para_out_m, para_out_p = fit_mcmc(f_mom0)

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import corner

with h5py.File(output_dir+"tutorial.h5", "r") as f:
    print(list(f.keys()))

f = h5py.File(output_dir+"tutorial.h5", "r")
accepted = f['mcmc']['accepted']
chain = f['mcmc']['chain']
log_prob = f['mcmc']['log_prob']

get_chain = np.reshape(chain[700:], (300*200, 7))

# %%
labels = ["$x_0$", "$y_0$", "Ie", "Re", "n", "e", "t"]
from IPython.display import display, Math
para_out = []
for i in range(len(get_chain[1])):
    para_out.append(np.percentile(get_chain[:,i], [50]))
    mcmc = (np.percentile(get_chain[:,i], [16, 50, 84]))
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))

# %%
#flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
#print(flat_samples.shape)
fig = corner.corner(
        get_chain, labels=labels, truths=[para_out[0],para_out[1],para_out[2],para_out[3],para_out[4],para_out[5],para_out[6]]
    )
#plt.savefig(output_dir+"corner.pdf", bbox_inches="tight")
# %%
x, y = np.mgrid[:2*size, :2*size]
f_model = Disk2D(hdu, x, y, para_out[0], para_out[1], para_out[2], para_out[3], para_out[4], para_out[5], para_out[6])
f_total_res = f_mom0 - f_model

fig, axes = plt.subplots(figsize=(18, 7), nrows=1, ncols=3)
plt.subplots_adjust(wspace=0)
ax0, ax1, ax2 = axes
im0 = ax0.imshow(f_mom0, vmin=-0.086, vmax=8.5, cmap='jet', origin='lower')
ax0.contour(f_mom0, mom0_level, colors=['k'], linewidths=0.5)
ax0.text(10, 10, "DATA", color="w")
im1 = ax1.imshow(f_model, vmin=0-0.086, vmax=8.5, cmap='jet', origin='lower')
ax1.contour(f_model, mom0_level, colors=['k'], linewidths=0.5)
ax1.text(10, 10, "MODEL", color="w")
im2 = ax2.imshow(f_total_res, vmin=-0.55, vmax=0.55, cmap='jet', origin='lower')
ax2.contour(f_total_res, mom0_level, colors=['k'], linewidths=0.5)

for ax in axes[:]:
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.13, 0.1, 0.51, 0.05])
cb_ax = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
cb_ax.set_label(r"FLUX [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]")

cbar_res = fig.add_axes([0.645, 0.1, 0.255, 0.05])
cb_res = fig.colorbar(im2, cax=cbar_res, orientation='horizontal')
cb_res.set_label(r"$f_\mathrm{res}$ [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]")

rec = matplotlib.patches.Rectangle((0, 0), 10, 10,
angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax0.add_artist(rec)
Beam = beam(hdu, 5., 5., 'w', pix_size)
ax0.add_artist(Beam[0])
#plt.savefig(output_dir+"CO21_mom0_fit.pdf", bbox_inches="tight", dpi=300)

# %%

fig, axes = plt.subplots(figsize=(12, 7), nrows=1, ncols=3)
plt.subplots_adjust(wspace=0)
ax0, ax1, ax2 = axes
im0 = ax0.imshow(f_model, vmin=-0.086, vmax=8.5, cmap='jet', origin='lower')
ax0.contour(f_model, mom0_level, colors=['k'], linewidths=0.5)
ax0.text(10, 10, "INNER", color="w")
im2 = ax2.imshow(f_model, vmin=0-0.086, vmax=8.5, cmap='jet', origin='lower')
ax2.contour(f_model, mom0_level, colors=['k'], linewidths=0.5)
ax2.text(10, 10, "MODEL", color="w")
for ax in axes[:]:
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.125, 0.1, 0.775, 0.05])
cb_ax = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
cb_ax.set_label(r"FLUX [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]")
#plt.savefig("/home/qyfei/Desktop/Codes/CODES/map_visualization/fitting/Results/PG0050/double_sersic/model.pdf", bbox_inches="tight", dpi=300)

# %%
