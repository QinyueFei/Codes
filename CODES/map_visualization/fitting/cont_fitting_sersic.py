# %%
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/cont/"
file = "PG0050-cont-mosaic.fits"

from map_visualization.cont import load_cont
from map_visualization.maps import beam
from map_visualization.fitting.module import Disk2D, Gauss2D

cont, wcs, size, pix_size, hdu, pos_cen = load_cont(path, file)
xpos, ypos = pos_cen
sigma=1.3e-02
size = 100
f_cont = cont[xpos-size:xpos+size, ypos-size:ypos+size]
f_err = sigma
cont_level = np.array([-1,1,2,4,8,16])*2*f_err

# %%
## Fit with single sersic profile
def log_likelihood(para, x, y, z, zerr):
    x0, y0, Ie, Re, n, e, t = para
    model = Disk2D(hdu, x, y, x0, y0, Ie, Re, n, e, t)         # The outer disk
    sigma2 = zerr ** 2
    return -0.5 * np.sum((z - model) ** 2 / sigma2 + np.log(2*np.pi*sigma2))

def log_prior(para):
    x0, y0, Ie, Re, n, e, t = para[:7]
    if not (-1<x0<1 and -1<y0<1 and 1e-5<Ie<20. and 0.<Re<3. and 0.0<n<10.0 and 0<e<=1 and 0<=t<360):
        return -np.inf
    n_mu = 4.0
    n_sigma = 0.1
    lp = 0#np.log(1./(np.sqrt(2*np.pi)*n_sigma)) - 0.5*(n - n_mu)**2/n_sigma**2-np.log(n_mu)
    return lp

def log_probability(para, x, y, f, ferr):
    lp = log_prior(para)
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + log_likelihood(para, x, y, f, ferr)
# %%
fit_results = [-0.215, 0.218, 2.559, 0.065, 9.0, 0.5, 38]#fit_mini(f_cont)
x, y = np.mgrid[:2*size, :2*size]
output_dir = "/home/qyfei/Desktop/Results/map_visualization/fitting/Results/PG0050/sersic/"
print("Begin mcmc fitting:")
from multiprocessing import Pool
import emcee
pos = fit_results + 1e-4 * np.random.randn(200, 7)
backname = "cont_tutorial.h5"
nwalkers, ndim = pos.shape
backend = emcee.backends.HDFBackend(output_dir+backname)
backend.reset(nwalkers, ndim)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, f_cont, f_err), pool=pool, backend=backend)
    sampler.run_mcmc(pos, 1000, progress=True)

# %%
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
    flat_samples, labels=labels, truths=[para_out[0],para_out[1],para_out[2],para_out[3],para_out[4],para_out[5],para_out[6]]
    )
#np.savetxt(output_dir+'mom0_fit_para.txt', np.array([para_out, para_out_m, para_out_p]).T)
#return para_out, para_out_m, para_out_p

# %%
output_dir = "/home/qyfei/Desktop/Results/map_visualization/fitting/Results/PG0050/sersic/"
import h5py
import numpy as np
import matplotlib.pyplot as plt
import corner

#with h5py.File(output_dir+"tutorial.h5", "r") as f:
#    print(list(f.keys()))

f = h5py.File(output_dir+"cont_tutorial.h5", "r")
accepted = f['mcmc']['accepted']
chain = f['mcmc']['chain']
log_prob = f['mcmc']['log_prob']

get_chain = np.reshape(chain[600:], (400*200, 7))

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
f_total_res = f_cont - f_model
cmap = "Greys"
vmin, vmax = np.percentile(f_cont, [0.5, 99.99])

fig, axes = plt.subplots(figsize=(18, 7), nrows=1, ncols=3)
plt.subplots_adjust(wspace=0)
ax0, ax1, ax2 = axes
im0 = ax0.imshow(f_cont, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
ax0.contour(f_cont, cont_level, colors=['k'], linewidths=1)
ax0.text(10, 10, "DATA", color="k")
im1 = ax1.imshow(f_model, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
ax1.contour(f_model, cont_level, colors=['k'], linewidths=1)
ax1.text(10, 10, "MODEL", color="k")
im2 = ax2.imshow(f_total_res, vmin=-0.06, vmax=0.06, cmap=cmap, origin='lower')
ax2.contour(f_total_res, cont_level, colors=['k'], linewidths=1)

for ax in axes[:]:
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.125, 0.115, 0.517, 0.05])
cb_ax = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
cb_ax.set_label(r"CONTINUUM [$\mathrm{Jy\,beam^{-1}}$]")

cbar_res = fig.add_axes([0.643, 0.115, 0.257, 0.05])
cb_res = fig.colorbar(im2, cax=cbar_res, orientation='horizontal')
cb_res.set_label(r"$f_\mathrm{res}$ [$\mathrm{Jy\,beam^{-1}}$]")

rec = matplotlib.patches.Rectangle((0, 0), 10, 10,
angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax0.add_artist(rec)
Beam = beam(hdu, 5., 5., 'w', pix_size)
ax0.add_artist(Beam[0])
#plt.savefig(output_dir+"cont_fit.pdf", bbox_inches="tight", dpi=300)

# %%

# %%
