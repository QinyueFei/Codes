# %%
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
from map_visualization.fitting.module import Gauss2D

cont, wcs, size, pix_size, hdu, pos_cen = load_cont(path, file)
xpos, ypos = pos_cen
size = 100
sigma=1.3e-02
f_cont = cont[xpos-size:xpos+size, ypos-size:ypos+size]
f_err = sigma
cont_level = np.array([-1,1,2,4,8,16])*2*f_err

# %%
def log_likelihood(para, x, y, z, zerr):
    dx0, dy0, I0, xstd0, ystd0, phi = para
    model = Gauss2D(hdu, x, y, dx0, dy0, I0, xstd0, ystd0, phi)
    sigma2 = zerr ** 2# + model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((z - model) ** 2 / sigma2)

def fit_mini(f_cont):
    x, y = np.mgrid[:2*size, :2*size]
    dx0, dy0, I0, xstd, ystd, phi = 0, 0, 0.8, 10, 10, 1
    print("Begin Maximum likelyhood fitting:")
    np.random.seed(42)
    from scipy.optimize import minimize
    nll = lambda *args: -log_likelihood(*args)
    initial = np.array([dx0, dy0, I0, xstd, ystd, phi]) + 0.1 * np.random.randn(6)
    soln = minimize(nll, initial, args=(x, y, f_cont, f_err))
    x0_g_ml, y0_g_ml, I0_ml, xstd_ml, ystd_ml, phi_ml = soln.x[:]
    print("Maximum likelihood estimates of Core:")
    print("x0 = {0:.3f}".format(x0_g_ml))
    print("y0 = {0:.3f}".format(y0_g_ml))
    print("Ie = {0:.3f}".format(I0_ml))
    print("xstd = {0:.3f}".format(xstd_ml))
    print("ystd = {0:.3f}".format(ystd_ml))
    print("phi = {0:.3f}".format(phi_ml))
    return soln.x[:]

#fit_results = fit_mini(f_cont)

def plot_fit_mini(fit_results):
    ## fit_result = fit_mini(path, file)
    x, y = np.mgrid[:2*size, :2*size]
    dx0_ml, dy0_ml, I0_ml, xstd_ml, ystd_ml, phi_ml = fit_results
    f_ml = Gauss2D(hdu, x, y, dx0_ml, dy0_ml, I0_ml, xstd_ml, ystd_ml, phi_ml)
    f_res = f_cont - f_ml

    ## Plot the fitting result
    fig, axes = plt.subplots(figsize=(18, 7), nrows=1, ncols=3)
    plt.subplots_adjust(wspace=0)
    ax0, ax1, ax2 = axes
    im0 = ax0.imshow(f_cont, vmin=0, vmax=0.9, cmap='jet', origin='lower')
    ax0.contour(f_cont, cont_level, colors=['k'], linewidths=0.5)
    ax0.text(10, 10, "DATA", color="w")
    im1 = ax1.imshow(f_ml, vmin=0, vmax=0.9, cmap='jet', origin='lower')
    ax1.contour(f_ml, cont_level, colors=['k'], linewidths=0.5)
    ax1.text(10, 10, "MODEL", color="w")
    im2 = ax2.imshow(f_res, vmin=-0.1, vmax=0.1, cmap='jet', origin='lower')
    ax2.contour(f_res, cont_level, colors=['k'], linewidths=0.5)
    for ax in axes[:]:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.13, 0.1, 0.51, 0.05])
    cb_ax = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
    cb_ax.set_label(r"CONTINUUM [$\mathrm{Jy\,beam^{-1}}$]")
    cbar_res = fig.add_axes([0.645, 0.1, 0.255, 0.05])
    cb_res = fig.colorbar(im2, cax=cbar_res, orientation='horizontal')
    cb_res.set_label(r"$f_\mathrm{res}$ [$\mathrm{Jy\,beam^{-1}}$]")
    #plt.show()

# %%
def log_prior(para):
    x0, y0, I0, xstd0, ystd0, phi = para
    if -1.<x0<1. and -1.<y0<1. and 1e-5<I0<50. and 0.<xstd0<=10. and 0.<ystd0<=10.0 and 0.<phi<=360:
        return 0.0
    return -np.inf

def log_probability(para, x, y, f, ferr):
    lp = log_prior(para)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(para, x, y, f, ferr)

x, y = np.mgrid[:2*size, :2*size]
fit_results = [-0.2, 0.2, 5., 0.05, 0.05, 30.]
#def fit_mcmc(fit_results):
from multiprocessing import Pool
import emcee
output_dir = "/home/qyfei/Desktop/Results/map_visualization/fitting/Results/PG0050/sersic/"

x0_g_ml, y0_g_ml, I0_ml, xstd_ml, ystd_ml, phi_ml = fit_results
pos = fit_results + 1e-4 * np.random.randn(200, 6)
backname = "cont_tutorial_gauss.h5"
nwalkers, ndim = pos.shape
backend = emcee.backends.HDFBackend(output_dir+backname)
backend.reset(nwalkers, ndim)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, f_cont, f_err), pool=pool, backend=backend)
    sampler.run_mcmc(pos, 1000, progress=True)
    ## Check fitting result

# %%
fig, axes = plt.subplots(6, figsize=(10, 6), sharex=True)
samples_c = sampler.get_chain()
labels = ["x0", "y0", "I0", "xstd", "ystd", "phi"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples_c[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples_c))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

# %%
output_dir = "/home/qyfei/Desktop/Results/map_visualization/fitting/Results/PG0050/sersic/"
import h5py
import numpy as np
import matplotlib.pyplot as plt
import corner

#with h5py.File(output_dir+"tutorial.h5", "r") as f:
#    print(list(f.keys()))

f = h5py.File(output_dir+"cont_tutorial_gauss.h5", "r")
accepted = f['mcmc']['accepted']
chain = f['mcmc']['chain']
log_prob = f['mcmc']['log_prob']

get_chain = np.reshape(chain[300:], (700*200, 6))

# %%
cont_pix = np.pi*0.315*0.279/4/np.log(2)/0.05**2

labels = ["x", "y", "Ie", "xstd", "ystd", "phi"]
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
        get_chain, labels=labels, truths=[para_out[0],para_out[1],para_out[2],para_out[3],para_out[4],para_out[5]]
    )
#plt.savefig(output_dir+"corner.pdf", bbox_inches="tight")

# %%
## Output the mcmc fitting result and compare it with observation
#para_outc, para_outc_m, para_outc_p = fit_mcmc(fit_results)

x, y = np.mgrid[:2*size, :2*size]
f_model_cont = Gauss2D(hdu, x, y, para_out[0], para_out[1], para_out[2], para_out[3], para_out[4], para_out[5])
f_res_cont = f_cont - f_model_cont
cmap = "Greys"
fig, axes = plt.subplots(figsize=(18, 7), nrows=1, ncols=3)
plt.subplots_adjust(wspace=0)
ax0, ax1, ax2 = axes
im0 = ax0.imshow(f_cont, vmin=-0.039, vmax=0.9, cmap=cmap, origin='lower')
ax0.contour(f_cont, cont_level, colors=['k'], linewidths=1)
ax0.text(10, 10, "DATA", color="k")
im1 = ax1.imshow(f_model_cont, vmin=-0.039, vmax=0.9, cmap=cmap, origin='lower')
ax1.contour(f_model_cont, cont_level, colors=['k'], linewidths=1)
ax1.text(10, 10, "MODEL", color="k")
im2 = ax2.imshow(f_res_cont, vmin=-0.06, vmax=0.06, cmap=cmap, origin='lower')
ax2.contour(f_res_cont, cont_level, colors=['k'], linewidths=1)

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

#plt.savefig('/home/qyfei/Desktop/Results/map_visualization/fitting/Results/PG0050/sersic/CO21_cont_fit_gauss.pdf', bbox_inches='tight', dpi=300)

# %%
from astropy.modeling.models import Sersic2D, Gaussian2D
model_cont = Gaussian2D(5.057, 100, 100, 0.074/0.05, 0.042/0.05, -np.radians(23.447))
model_cont_image = model_cont(x, y)

model_mom0 = Sersic2D(7.454, 0.136/0.05, 3.506, 100, 100, 0.500, -np.radians(208.069)) + Sersic2D(0.426, 1.313/0.05, 0.454, 100, 100, 0.213, -np.radians(140.881)) + Sersic2D(1.901, 0.517/0.05, 0.226, 100, 100, 0.707, -np.radians(33.514))
model_mom0_image = model_mom0(x, y)

plt.imshow(model_cont_image, origin='lower', cmap="Greys")
plt.contour(f_model, mom0_level)
plt.xlim(50, 150)
plt.ylim(50, 150)

# %%
CO_pix = np.pi*0.356*0.316/0.05/0.05/4/np.log(2)
N = np.where(model_cont_image>=2*0.013)
N = np.where(f_cont>=2*0.013)

radius = np.sqrt((x-100)**2 + (y-100)**2)
mask = radius<=2*0.174/0.05
print(len(N[0]))
np.nansum(f_mom0[N])/CO_pix
np.nansum(f_mom0*mask)/CO_pix

# %%
np.nansum(model_mom0_image)/CO_pix
# %%
np.nansum(model_cont_image[N])/CO_pix
# %%
