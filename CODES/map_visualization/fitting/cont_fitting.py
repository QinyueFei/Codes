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
ypos, xpos = pos_cen[1][0], pos_cen[0][0]
size = 100
sigma=1.3e-02
f_cont = cont[xpos-size:xpos+size, ypos-size:ypos+size]
f_err = sigma
cont_level = np.array([-1,1,2,4,8,16])*3*f_err

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
    if -1.<x0<1. and -1.<y0<1. and 1e-5<I0<50. and 0.<xstd0<=10. and 0.<ystd0<=10.0 and -360<phi<=360:
        return 0.0
    return -np.inf

def log_probability(para, x, y, f, ferr):
    lp = log_prior(para)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(para, x, y, f, ferr)

x, y = np.mgrid[:2*size, :2*size]
fit_results = [0., 0., 5.057, 0.074, 0.042, 23.447]
#def fit_mcmc(fit_results):
from multiprocessing import Pool
import emcee
output_dir = "/home/qyfei/Desktop/Results/map_visualization/fitting/Results/PG0050/cont/"

x0_g_ml, y0_g_ml, I0_ml, xstd_ml, ystd_ml, phi_ml = fit_results
pos = fit_results + 1e-4 * np.random.randn(200, 6)
backname = "cont_tutorial_gauss_new.h5"
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
output_dir = "/home/qyfei/Desktop/Results/map_visualization/fitting/Results/PG0050/cont/"
import h5py
import numpy as np
import matplotlib.pyplot as plt
import corner

#with h5py.File(output_dir+"tutorial.h5", "r") as f:
#    print(list(f.keys()))

f = h5py.File(output_dir+"cont_tutorial_gauss_new.h5", "r")
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
## Deconvolved result
from matplotlib.colors import LogNorm
from astropy.modeling.models import Sersic2D, Gaussian2D
model_cont = Gaussian2D(5.057, 100+0.010/0.05, 100-0.019/0.05, 0.074/0.05, 0.042/0.05, -np.radians(23.447))
model_cont_image = model_cont(x, y)

# model_mom0 = Sersic2D(7.454, 0.136/0.05, 3.506, 100, 100, 0.500, -np.radians(208.069)) + Sersic2D(0.426, 1.313/0.05, 0.454, 100, 100, 0.213, -np.radians(140.881)) + Sersic2D(1.901, 0.517/0.05, 0.226, 100, 100, 0.707, -np.radians(33.514))
model_mom0 = Gaussian2D(29.903, 100+0.015/0.05, 100-0.016/0.05, 0.094/0.05, 0.052/0.05, -np.radians(16.592)) + Sersic2D(0.437, 1.297/0.05, 0.476, 100+0.015/0.05, 100-0.016/0.05, 0.202, -np.radians(141.901)) + Sersic2D(2.367, 0.495/0.05, 0.299, 100+0.015/0.05, 100-0.016/0.05, 0.684, -np.radians(33.653))
model_mom0_image = model_mom0(x, y)

plt.imshow(model_cont_image, origin='lower', vmin=0.013, vmax=np.percentile(model_cont_image, [99.999]), cmap="Greys", norm=LogNorm())
plt.contour(model_cont_image, levels=np.array([0.01, 0.05, 0.1, 0.68, 0.95, 0.997])*np.nanmax(model_cont_image))
# plt.contour(f_model, mom0_level)
plt.xlim(80, 120)
plt.ylim(80, 120)

beam2pixels = 39.88
N = np.where(model_cont_image>=0.01*np.nanmax(model_cont_image))
f_cont_decon = np.nansum(model_cont_image[N]/beam2pixels)
print("Continuum flux in region is", f_cont_decon, "mJy")

# %%
CO_pix = np.pi*0.356*0.316/0.05/0.05/4/np.log(2)

radius = np.sqrt((x-100)**2 + (y-100)**2)
mask = radius<=2*0.174/0.05

print(len(N[0]))
f_CO_decon = np.nansum(model_mom0_image[N])/CO_pix
print("CO(2-1) flux is:", f_CO_decon, "Jy/beam km/s")
# print(np.nansum(model_mom0_image*mask)/CO_pix)

# %%
## Area
from astropy.cosmology import Planck15
import astropy.units as u
A_decon = np.pi*9*0.074*0.042*u.arcsec**2/Planck15.arcsec_per_kpc_proper(0.061)**2/np.cos(np.deg2rad(41))
A_decon
# %%
## surface density

## SFR surface density
ef_cont = 4.2e-05*1e3

SFR_decon = f_cont_decon/2.2*26.3*u.Unit("M_sun/yr")
eSFR_decon = ef_cont/2.2*26.3*u.Unit("M_sun/yr")
Sigma_SFR_decon = SFR_decon/A_decon
eSigma_SFR_decon = eSFR_decon/A_decon

## Molecular gas surface density
DL = Planck15.luminosity_distance(0.061)
L_CO_decon = 3.25e7*f_CO_decon*DL.value**2/(1+0.061)/(230.58)**2/0.9
M_H2_decon = 1.27*L_CO_decon
M_H2_low_decon = (1.27-0.71)*L_CO_decon
M_H2_up_decon = (1.27+0.83)*L_CO_decon
M_H2_decon_tra = 3.1*1.4*L_CO_decon

Sigma_H2_decon = M_H2_decon*u.Unit("M_sun")/A_decon.to("pc^2")
Sigma_H2_low_decon = M_H2_low_decon*u.Unit("M_sun")/A_decon.to("pc^2")
Sigma_H2_up_decon = M_H2_up_decon*u.Unit("M_sun")/A_decon.to("pc^2")
e1Sigma_H2_decon = Sigma_H2_low_decon - Sigma_H2_decon
e2Sigma_H2_decon = Sigma_H2_up_decon - Sigma_H2_decon
Sigma_H2_decon_tra = M_H2_decon_tra*u.Unit("M_sun")/A_decon.to("pc^2")

# %%
