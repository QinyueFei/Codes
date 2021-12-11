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
f_cont = cont[xpos-size:xpos+size, ypos-size:ypos+size]
f_err = sigma
cont_level = np.array([-1,1,2,4,8,16])*2*f_err

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

fit_results = fit_mini(f_cont)

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

def log_prior(para):
    x0, y0, I0, xstd0, ystd0, phi = para
    if -1.<x0<1. and -1.<y0<1. and 1e-5<I0<50. and -2.<xstd0<=2. and -2.0<ystd0<=2.0 and -2*np.pi<phi<=2*np.pi:
        return 0.0
    return -np.inf

def log_probability(para, x, y, f, ferr):
    lp = log_prior(para)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(para, x, y, f, ferr)

def fit_mcmc(fit_results):
    from multiprocessing import Pool
    import emcee
    x0_g_ml, y0_g_ml, I0_ml, xstd_ml, ystd_ml, phi_ml = fit_results
    pos_c = fit_results + 1e-4 * np.random.randn(32, 6)
    nwalkers_c, ndim_c = pos_c.shape
    x, y = np.mgrid[:2*size, :2*size]
    with Pool() as pool:
        sampler_c = emcee.EnsembleSampler(nwalkers_c, ndim_c, log_probability, args=(x, y, f_cont, f_err), pool=pool)
        sampler_c.run_mcmc(pos_c, 5000, progress=True)
    ## Check fitting result

    fig, axes = plt.subplots(6, figsize=(10, 6), sharex=True)
    samples_c = sampler_c.get_chain()
    labels = ["x0", "y0", "I0", "xstd", "ystd", "phi"]
    for i in range(ndim_c):
        ax = axes[i]
        ax.plot(samples_c[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples_c))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    flat_samples_c = sampler_c.get_chain(discard=100, thin=15, flat=True)
    print(flat_samples_c.shape)

    import corner

    fig = corner.corner(
        flat_samples_c, labels=labels, truths=[x0_g_ml, y0_g_ml, I0_ml, xstd_ml, ystd_ml, phi_ml]
    )

    from IPython.display import display, Math

    for i in range(ndim_c):
        mcmc = np.percentile(flat_samples_c[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        display(Math(txt))

    ## Output parameters

    para_outc = np.zeros(ndim_c)
    para_outc_m = np.zeros(ndim_c)
    para_outc_p = np.zeros(ndim_c)

    for i in range(ndim_c):
        para_outc[i] = np.percentile(flat_samples_c[:, i], [50])
        para_outc_m[i] = np.percentile(flat_samples_c[:, i], [16])
        para_outc_p[i] = np.percentile(flat_samples_c[:, i], [84])
    return para_outc, para_outc_m, para_outc_p

## Output the mcmc fitting result and compare it with observation
para_outc, para_outc_m, para_outc_p = fit_mcmc(fit_results)
x, y = np.mgrid[:2*size, :2*size]
f_model_cont = Gauss2D(hdu, x, y, para_outc[0], para_outc[1], para_outc[2], para_outc[3], para_outc[4], para_outc[5])
f_res_cont = f_cont - f_model_cont

fig, axes = plt.subplots(figsize=(18, 7), nrows=1, ncols=3)
plt.subplots_adjust(wspace=0)
ax0, ax1, ax2 = axes
im0 = ax0.imshow(f_cont, vmin=-0.039, vmax=0.9, cmap='jet', origin='lower')
ax0.contour(f_cont, cont_level, colors=['k'], linewidths=0.5)
ax0.text(10, 10, "DATA", color="w")
im1 = ax1.imshow(f_model_cont, vmin=-0.039, vmax=0.9, cmap='jet', origin='lower')
ax1.contour(f_model_cont, cont_level, colors=['k'], linewidths=0.5)
ax1.text(10, 10, "MODEL", color="w")
im2 = ax2.imshow(f_res_cont, vmin=-0.1, vmax=0.1, cmap='jet', origin='lower')
ax2.contour(f_res_cont, cont_level, colors=['k'], linewidths=0.5)

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

rec = matplotlib.patches.Rectangle((0, 0), 10, 10,
angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax0.add_artist(rec)
Beam = beam(hdu, 5., 5., 'w', pix_size)
ax0.add_artist(Beam[0])
plt.show()
#plt.savefig('/home/qyfei/Desktop/Codes/Result/PG0050/CO21_cont_fit.pdf', bbox_inches='tight', dpi=300)
