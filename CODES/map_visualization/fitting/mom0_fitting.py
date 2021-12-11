from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colorbar as colorbar
from matplotlib.transforms import Affine2D
from astropy.visualization.wcsaxes import WCSAxes
import astropy.units as u
from spectral_cube import SpectralCube
from astropy.modeling import models, fitting
from astropy.stats import sigma_clipped_stats
import matplotlib

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
    x0, y0, Ie, Re, n, ellip, theta, x0_g, y0_g, I0, xstd0, ystd0, phi = para
    model0 = Disk2D(hdu, x, y, x0, y0, Ie, Re, n, ellip, theta)
    model1 = Gauss2D(hdu, x, y, x0_g, y0_g, I0, xstd0, ystd0, phi)
    model = model0 + model1
    sigma2 = zerr ** 2
    return -0.5 * np.sum((z - model) ** 2 / sigma2 + np.log(2*np.pi*sigma2))

mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file)
size = 100                  ## The size of map
xpos, ypos = pos_cen       ## The position of center
f_mom0 = mom0[xpos-size:xpos+size, ypos-size:ypos+size] ## The moment map that we want to fit
f_err = r                   ## The rms noise of the moment0 map
mom0_level = np.array([-1,1,2,4,8,16,32])*3*f_err

def fit_mini(f_mom0):
    x, y = np.mgrid[:2*size, :2*size]
    x0, y0, Ie, Re, n, ellip, theta = 0.4, 0.1, 0.5, 1.25, 0.589, 0.111, 0.483
    x0_g, y0_g, I0, xstd, ystd, phi = 0.15, 0., 17, 0.1, 0.2, 4.16

    print("Begin Maximum likelyhood fitting:")
    np.random.seed(42)
    from scipy.optimize import minimize
    nll = lambda *args: -log_likelihood(*args)
    initial = np.array([x0, y0, Ie, Re, n, ellip, theta, x0_g, y0_g, I0, xstd, ystd, phi]) + 0.1 * np.random.randn(13)#
    soln = minimize(nll, initial, args=(x, y, f_mom0, f_err), method="Nelder-Mead")
    x0_ml, y0_ml, Ie_ml, Re_ml, n_ml, ellip_ml, theta_ml = soln.x[:7]
    x0_g_ml, y0_g_ml, I0_ml, xstd_ml, ystd_ml, phi_ml = soln.x[7:]

    print("Maximum likelihood estimates of Disk:")
    print("x0 = {0:.3f}".format(x0_ml))
    print("y0 = {0:.3f}".format(y0_ml))
    print("Ie = {0:.3f}".format(Ie_ml))
    print("Re = {0:.3f}".format(Re_ml))
    print("n = {0:.3f}".format(n_ml))
    print("ellip = {0:.3f}".format(ellip_ml))
    print("theta = {0:.3f}".format(theta_ml))

    print("Maximum likelihood estimates of Core:")
    print("x0 = {0:.3f}".format(x0_g_ml))
    print("y0 = {0:.3f}".format(y0_g_ml))
    print("Ie = {0:.3f}".format(I0_ml))
    print("xstd = {0:.3f}".format(xstd_ml))
    print("ystd = {0:.3f}".format(ystd_ml))
    print("phi = {0:.3f}".format(phi_ml))
    return soln.x


def plot_fit_mini(fit_results):
    ## fit_result = fit_mini(path, file)
    x0_ml, y0_ml, Ie_ml, Re_ml, n_ml, ellip_ml, theta_ml = fit_results[:7]
    x0_g_ml, y0_g_ml, I0_ml, xstd_ml, ystd_ml, phi_ml = fit_results[7:]
    x, y = np.mgrid[:2*size, :2*size]
    level = np.array([-1,1,2,4,8,16])*3*f_err
    f_ml0 = Disk2D(hdu, x, y, x0_ml, y0_ml, Ie_ml, Re_ml, n_ml, ellip_ml, theta_ml)
    f_ml1 = Gauss2D(hdu, x, y, x0_g_ml, y0_g_ml, I0_ml, xstd_ml, ystd_ml, phi_ml)
    f_ml = f_ml0 + f_ml1
    f_res = f_mom0 - f_ml

    ## Plot the fitting result
    fig, axes = plt.subplots(figsize=(18, 7), nrows=1, ncols=3)
    plt.subplots_adjust(wspace=0)
    ax0, ax1, ax2 = axes
    im0 = ax0.imshow(f_mom0, vmin=0, vmax=8, cmap='jet', origin='lower')
    ax0.contour(f_mom0, mom0_level, colors=['k'], linewidths=0.5)
    ax0.text(10, 10, "DATA", color="w")
    im1 = ax1.imshow(f_ml, vmin=0, vmax=8, cmap='jet', origin='lower')
    ax1.contour(f_ml, mom0_level, colors=['k'], linewidths=0.5)
    ax1.text(10, 10, "MODEL", color="w")
    im2 = ax2.imshow(f_res, vmin=-0.6, vmax=0.6, cmap='jet', origin='lower')
    ax2.contour(f_res, mom0_level, colors=['k'], linewidths=0.5)
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
    #plt.show()

def log_prior(para):
    x0, y0, Ie, Re, n, ellip, theta = para[:7]
    x0_g, y0_g, I0, xstd, ystd, phi = para[7:]
    if -1<x0<1 and -1<y0<1 and 1e-5<Ie<20. and 0.<Re<3. and 0.0<n<10.0 and 0<theta<2*np.pi and -1<x0_g<1 and -1<y0_g<1 and 1e-5<Ie<50. and 0.<xstd<=2. and 0.0<ystd<=2.0 and 0<phi<=2*np.pi:
        return 0.0
    return -np.inf

def log_probability(para, x, y, f, ferr):
    lp = log_prior(para)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(para, x, y, f, ferr)

def fit_mcmc(f_mom0):
    fit_results = fit_mini(f_mom0)
    x, y = np.mgrid[:2*size, :2*size]

    print("Begin mcmc fitting:")
    from multiprocessing import Pool
    import emcee
    pos = fit_results + 1e-4 * np.random.randn(50, 13)
    nwalkers, ndim = pos.shape
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, f_mom0, f_err), pool=pool)
        sampler.run_mcmc(pos, 5000, progress=True)

    fig, axes = plt.subplots(13, figsize=(10, 13), sharex=True)
    samples = sampler.get_chain()
    labels = ["$x_0$", "$y_0$", "Ie", "Re", "n", "ellip", "theta", "$x_0g$", "$y_0g$", "I0", "xstd", "ystd", "phi"]
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
        flat_samples, labels=labels, truths=[para_out[0],para_out[1],para_out[2],para_out[3],para_out[4],para_out[5],para_out[6],para_out[7],para_out[8],para_out[9],para_out[10],para_out[11],para_out[12]]
    )
    np.savetxt('/home/qyfei/Desktop/Codes/Result/PG0050/mom0_fit_para.txt', np.array([para_out, para_out_m, para_out_p]).T)
    return para_out, para_out_m, para_out_p

para_out, para_out_m, para_out_p = fit_mcmc(f_mom0)
x, y = np.mgrid[:2*size, :2*size]
f_outer = Disk2D(x, y, para_out[0], para_out[1], para_out[2], para_out[3], para_out[4], para_out[5], para_out[6])
f_inner = Gauss2D(x, y, para_out[7], para_out[8], para_out[9], para_out[10], para_out[11], para_out[12])
f_model = f_outer + f_inner
f_total_res = f_mom0 - f_model

fig, axes = plt.subplots(figsize=(18, 7), nrows=1, ncols=3)
plt.subplots_adjust(wspace=0)
ax0, ax1, ax2 = axes
im0 = ax0.imshow(f_mom0, vmin=-0.086, vmax=8, cmap='jet', origin='lower')
ax0.contour(f_mom0, mom0_level, colors=['k'], linewidths=0.5)
ax0.text(10, 10, "DATA", color="w")
im1 = ax1.imshow(f_model, vmin=0-0.086, vmax=8, cmap='jet', origin='lower')
ax1.contour(f_model, mom0_level, colors=['k'], linewidths=0.5)
ax1.text(10, 10, "MODEL", color="w")
im2 = ax2.imshow(f_total_res, vmin=-0.6, vmax=0.6, cmap='jet', origin='lower')
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

plt.savefig('/home/qyfei/Desktop/Codes/Result/PG0050/CO21_mom0_fit.pdf', bbox_inches='tight', dpi=300)
