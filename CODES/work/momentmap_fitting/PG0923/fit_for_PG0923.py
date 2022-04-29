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
file = "PG0923+129_CO21_final_image_mom0.fits"
freefile = "PG0923+129_CO21_final_image_mom0_free.fits"

from map_visualization.maps import beam
from map_visualization.fitting.module import Disk2D, Gauss2D, kernel, truncated_Disk2D_ring
# freemom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, freefile)
# mean, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, meanfile)

hdu = fits.open(path+file)[0]
mom0 = hdu.data[0][0]
wcs = WCS(hdu.header)
pos_cen = np.where(mom0 == np.nanmax(mom0))
pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
size = 80

free_mom0 = fits.open(path+freefile)[0].data[0][0]
r = np.nanstd(free_mom0)

# %%
fit_results = [-0.08, 0.048, 1.16, 2.51, 0.46, 0.31, 61]#fit_mini(f_mom0)
x, y = np.mgrid[:2*size, :2*size]

size = 75                  ## The size of map
xpos, ypos = 88, 90#pos_cen[0][0], pos_cen[1][0]       ## The position of center
f_mom0 = mom0[xpos-size:xpos+size, ypos-size:ypos+size] ## The moment map that we want to fit
f_err = r                   ## The rms noise of the moment0 map
mom0_level = np.array([-1,1,2,4,8,16,32])*2*f_err

# %%

def log_likelihood(para, x, y, z, zerr):
    x0, y0, I0, xstd0, ystd0, phi = para[:6]
    Ie1, Re1, n1, e1, t1 = para[6:11]
    # Ie2, Re2, n2, e2, t2 = para[11:]
    model0 = Gauss2D(hdu, x, y, x0, y0, I0, xstd0, ystd0, phi)  # The core component
    #model0 = Disk2D(hdu, x, y, x0, y0, Ie, Re, n, e, t)         
    model1 = Disk2D(hdu, x, y, x0, y0, Ie1, Re1, n1, e1, t1)    # The disk component
    # model2 = Disk2D(hdu, x, y, x0, y0, Ie2, Re2, n2, e2, t2)    # The bar component
    model = model0 + model1 #+ model2
    sigma2 = zerr ** 2
    return -0.5 * np.sum((z - model) ** 2 / sigma2 + np.log(2*np.pi*sigma2))

# %%
## Fit the observation with double Sersic profile and one Gaussian profile

def log_prior(para):
    x0, y0, I0, xstd0, ystd0, phi = para[:6]
    Ie1, Re1, n1, e1, t1 = para[6:11]
    # Ie2, Re2, n2, e2, t2 = para[11:]

    if not (-10<x0<10 and -10<y0<10 and 0.01<xstd0<50. and 0.01<ystd0<50.0 and -360<phi<=360 and 1e-5<Ie1<1e5 and 0.<Re1<=20. and 0.01<n1<10. and 0.01<e1<=1 and -360<t1<=360):
        #  and 1e-5<Ie2<1e5 and 0.1<Re2<=20. and 0.01<n2<10. and 0.01<e2<=1. and -360<t2<=360
        return -np.inf
    lp = 0
    return lp

def log_probability(para, x, y, f, ferr):
    lp = log_prior(para)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(para, x, y, f, ferr)

#def fit_mcmc(f_mom0_, f_err_):
#f_mom0, f_err = f_mom0_, f_err_
    # Ie    Re    n   e   t  
comp_0 = [0.27, -0.36, -1.23, 0.49, 1.32, 151.59]# *250
comp_1 = [1.94, 2.02, 0.04, 0.40, 67.05]
# comp_2 = [0.27, 3.75, 0.14, 0.04, 162.02]
fit_results = comp_0 + comp_1# + comp_2

x, y = np.mgrid[:2*size, :2*size]

# %%
from scipy.optimize import minimize
para_ini = [0.27, -0.36, -1.23, 0.49, 1.32, 151.59,
            1.94, 2.02, 0.04, 0.40, 67.05]#,
            # 0.27, 3.75, 0.14, 0.04, 162.02]
np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array(para_ini) + 0.1 * np.random.randn(11)
soln = minimize(nll, initial, args=(x, y, f_mom0, r), method="Nelder-Mead")
para_mini = soln.x

print("Maximum likelihood estimates:")
print("dx0 = {0:.3f}".format(para_mini[0]))
print("dy0 = {0:.3f}".format(para_mini[1]))
print("I0 = {0:.3f}".format(para_mini[2]))
print("xstd = {0:.3f}".format(para_mini[3]))
print("ystd = {0:.3f}".format(para_mini[4]))
print("phi = {0:.3f}".format(para_mini[5]))
print("Ie1 = {0:.3f}".format(para_mini[6]))
print("Re1 = {0:.3f}".format(para_mini[7]))
print("n1 = {0:.3f}".format(para_mini[8]))
print("e1 = {0:.3f}".format(para_mini[9]))
print("theta1 = {0:.3f}".format(para_mini[10]))
# print("Ie2 = {0:.3f}".format(para_mini[11]))
# print("Re2 = {0:.3f}".format(para_mini[12]))
# print("n2 = {0:.3f}".format(para_mini[13]))
# print("e2 = {0:.3f}".format(para_mini[14]))
# print("theta2 = {0:.3f}".format(para_mini[15]))

f_void = Gauss2D(hdu, x, y, para_mini[0], para_mini[1], para_mini[2], para_mini[3], para_mini[4], para_mini[5])
f_disk1 = Disk2D(hdu, x, y, para_mini[0], para_mini[1], para_mini[6], para_mini[7], para_mini[8], para_mini[9], para_mini[10])
# f_disk2 = Disk2D(hdu, x, y, para_mini[0], para_mini[1], para_mini[11], para_mini[12], para_mini[13], para_mini[14], para_mini[15])

# %%
output_dir = "/home/qyfei/Desktop/Results/map_visualization/fitting/Results/PG0923/two_components/"
print("Begin mcmc fitting:")
from multiprocessing import Pool
import emcee

pos = fit_results + 1e-4 * np.random.randn(400, 11)
backname = "tutorial.h5"
nwalkers, ndim = pos.shape
backend = emcee.backends.HDFBackend(output_dir+backname)
backend.reset(nwalkers, ndim)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, f_mom0, r), pool=pool, backend=backend)
    sampler.run_mcmc(pos, 1000, progress=True)

# %%
fig, axes = plt.subplots(16, figsize=(10, 16), sharex=True)
samples = sampler.get_chain()
labels = ["$x_0$", "$y_0$", "I0", "xstd", "ystd", "phi", "Ie1", "Re1", "n1", "e1", "t1", "Ie2", "Re2", "n2", "e2", "t2"]
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

output_dir = "/home/qyfei/Desktop/Results/map_visualization/fitting/Results/PG0923/two_components/"

with h5py.File(output_dir+"tutorial.h5", "r") as f:
    print(list(f.keys()))
f = h5py.File(output_dir+"tutorial.h5", "r")
accepted = f['mcmc']['accepted']
chain = f['mcmc']['chain']
log_prob = f['mcmc']['log_prob']
get_chain = np.reshape(chain[800:], (400*200, 11))

# %%
labels = ["$x_0$", "$y_0$", "I0", "xstd", "ystd", "phi", "Ie1", "Re1", "n1", "e1", "t1"]#, "Ie2", "Re2", "n2", "e2", "t2"
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
        get_chain, labels=labels, truths=[para_out[0],para_out[1],para_out[2],para_out[3],para_out[4],para_out[5],para_out[6],para_out[7],para_out[8],para_out[9],para_out[10]]
    )#,para_out[11], para_out[12], para_out[13], para_out[14], para_out[15]

# %%
from matplotlib import colors
# para_out = [0, 0, -0.5, 1.0, 0.8, 60,
            # 0.4/2, 5, 0.5, 0.2, 60]#,
# para_out = para_mini

cmap = "jet"
x, y = np.mgrid[:2*size, :2*size]
f_bulge = Gauss2D(hdu, x, y, para_out[0], para_out[1], para_out[2], para_out[3], para_out[4], para_out[5])
f_disk1 = Disk2D(hdu, x, y, para_out[0], para_out[1], para_out[6], para_out[7], para_out[8], para_out[9], para_out[10])
# f_disk2 = Disk2D(hdu, x, y, para_mini[0], para_mini[1], para_mini[11], para_mini[12], para_mini[13], para_mini[14], para_mini[15])


f_model = f_bulge + f_disk1# + f_disk2
f_total_res = f_mom0 - f_model

mom0_level = np.array([-1,1,2,4,8,16,32,64,128])*3*0.055#r_test
vmin, vmax = 0.25, np.percentile(f_mom0, [99.999])

fig, axes = plt.subplots(figsize=(18, 7), nrows=1, ncols=3)
plt.subplots_adjust(wspace=0)
ax0, ax1, ax2 = axes
im0 = ax0.imshow(f_mom0, vmin=-vmin, vmax=vmax, cmap=cmap, origin='lower', norm=colors.PowerNorm(gamma=0.525))
ax0.contour(f_mom0, mom0_level, colors=["k"], linewidths=1.)
ax0.text(10, 10, "DATA", color="w")
im1 = ax1.imshow(f_model, vmin=-vmin, vmax=vmax, cmap=cmap, origin='lower', norm=colors.PowerNorm(gamma=0.525))
ax1.contour(f_model, mom0_level, colors=["k"], linewidths=1.)
ax1.text(10, 10, "MODEL", color="w")
im2 = ax2.imshow(f_total_res, vmin=-0.86, vmax=0.86, cmap=cmap, origin='lower')
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

# %%
fig, axes = plt.subplots(figsize=(24, 7), nrows=1, ncols=4)
plt.subplots_adjust(wspace=0)
ax0, ax1, ax2, ax3 = axes
im0 = ax0.imshow(f_bulge, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower', norm=LogNorm())
ax0.contour(f_bulge, mom0_level, colors=['k'], linewidths=1)
ax0.text(10, 10, "CORE", color="k")
im1 = ax1.imshow(f_disk1, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower', norm=LogNorm())
ax1.contour(f_disk1, mom0_level, colors=['k'], linewidths=1)
ax1.text(10, 10, "DISK", color="k")
# im2 = ax2.imshow(f_disk2, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower', norm=LogNorm())
# ax2.contour(f_disk2, mom0_level, colors=['k'], linewidths=1)
ax2.text(10, 10, "BAR", color="k")
im3 = ax3.imshow(f_model, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower', norm=LogNorm())
ax3.contour(f_model, mom0_level, colors=['k'], linewidths=1)
ax3.text(10, 10, "TOTAL", color="k")
for ax in axes[:]:
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.125, 0.1, 0.775, 0.05])
cb_ax = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
cb_ax.set_label(r"FLUX [$\mathrm{K\,km\,s^{-1}}$]")

# %%
def truncation(hdu, x, y, dx0, dy0, r_break, dr_soft, q_, theta_):
    pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    size = len(x)/2
    x0, y0 = size - dx0/pix_size, size - dy0/pix_size
    theta = -np.radians(theta_)
    a, b  = r_break, q_ * r_break
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    x_maj = (x - x0) * cos_theta + (y - y0) * sin_theta
    x_min = -(x - x0) * sin_theta + (y - y0) * cos_theta
    z = np.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)
    B = 2.65 - 4.98 * r_break/dr_soft
    trunc = 0.5 * (np.tanh((2 - B) * z + B) + 1)
    return trunc

# %%
kernel_CO = kernel(hdu)

from scipy.signal import convolve as scipy_convolve
testf = Disk2D(hdu, x, y, 0, 0, 1, 5, 1, 0.5, 60)

test_trunc = scipy_convolve(truncation(hdu, x, y, 0, 0, 10, 10, 0.5, 60), kernel_CO, mode='same', method='fft')

plt.figure(figsize=(10, 8))
plt.imshow(testf * test_trunc, origin='lower', cmap='jet', norm=colors.PowerNorm(0.5))
plt.colorbar()

