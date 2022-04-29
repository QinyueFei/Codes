# %%
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
#from Dynamics.models import Rs
#import astropy.units as u
import matplotlib
#from scipy import integrate

plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

# path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
# file = "PG0050_CO21-combine-line-10km-mosaic-mom0.fits"
# path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/NOEMA/w20cf/w20cf_taper/w20cf002/"
# file = "F08542+1920_CO32_taper_mom0.fits"
#name_pbc = "PG0050_CO21-combine-line-10km-mosaic-pbc-mom0-rms.fits"

path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG0923/data/"
# file = "PG0923.dilmsk.mom0.fits.gz"
file = "PG0923+129_CO21_final_image_mom0.fits"
# efile = "PG0923.dilmsk.emom0.fits.gz"
# freefile = "PG0923+129_CO21_final_image_mom0_free.fits"
# meanfile = "PG0923+129_CO21_final_image_mean.fits"

from map_visualization.maps import beam, load_mom0, load_mom0_NOEMA
from map_visualization.fitting.module import Disk2D
# freemom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, freefile)
# mean, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, meanfile)
mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file)

# mom0 = fits.open(path+file)[0].data
# mom0_origin = fits.open(path+file)[0].data
# emom0 = fits.open(path+efile)[0].data

# %%
#mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0_NOEMA(path, file)
# mom0[np.isnan(mom0)] = freemom0[np.isnan(mom0)]
# emom0[np.isnan(emom0)] = mean[np.isnan(emom0)]/1e3*10.14 
#1e3 is used to convert to mJy, 10.14 is the chan width in km/s

size = 75                  ## The size of map
xpos, ypos = 88, 90#pos_cen[0][0], pos_cen[1][0]       ## The position of center
f_mom0 = mom0[xpos-size:xpos+size, ypos-size:ypos+size] ## The moment map that we want to fit
f_err = 0.055#emom0[xpos-size:xpos+size, ypos-size:ypos+size]                   ## The rms noise of the moment0 map
mom0_level = np.array([-1,1,2,4,8,16,32])*2*f_err
# f_mom0_origin = mom0_origin[xpos-size:xpos+size, ypos-size:ypos+size]

# %%
plt.imshow(f_mom0, origin='lower', cmap='jet')
plt.colorbar()
plt.contour(f_mom0, mom0_level, colors=['k'])

# %%
## Fit with single sersic profile
def log_likelihood(para, x, y, z, zerr):
    x0, y0, Ie, Re, n, e, t = para
    emission = Disk2D(hdu, x, y, x0, y0, Ie, Re, n, e, t)         # The outer disk
    model = emission
    sigma2 = zerr ** 2
    return -0.5 * np.sum((z - model) ** 2 / sigma2 + np.log(2*np.pi*sigma2))

def log_prior(para):
    x0, y0, Ie, Re, n, e, t = para[:7]
    if not (-10<x0<10 and -10<y0<10 and 1e-5<Ie<50. and 0.<Re<5. and 0.0<n<10.0 and 0<e<=1 and -360<=t<=360):
        return -np.inf
    lp = 0#np.log(1./(np.sqrt(2*np.pi)*t_sigma)) - 0.5*(t - t_mu)**2/t_sigma**2-np.log(t_mu)
    return lp

def log_probability(para, x, y, f, ferr):
    lp = log_prior(para)
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + log_likelihood(para, x, y, f, ferr)


#def fit_mcmc(f_mom0):

fit_results = [-0.08, 0.048, 1.16, 2.51, 0.46, 0.31, 61]#fit_mini(f_mom0)
x, y = np.mgrid[:2*size, :2*size]
# f_mom0_cen = f_mom0 - Disk2D(hdu, x, y, 0.425, 0.948, 0.700, 3.469, 0.677, 0.778, 153.205)
#output_dir = "/home/qyfei/Desktop/Codes/CODES/map_visualization/fitting/Results/PG0050/sersic/"
# output_dir = "/home/qyfei/Desktop/Results/map_visualization/fitting/NOEMA/F08542/"
output_dir = "/home/qyfei/Desktop/Results/map_visualization/fitting/Results/PG0923/single_sersic_new/"

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
    sampler.run_mcmc(pos, 500, progress=True)

# %%
fig, axes = plt.subplots(7, figsize=(10, 7), sharex=True)
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
# np.savetxt(output_dir+'mom0_fit_para.txt', np.array([para_out, para_out_m, para_out_p]).T)

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import corner

output_dir = "/home/qyfei/Desktop/Results/map_visualization/fitting/Results/PG0923/single_sersic_new/"

with h5py.File(output_dir+"tutorial.h5", "r") as f:
    print(list(f.keys()))

f = h5py.File(output_dir+"tutorial.h5", "r")
accepted = f['mcmc']['accepted']
chain = f['mcmc']['chain']
log_prob = f['mcmc']['log_prob']

get_chain = np.reshape(chain[300:], (200*200, 7))

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
nlabels = len(labels)
ns = np.array([0,1,2,3,4,5,6,7])
plt.rc('font', family='dejavuserif', size=20)

fig = corner.corner(
        get_chain, labels=labels, truths=[para_out[0],para_out[1],para_out[2],para_out[3],para_out[4],para_out[5],para_out[6]], show_titles=True
    )

axes = np.array(fig.axes).reshape((nlabels, nlabels))

# Loop over the diagonal

# plt.savefig(output_dir+"corner.pdf", bbox_inches="tight")

# %%
x, y = np.mgrid[:2*size, :2*size]
#fit_results = [0.0, 0.0, 0.42, 1.3, 2.0, 0.25, 125]
# para_out = [-0., -0., 1.2, 2.0, 1.0, 0.4, 45]
#para_out = fit_results


f_model = Disk2D(hdu, x, y, para_out[0], para_out[1], para_out[2], para_out[3], para_out[4], para_out[5], para_out[6])
f_total_res = f_mom0 - f_model

fig, axes = plt.subplots(figsize=(18, 7), nrows=1, ncols=3)
plt.subplots_adjust(wspace=0)
ax0, ax1, ax2 = axes
vmin = -10*f_err
vmax = np.percentile(f_mom0, [99.99])
im0 = ax0.imshow(f_mom0, cmap='jet', origin='lower')
ax0.contour(f_mom0, mom0_level, colors=['k'], linewidths=0.5)
ax0.text(10, 10, "DATA", color="w")
im1 = ax1.imshow(f_model, cmap='jet', origin='lower')
ax1.contour(f_model, mom0_level, colors=['k'], linewidths=0.5)
ax1.text(10, 10, "MODEL", color="w")
im2 = ax2.imshow(f_total_res, cmap='jet', origin='lower')
ax2.contour(f_total_res, mom0_level, colors=['k'], linewidths=0.5)

for ax in axes[:]:
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.125, 0.115, 0.517, 0.05])
cb_ax = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
cb_ax.set_label(r"FLUX [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]")

cbar_res = fig.add_axes([0.643, 0.115, 0.257, 0.05])
cb_res = fig.colorbar(im2, cax=cbar_res, orientation='horizontal')
cb_res.set_label(r"$f_\mathrm{res}$ [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]")

rec = matplotlib.patches.Rectangle((-1, -1), 11, 11,
angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax0.add_artist(rec)
Beam = beam(hdu, 5., 5., 'w', pix_size)
ax0.add_artist(Beam[0])
# plt.savefig(output_dir+"mom0_fit.pdf", bbox_inches="tight", dpi=300)

# %%

fig, axes = plt.subplots(figsize=(12, 7), nrows=1, ncols=3)
plt.subplots_adjust(wspace=0)
ax0, ax1, ax2 = axes
im0 = ax0.imshow(f_model, vmin=vmin, vmax=vmax, cmap='jet', origin='lower')
ax0.contour(f_model, mom0_level, colors=['k'], linewidths=0.5)
ax0.text(10, 10, "INNER", color="w")
im2 = ax2.imshow(f_model, vmin=vmin, vmax=vmax, cmap='jet', origin='lower')
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
cosi = np.sqrt((0.798**2-0.14**2)/(1-0.14**2))
inc = np.arccos(cosi)
np.rad2deg(inc)
# %%
np.rad2deg(np.arccos(0.798))
# %%
