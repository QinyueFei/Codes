# %%
import numpy as np
import matplotlib
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
import astropy.constants as c
plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)
plt.rcParams['xtick.major.bottom'] = True
plt.rcParams['ytick.major.left'] = True

def beam(HDU, XPOS, YPOS, col, cellsize):
    hdu = HDU
    xpos, ypos = XPOS, YPOS
    c = col
    cell = cellsize
    bmaj, bmin, bpa = hdu.header['BMAJ'], hdu.header['BMIN'], hdu.header['BPA']
    Bmaj = bmaj*u.Unit('deg').to('arcsec')/cell
    Bmin = bmin*u.Unit('deg').to('arcsec')/cell
    Beam = matplotlib.patches.Ellipse((xpos,ypos), Bmaj, Bmin, 90+bpa, edgecolor='k', facecolor='gray', fill=True, zorder=3)
    return Beam, Bmaj, Bmin

# %%
###############
## Load data ##
###############
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
file = "PG0050_CO21-combine-line-10km-mosaic-pbc.fits"
cube = "PG0050_CO21-combine-line-10km-mosaic-pbc.fits"
#spec = "F08238p0752_CO3-2_spectrum.txt"
# Load spectrum
#freq = np.loadtxt(path+spec)[:,0]

hdu = fits.open(path+file)[0]
mom0 = hdu.data[0]
sigma = sigma_clipped_stats(mom0)[-1]

# build the mask
pos_cen = [395, 399]
yy,xx = np.indices([800, 800],dtype='float')
radius = ((yy-pos_cen[1])**2+(xx-pos_cen[0])**2)**0.5

mask_tot = (abs(mom0)>=2.*sigma) & (radius<=200)
mask_cen = radius<=16
annulus = (radius>20) & (radius<=40)

# load datacube and calculate the spectrum in km/s
cube_data = SpectralCube.read(path+cube)
freq = cube_data.spectral_axis#.to('km/s')
#cube_data_with_mask = cube_data.with_mask(mask)
masked_cube_data = cube_data.with_spectral_unit(unit='km/s', rest_value=217.253*u.GHz, velocity_convention='radio')
# calculate the spectrum with mask in Jy/beam

hdu = fits.open(path+cube)[0]
cube_data = hdu.data[0]

spectrum_tot = cube_data*mask_tot
spectrum_cen = cube_data*mask_cen
spectrum_ann = cube_data*annulus
# calculate the beam area

bmaj = hdu.header['BMAJ']
bmin = hdu.header['BMIN']
delt = hdu.header['CDELT1']
beam_area = np.pi*bmaj*bmin/delt**2/4/np.log(2)

# calculate the total flux, y-axis
flux_tot = np.nansum(spectrum_tot, axis=(1,2))/beam_area*1e3
flux_cen = np.nansum(spectrum_cen, axis=(1,2))/beam_area*1e3
flux_ann = np.nansum(spectrum_ann, axis=(1,2))/beam_area*1e3
# calculate the x-axis
velo = masked_cube_data.spectral_axis

sigma_tot = sigma_clipped_stats(flux_tot[:112])[-1]
sigma_cen = sigma_clipped_stats(flux_cen[:112])[-1]
sigma_ann = sigma_clipped_stats(flux_ann[:112])[-1]

#%%
#####################
## plot the figure ##
#####################
flux_cen_norm = flux_cen/np.nanmax(flux_cen)

plt.figure(figsize=(8,12))
grid=plt.GridSpec(5,1,wspace=0,hspace=0)
ax1=plt.subplot(grid[0:3])
ax1.plot(velo, flux_tot/np.nanmax(flux_tot), 'grey', label='Whole')
ax1.plot(velo, flux_cen_norm, 'k', lw=3,label='Central')
ax1.scatter(velo[93], flux_cen_norm[93]+0.01, marker='v', c='g', s=100, zorder=3)
ax1.scatter(velo[141], flux_cen_norm[141]+0.01, marker='v', c='g', s=100, zorder=3)
ax1.scatter(velo[97], flux_cen_norm[97]+0.05, marker='v', c='g', s=100, zorder=3)
ax1.scatter(velo[137], flux_cen_norm[137]+0.05, marker='v', c='g', s=100, zorder=3)

ax1.fill_between([velo[93].value, velo[97].value], [2, 2], color='b', alpha=0.5)
ax1.fill_between([velo[93].value, velo[97].value], [-1, -1], color='b', alpha=0.5)
ax1.fill_between([velo[137].value, velo[141].value], [2, 2], color='r', alpha=0.5)
ax1.fill_between([velo[137].value, velo[141].value], [-1, -1], color='r', alpha=0.5)

#ax1.plot(velo, flux_ann/np.nanmax(flux_ann), 'b', label='Annulus')

ax1.hlines(0, np.min(velo.value), np.max(velo.value),'k','--')
ax1.hlines(0.05, np.min(velo.value), np.max(velo.value),'g')
ax1.hlines(0.20, np.min(velo.value), np.max(velo.value),'g')
ax1.set_xlim(-500, 500)
ax1.set_ylim(-0.1, 1.1)
#ax1.fill_between(velo.value, -sigma, sigma, facecolor='k',hatch='/',linestyle=':',alpha=0.5)
ax1.set_xlabel('Velocity [$\mathrm{km\,s^{-1}}$]')
ax1.set_ylabel('Normalized Flux')
ax1.legend(loc = 'upper left', fontsize=15)
ax1.grid()
#plt.savefig('/home/qyfei/Desktop/Codes/Result/spectrum.pdf', bbox_inches='tight', dpi=300)

# %%
## Fitting with Gaussian profile

from astropy.modeling import models
from astropy.convolution import Gaussian1DKernel, convolve

def Gauss(x_, a_, m_, s_):
    x, a, m, s = x_, a_, m_, s_
    gauss = models.Gaussian1D(a, m, s)
    return gauss(x)
    

def log_likelihood(theta, x, y, yerr):
    a, m, s = theta
    model = Gauss(x, a, m, s)
    sigma2 = yerr ** 2# + model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2)# + np.log(sigma2))

def log_prior(para):
    a, m, s = para
    if 0<a<30 and -500<m<500 and 0<s<500:
        return 0.0
    return -np.inf

def log_probability(para, x, y, yerr):
    lp = log_prior(para)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(para, x, y, yerr)

from scipy.optimize import minimize

x = velo.value
a_g = np.max(flux_cen)
N_g = np.where(flux_cen == np.max(flux_cen))[0][0]
m_g = x[N_g]

y = flux_cen
yerr = sigma_cen

np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([a_g, m_g, 100]) + 0.1 * np.random.randn(3)
soln = minimize(nll, initial, args=(x, y, yerr))
a_ml, m_ml, s_ml = soln.x

print("Maximum likelihood estimates:")
print("a = {0:.3f}".format(a_ml))
print("m = {0:.3f}".format(m_ml))
print("s = {0:.3f}".format(s_ml))

plt.plot(x, Gauss(x, a_ml, m_ml, s_ml))
plt.plot(x, y)
# %%
# MCMC fitting

import emcee
from multiprocessing import Pool

pos = soln.x + 1e-4 * np.random.randn(32, 3)
nwalkers, ndim = pos.shape
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr), pool=pool)
    sampler.run_mcmc(pos, 5000, progress=True)

# %%
## Check fitting

fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["A", "m", "sigma"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)

import corner

fig = corner.corner(
    flat_samples, labels=labels#, truths=[]
)

# %%
## Output and save parameters

parameters = []

print("")
from IPython.display import display, Math
para_fit = np.zeros(3)
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    para_fit[i] = mcmc[1]
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))
    parameters.append([mcmc[1], q[0], q[1]])
inds = np.random.randint(len(flat_samples), size=100)

## Calculate flux
from scipy.integrate import trapz
x_test = np.linspace(x[0], x[-1], 1000)
model_flux = np.zeros(len(flat_samples))
for i in range(len(flat_samples)):
    f = Gauss(x_test, flat_samples[i][0], flat_samples[i][1], flat_samples[i][2])
    F = trapz(f, x_test)
    model_flux[i] = F

FLUX = np.percentile(model_flux, [16, 50, 84])
q = np.diff(FLUX)
txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
txt = txt.format(FLUX[1], q[0], q[1], "f")
display(Math(txt))
parameters.append([FLUX[1], q[0], q[1]])

# %%
#############################
## Plot the fitting result ##
#############################

plt.figure(figsize=(16, 8))
grid=plt.GridSpec(6,1,wspace=0,hspace=0)
ax1=plt.subplot(grid[0:5])
ax2=plt.subplot(grid[5:])
ax1.step(x, y, 'k',label='Spectrum', where='mid')
for ind in inds:
    sample = flat_samples[ind]
    ax1.plot(x_test, Gauss(x_test, sample[0], sample[1], sample[2]), "r", alpha=0.1)

ax1.plot(x_test, Gauss(x_test, para_fit[0], para_fit[1], para_fit[2]), 'r', label='Fit')
ax1.fill_between(x, -yerr, yerr, facecolor='k',hatch='/',linestyle=':',alpha=0.5, label=r'1$\sigma$ noise')
w50 = 2*np.log(2)*para_fit[2]#+np.sqrt(2*np.log(2))*para_fit[3]
ax1.vlines(para_fit[1]-w50, -100,500, 'b', ls='--', label='$W_{50}$')
ax1.vlines(para_fit[1]+w50, -100,500, 'b', ls='--')

ax1.hlines(0,-1000,1000,'k',':')
ax1.set_xlim(para_fit[1]-10*para_fit[2], para_fit[1]+10*para_fit[2])
ax1.set_ylim(-3*yerr, para_fit[0]+5*yerr)
ax1.set_ylabel('Flux density [mJy]')
ax1.legend(loc='upper left', frameon=False)

res = y - Gauss(x, para_fit[0], para_fit[1], para_fit[2])
ax2.step(x, res, 'k', where='mid')
ax2.fill_between(x, -yerr, yerr, facecolor='k',hatch='/',linestyle=':',alpha=0.5)
ax2.hlines(0,-1000,1000,'k',':')
ax2.set_xlim(para_fit[1]-10*para_fit[2], para_fit[1]+10*para_fit[2])
ax2.set_ylim(-5*yerr, 5*yerr)

ax2.set_xlabel("Velocity [km/s]")
ax2.set_ylabel("Residual [mJy]")
#plt.savefig('/home/qyfei/Desktop/Codes/Result/clouds/region%2i_spec.pdf'%n, bbox_inches='tight', dpi=300)

# %%
## chi2 evaluation
z2 = (res/yerr)**2
chi2 = np.sum(z2)/(len(z2)-3)
print(chi2)

# %%
