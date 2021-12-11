# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.modeling.projections import Projection
from astropy.units import si
from astropy.wcs import WCS
import matplotlib.colorbar as colorbar
from matplotlib.transforms import Affine2D
from astropy.visualization.wcsaxes import WCSAxes
import astropy.units as u
from spectral_cube import SpectralCube
from astropy.modeling import models, fitting
from astropy.stats import sigma_clipped_stats
import astropy.constants as c
from skimage.feature import peak_local_max

plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=15)
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
## Load data ##
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/NOEMA/w20cf/w20cf_origin/w20cf001/"
name = "F08238+0752"
file = name + "_CO32.fits"
file_mom0 = name + "_CO32_mom0.fits"


hdu = fits.open(path+file)[0]

bmaj = hdu.header['BMAJ']
bmin = hdu.header['BMIN']
delt = hdu.header['CDELT1']
pix_size = delt*u.deg.to('arcsec')
beam_area = np.pi*bmaj*bmin/delt**2/4/np.log(2)

wcs = WCS(hdu.header)
# %%
# find source, estimate spectral resolution
cube_data = hdu.data
hdr = hdu.header
CO32_data = SpectralCube.read(path+file)
velo = CO32_data.spectral_axis
restfreq = hdr['RESTFREQ']/1e9
freq = (velo/c.c).si*restfreq+restfreq

hdu = fits.open(path+file_mom0)[0]
mom0 = hdu.data[0]
coordinates = peak_local_max(mom0, min_distance=100)
print(coordinates)
print(velo[1]-velo[2])

# %%
## Plot moment 0
pos_cen = coordinates[0]
yy, xx = np.indices([hdr['NAXIS1'], hdr['NAXIS2']],dtype='float')
radius = ((yy-pos_cen[0])**2+(xx-pos_cen[1])**2)**0.5
rad = 1.0
ring = abs(rad/pix_size)

size = abs(4/pix_size)
mom0_rms = sigma_clipped_stats(mom0)[-1]
print("The rms noise of moment 0 is:", mom0_rms, "Jy/beam km/s")
mom0_level = np.array([-1,1,2,4,8,16,32])*2*mom0_rms

fig = plt.figure(figsize=(8,10))
ax = plt.subplot(projection=wcs[0])

im = ax.imshow(mom0, cmap='jet', origin='lower')
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label(name+' CO(3-2) [Jy/beam$\cdot$km/s]', fontsize=20)
ax.contour(mom0, mom0_level, colors=['k'])

circ = matplotlib.patches.Circle((pos_cen[1], pos_cen[0]), ring, fill=False, edgecolor='m')
ax.add_artist(circ)

rec_size = abs(bmaj/delt)*1.25
rec = matplotlib.patches.Rectangle((pos_cen[1]-size, pos_cen[0]-size), rec_size, rec_size, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax.add_artist(rec)
Beam = beam(hdu, pos_cen[1]-size+rec_size/2,pos_cen[0]-size+rec_size/2, 'k', pix_size)
ax.add_artist(Beam[0])
ax.set_xlim(pos_cen[1]-size, pos_cen[1]+size)
ax.set_ylim(pos_cen[0]-size, pos_cen[0]+size)
ax.set_xlabel("R.A. (J2000)", labelpad=0.5, fontsize=20)
ax.set_ylabel("Dec (J2000)", labelpad=-1.0, fontsize=20)

ax.scatter(512, 512, marker="+", color='w', s=200, zorder=3)
ax.scatter(pos_cen[1], pos_cen[0], marker='*', color='k', zorder=3)

plt.savefig("/home/qyfei/Desktop/Codes/Result/NOEMA_detection/origin/"+name+"_mom0.pdf", bbox_inches="tight", dpi=300)

# %%
## Extract the spectrum from datacube
## build the mask and extract the spectrum
mask = (radius<=ring) & (mom0>=2*mom0_rms)
hdu = fits.open(path+file)[0]
cube_data = hdu.data
hdr = hdu.header

# calculate the total flux within aperture
spectrum = cube_data*mask
flux = np.nansum(spectrum, axis=(1,2))/beam_area*u.Unit('Jy').to('mJy')

# Peak region flux
#spectrum = cube_data[:, pos_cen[0], pos_cen[1]]#*mask
#flux = spectrum*u.Unit('Jy').to('mJy')

sigma = sigma_clipped_stats(flux[112:350])[-1]

#%%
## plot the figure of spectra
#flux_cen_norm = flux_cen/np.nanmax(flux_cen)

plt.figure(figsize=(8,12))
grid=plt.GridSpec(5,1,wspace=0,hspace=0)
ax1=plt.subplot(grid[0:3])
ax1.step(freq, flux, 'k', label='Whole')
ax1.fill_between(freq, -sigma, sigma, facecolor='none',hatch='/',linestyle=':',alpha=0.5)
ax1.set_xlabel('Frequency [$\mathrm{GHz}$]')
ax1.set_ylabel('Flux [$\mathrm{mJy}$]')
ax1.legend(loc = 'upper left', fontsize=15)
ax1.grid()
ax1.set_xlim(freq.min(), freq.max())
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
    if 0<a<100 and -500<m<1000 and 0<s<800:
        return 0.0
    return -np.inf

def log_probability(para, x, y, yerr):
    lp = log_prior(para)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(para, x, y, yerr)

from scipy.optimize import minimize

x = freq.value

a_g = np.max(flux)
N = peak_local_max(flux)[0][0]
a_g = flux[N]
m_g = x[N]
y = flux
yerr = sigma

np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([a_g, m_g, 0.1]) + 0.1 * np.random.randn(3)
soln = minimize(nll, initial, args=(x, y, yerr))
a_ml, m_ml, s_ml = soln.x

print("Maximum likelihood estimates:")
print("a = {0:.3f}".format(a_ml))
print("m = {0:.3f}".format(m_ml))
print("s = {0:.3f}".format(s_ml))

plt.plot(x, Gauss(x, a_ml, m_ml, s_ml), 'grey')
plt.plot(x, y, 'red')
plt.xlim(m_ml-3*np.sqrt(8*np.log(2))*s_ml, m_ml+3*np.sqrt(8*np.log(2))*s_ml)

# %%
# MCMC fitting

import emcee
from multiprocessing import Pool

pos = [a_ml, m_ml, s_ml] + 1e-4 * np.random.randn(32, 3)
nwalkers, ndim = pos.shape
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr), pool=pool)
    sampler.run_mcmc(pos, 5000, progress=True)

# %%
## Check fitting

#fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["A", "m", "sigma"]
'''for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)'''

#axes[-1].set_xlabel("step number")

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)
'''
import corner
fig = corner.corner(
    flat_samples, labels=labels#, truths=[]
)'''

# %%
## Output and save parameters

parameters = []

print("Fitting result:")
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
x_test = np.linspace(freq.value[0], freq.value[-1], 1000)
redshifts = np.zeros(len(flat_samples))
for i in range(len(flat_samples)):
    redshift = 345.80/flat_samples[i][1]-1
    redshifts[i] = redshift

print("Estimate redshift:")
Z = np.percentile(redshifts, [16, 50, 84])
zs = np.diff(Z)
txt = "\mathrm{{{3}}} = {0:.5f}_{{-{1:.5f}}}^{{{2:.5f}}}"
txt = txt.format(Z[1], zs[0], zs[1], "z")
display(Math(txt))
parameters.append([Z[1], zs[0], zs[1]])


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

ax2.set_xlabel("Freq [GHz]")
ax2.set_ylabel("Residual [mJy]")
#plt.savefig("/home/qyfei/Desktop/Codes/Result/NOEMA_detection/"+name+"spectrum_freq.pdf", bbox_inches='tight', dpi=300)

# %%
## chi2 evaluation
z2 = (res/yerr)**2
chi2 = np.sum(z2)/(len(z2)-3)
print(chi2)

# %%
ckms = (c.c.to('km/s')).value
velo = (freq.value/345.8*(1+Z[1]) - 1)*ckms

x = velo
N = peak_local_max(flux)[0][0]
a_g = flux[N]#np.max(flux)
#N_g = np.where(flux == np.max(flux))[0][0]
m_g = x[N]
y = flux
yerr = sigma

np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([a_g, 0, 100]) + 0.1 * np.random.randn(3)
soln = minimize(nll, initial, args=(x, y, yerr))
a_ml, m_ml, s_ml = soln.x

print("Maximum likelihood estimates:")
print("a = {0:.3f}".format(a_ml))
print("m = {0:.3f}".format(m_ml))
print("s = {0:.3f}".format(s_ml))

plt.plot(x, Gauss(x, a_ml, m_ml, s_ml), 'grey')
plt.plot(x, y, 'red')
plt.xlim(m_ml-3*np.sqrt(8*np.log(2))*s_ml, m_ml+3*np.sqrt(8*np.log(2))*s_ml)

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

samples = sampler.get_chain()

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)

parameters = []
print("Fitting result:")
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

model_flux = np.zeros(len(flat_samples))
x_test = np.linspace(velo[0], velo[-1], 1000)
redshifts = np.zeros(len(flat_samples))
for i in range(len(flat_samples)):
    f = Gauss(x_test, flat_samples[i][0], flat_samples[i][1], flat_samples[i][2])
    F = trapz(f, x_test)
    model_flux[i] = F

print("Estimate flux:")
FLUX = np.percentile(model_flux, [16, 50, 84])
q = np.diff(FLUX)
txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
txt = txt.format(FLUX[1], q[0], q[1], "f")
display(Math(txt))
parameters.append([FLUX[1], q[0], q[1]])
parameters.append([Z[1], zs[0], zs[1]])

np.savetxt('/home/qyfei/Desktop/Codes/Result/NOEMA_detection/origin/'+name+'_properties.txt', np.array(parameters))

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
ax1.text(para_fit[1]+5*para_fit[2], para_fit[0], "z=%.3f"%Z[1])#ax1.text(para_fit[1]+5*para_fit[2], para_fit[0]*2/3, "f=%.3f"%abs(FLUX[1]))

res = y - Gauss(x, para_fit[0], para_fit[1], para_fit[2])
ax2.step(x, res, 'k', where='mid')
ax2.fill_between(x, -yerr, yerr, facecolor='k',hatch='/',linestyle=':',alpha=0.5)
ax2.hlines(0,-1000,1000,'k',':')
ax2.set_xlim(para_fit[1]-10*para_fit[2], para_fit[1]+10*para_fit[2])
ax2.set_ylim(-5*yerr, 5*yerr)
ax2.set_xlabel("Velocity [$\mathrm{km\,s^{-1}}$]")
ax2.set_ylabel("Residual [mJy]")

fer = "_aper"
#fer = "_peak"

plt.savefig("/home/qyfei/Desktop/Codes/Result/NOEMA_detection/origin/"+name+"_spectrum_velo"+fer+".pdf", bbox_inches='tight', dpi=300)

# %%
## Test result

from astropy.cosmology import Planck15
log_LIR = np.array([12.41, 12.46, 12.66, 12.39, 12.70])
log_LCO32 = (log_LIR - 3.28)/0.92
L_IR = np.power(10, log_LIR)

z = np.array([0.311, 0.331, 0.439, 0.326, 0.421])
DL = Planck15.luminosity_distance(z).to('Mpc')
nuobs = 345.80/(1+z)
LCO32 = np.power(10, log_LCO32)
S_CO32 = LCO32/(3.25e7*DL**2/(1+z)**3/nuobs**2)
print(S_CO32)

# %%
S_CO32_detect = np.array([2.480, 5.172, 0.477, 2.156, 3.126])
e1S_CO32_detect = np.array([0.256, 0.209, 0.088, 0.467, 0.165])
e2S_CO32_detect = np.array([0.246, 0.206, 0.087, 0.459, 0.163])

L_CO32 = 3.25e7*S_CO32_detect*DL**2/((1+z)**3*nuobs**2)

names = np.array(["F08238+0752", "F08542+1920", "F11557+1342", "F13403-0038", "F14167+4247"])


# %%
## Estimate the luminosity of CO(3-2) of 5 galaxies

L_CO32_detection = 3.25e7*S_CO32_detect*DL**2/((1+z)**3*nuobs**2)
e1_L_CO32_detection = 3.25e7*e1S_CO32_detect*DL**2/((1+z)**3*nuobs**2)
e2_L_CO32_detection = 3.25e7*e2S_CO32_detect*DL**2/((1+z)**3*nuobs**2)

log_LCO32_detection = np.log10(L_CO32_detection.value)
e1log_LCO32_detection = log_LCO32_detection - np.log10(L_CO32_detection.value - e1_L_CO32_detection.value)
e2log_LCO32_detection = np.log10(L_CO32_detection.value + e2_L_CO32_detection.value) - log_LCO32_detection

# %% 
## Comparison between prediction and measurement

plt.figure(figsize=(8, 8))
plt.errorbar(log_LCO32, log_LCO32_detection, yerr=[e1log_LCO32_detection, e2log_LCO32_detection], fmt='bo', mfc='none', ms=10, capsize=5)
plt.scatter(log_LCO32, log_LCO32_detection, s=50, c='b', zorder=3)

plt.plot([0, 15], [0, 15], 'k--')
plt.xlim(9.5, 10.5)
plt.ylim(9.5, 10.5)

plt.xlabel('Predicted log $L_\mathrm{CO(3-2)}$ [$\mathrm{K\,km\,s^{-1}}$]')
plt.ylabel('Measured log $L_\mathrm{CO(3-2)}$ [$\mathrm{K\,km\,s^{-1}}$]')

# %%
## Comparison CO(3-2) flux between aperture and peak

taper_flux_aper = np.array([3.319, 6.424, 0.300, 3.369, 3.150])
taper_eflux_aper = np.array([[0.331, 0.307, 0.060, 0.589, 0.186], [0.298, 0.292, 0.060, 0.560, 0.180]])

taper_flux_peak = np.array([1.348, 5.287, 0.551, 1.982, 2.935])
taper_eflux_peak = np.array([[0.148, 0.206, 0.105, 0.491, 0.130], [0.139, 0.205, 0.106, 0.462, 0.128]])

flux_aper = np.array([2.480, 5.172, 0.477, 2.156, 3.126])
eflux_aper = np.array([[0.256, 0.209, 0.088, 0.467, 0.165], [0.246, 0.206, 0.087, 0.459, 0.163]])

flux_peak = np.array([1.435, 3.559, 0.585, 1.396, 2.745])
eflux_peak = np.array([[0.173, 0.132, 0.121, 0.322, 0.110], [0.161, 0.130, 0.121, 0.299, 0.108]])

# %%
plt.figure(figsize=(8, 8))
plt.errorbar(taper_flux_peak, taper_flux_aper, xerr=[taper_eflux_peak[1], taper_eflux_peak[0]], yerr=[taper_eflux_aper[1], taper_eflux_aper[0]], fmt='bo', mfc='none', ms=10, capsize=5, zorder=3)
plt.scatter(taper_flux_peak, taper_flux_aper, color='b', marker='o', s=100, zorder=3)
plt.plot([0,50], [0,50], 'k--')
plt.xlim(0.1, 7)
plt.ylim(0.1, 7)
for i in range(len(flux_peak)):
    plt.text(taper_flux_peak[i]+0.1, taper_flux_aper[i], names[i], fontsize=20)
plt.grid()
plt.xlabel('Taper Peak Flux [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]')
plt.ylabel('Taper Aperture Flux [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]')

#plt.savefig('/home/qyfei/Desktop/Codes/Result/NOEMA_detection/taper/flux_comparison.pdf', bbox_inches='tight', dpi=300)

# %%
## Compare taper result and default result

plt.figure(figsize=(8, 8))
plt.errorbar(taper_flux_aper, flux_aper, xerr=[taper_eflux_peak[1], taper_eflux_peak[0]], yerr=[eflux_aper[1], eflux_aper[0]], fmt='bo', mfc='none', ms=10, capsize=5, zorder=3)
plt.scatter(taper_flux_aper, flux_aper, color='b', marker='o', s=100, zorder=3)
plt.plot([0,50], [0,50], 'k--')
plt.xlim(0.1, 7)
plt.ylim(0.1, 7)
for i in range(len(flux_peak)):
    plt.text(taper_flux_aper[i]+0.1, flux_aper[i], names[i], fontsize=20)
plt.grid()
plt.xlabel('Taper Aper. Flux [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]')
plt.ylabel('Aper. Flux [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]')

plt.savefig('/home/qyfei/Desktop/Codes/Result/NOEMA_detection/flux_comparison_taper.pdf')

# %%
