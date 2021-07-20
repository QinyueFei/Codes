# %%
## Load module

## This file introduce extracting spectrum from a datacube individually.

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colorbar as colorbar
from matplotlib.transforms import Affine2D
from astropy.visualization.wcsaxes import WCSAxes
from spectral_cube import SpectralCube
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from matplotlib.colors import LogNorm

plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

def beam(HDU, XPOS, YPOS, col, cellsize):
    hdu = HDU
    xpos, ypos = XPOS, YPOS
    c = col
    cell = cellsize
    bmaj, bmin, bpa = hdu.header['BMAJ'], hdu.header['BMIN'], hdu.header['BPA']
    Bmaj = bmaj*u.Unit('deg').to('arcsec')/cell
    Bmin = bmin*u.Unit('deg').to('arcsec')/cell
    Beam = matplotlib.patches.Ellipse((xpos,ypos), Bmaj, Bmin, 90+bpa, 
                                      edgecolor='k', facecolor='gray', fill=True, zorder=3)    
    return Beam, Bmaj, Bmin

###############
## Load data ##
###############
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/working/"
file = "PG0050_CO21.combine.all.line.10km.mom0.fits"

hdu = fits.open(path+file)[0]
mom0 = hdu.data[0][0]
rms = sigma_clipped_stats(mom0)[-1]

pos_cen = [299.37, 299.28]
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/Barolo_fit/"
file_mom0 = "PG0050_CO21.combine.all.line.10km.sub.4sigma.mom0.fits"
hdu = fits.open(path+file_mom0)[0]
mom0_rms = hdu.data[0][0]

file_mom1 = "PG0050_CO21.combine.all.line.10km.sub.mom1.fits"
hdu = fits.open(path+file_mom1)[0]
mom1 = hdu.data[0][0]

file_mom2 = "PG0050_CO21.combine.all.line.10km.sub.mom2.fits"
hdu = fits.open(path+file_mom2)[0]
mom2 = hdu.data[0][0]
# %%
# load datacube and calculate the spectrum in km/s
name = '/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/Barolo_fit/PG0050_CO21.combine.all.line.10km.sub.fits'
CO21_cube = SpectralCube.read(name)
masked_CO21_cube = CO21_cube.with_spectral_unit(unit='km/s', rest_value=217.253*u.GHz, velocity_convention='radio') #just for velocity
# calculate the x-axis
velo = masked_CO21_cube.spectral_axis

#####################
## calculate the beam area
#####################
bmaj = hdu.header['BMAJ']
bmin = hdu.header['BMIN']
delt = hdu.header['CDELT1']
beam_area = np.pi*bmaj*bmin/delt**2/4/np.log(2)

hdu = fits.open(name)[0]
CO21_cube = hdu.data[0]

# %%
## Choose the region
# set the coordinates of the region where spectrum is extracted
yy,xx = np.indices([600, 597],dtype='float')

#n = 0 #region number
#off_set = np.array([60, -80])


xpos, ypos = 299.37+off_set[0], 299.28+off_set[1] #center of region where extracting the spectrum
ring = 0.36 #units arcsec
rad = ring/2/0.05 #units pix
radius = ((yy-ypos)**2+(xx-xpos)**2)**0.5
mask = radius<=rad
spectrum = CO21_cube*mask

mom2_level = np.linspace(0, 100, 11)
##############
## Calculate the flux
##############
flux = np.nansum(spectrum,axis=(1,2))/beam_area*1e3*u.Unit('mJy')

sigma = sigma_clipped_stats(list(flux[0:90].value)+list(flux[144:].value))[-1]
sigma_p = sigma
sigma_m = -sigma

import matplotlib
size = 200
pix_size = 0.05
mom0_level = np.array([-1,1,2,4,8,16,32])*2*rms

transform = Affine2D()
transform.scale(pix_size, pix_size)
transform.translate(-pos_cen[0]*pix_size, -pos_cen[1]*pix_size)
#transform.translate(-100*pix_size, -100*pix_size)
transform.rotate(0.)  # radians

# Set up metadata dictionary
coord_meta = {}
coord_meta['name'] = 'RA (J2000)', 'DEC (J2000)'
coord_meta['type'] = 'longitude', 'latitude'
coord_meta['wrap'] = 180, None
coord_meta['unit'] = u.arcsec, u.arcsec
coord_meta['format_unit'] = None, None

fig = plt.figure(figsize=(8,10))

ax = WCSAxes(fig, [0.1,0.1,0.8,0.8], aspect='equal',
             transform=transform, coord_meta=coord_meta)
fig.add_axes(ax)
im = ax.imshow(mom0_rms, cmap='jet', origin='lower', norm=LogNorm())
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label(r'CO(2-1) [Jy/beam$\cdot$km/s]')
ax.contour(mom0_rms, mom0_level/3, colors=['k'])

#######################
## annotate the beam ##
#######################
rec = matplotlib.patches.Rectangle((pos_cen[0]-size*3/4, pos_cen[1]-size/2), 10, 10, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax.add_artist(rec)
Beam = beam(hdu, pos_cen[0]-size*3/4+5, pos_cen[1]-size/2+5, 'k', pix_size)
ax.add_artist(Beam[0])
## Left part region
#ax.set_xlim(pos_cen[0]-size*3/4, pos_cen[0]+size*1/4)
#ax.set_ylim(pos_cen[1]-size/2, pos_cen[1]+size/2)

## Right part region
ax.set_xlim(pos_cen[0]-size, pos_cen[0]+size)
ax.set_ylim(pos_cen[1]-size, pos_cen[1]+size)

circ = matplotlib.patches.Circle((xpos, ypos), rad, fill=False, edgecolor='C1', zorder=2, lw=3, linestyle='--')
ax.add_artist(circ)
#plt.savefig('/home/qyfei/Desktop/Codes/Result/clouds/region_%i.pdf'%n, bbox_inches='tight')

#######################
## plot the spectrum ##
#######################
N = np.where(flux == np.max(flux))[0]
m_g = velo[N[0]].value
a_g = flux[N[0]].value
fig = plt.figure(figsize=(8,10))
grid=plt.GridSpec(5,1,wspace=0,hspace=0)
ax1=plt.subplot(grid[0:3])
ax1.step(velo, flux,'k',label='%.2f arcsec'%ring, where='mid')
ax1.hlines(0, -1000, 1000,'k','--')
ax1.set_xlim(m_g-100, m_g+100)
ax1.fill_between(velo, sigma_p, sigma_m, facecolor='k',hatch='/',linestyle=':',alpha=0.5)
ax1.set_xlabel('Velocity [km/s]')
ax1.set_ylabel('Flux density [mJy]')
ax1.legend(loc='upper right')

# %% Fit the spectra with Gaussian profile

# minimization algorithm
from astropy.modeling import models
from astropy.convolution import Gaussian1DKernel, convolve

Kernel_spec = Gaussian1DKernel(0.85)

def Gauss(x_, a_, m_, s_):
    x, a, m, s = x_, a_, m_, s_
    gauss = models.Gaussian1D(a, m, s)
    return convolve(gauss(x), Kernel_spec)#gauss(x)
    

def log_likelihood(theta, x, y, yerr):
    a, m, s = theta
    model = Gauss(x, a, m, s)
    sigma2 = yerr ** 2# + model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2)# + np.log(sigma2))

def log_prior(para):
    a, m, s = para
    if 0<a<30 and -500<m<500 and 0<s<80:
        return 0.0
    return -np.inf

def log_probability(para, x, y, yerr):
    lp = log_prior(para)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(para, x, y, yerr)

from scipy.optimize import minimize

np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([a_g, m_g, 20]) + 0.1 * np.random.randn(3)
soln = minimize(nll, initial, args=(velo.value, flux.value, sigma))
a_ml, m_ml, s_ml = soln.x

print("Maximum likelihood estimates:")
print("a = {0:.3f}".format(a_ml))
print("m = {0:.3f}".format(m_ml))
print("s = {0:.3f}".format(s_ml))


import emcee
from multiprocessing import Pool

pos = soln.x + 1e-4 * np.random.randn(32, 3)
nwalkers, ndim = pos.shape
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(velo.value, flux.value, sigma), pool=pool)
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
x_test = np.linspace(velo.value[0], velo.value[-1], 1000)
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

#print("\n")
#print(parameters)

# %%
#############################
## Plot the fitting result ##
#############################

plt.figure(figsize=(16, 8))
grid=plt.GridSpec(6,1,wspace=0,hspace=0)
ax1=plt.subplot(grid[0:5])
ax2=plt.subplot(grid[5:])
ax1.step(velo, flux,'k',label='Spectrum', where='mid')
for ind in inds:
    sample = flat_samples[ind]
    ax1.plot(x_test, Gauss(x_test, sample[0], sample[1], sample[2]), "r", alpha=0.1)

ax1.plot(x_test, Gauss(x_test, para_fit[0], para_fit[1], para_fit[2]), 'r', label='Fit')
ax1.fill_between(velo, sigma_p, sigma_m, facecolor='k',hatch='/',linestyle=':',alpha=0.5, label=r'1$\sigma$ noise')
w50 = 2*np.log(2)*para_fit[2]#+np.sqrt(2*np.log(2))*para_fit[3]
ax1.vlines(para_fit[1]-w50, -100,500, 'b', ls='--', label='$W_{50}$')
ax1.vlines(para_fit[1]+w50, -100,500, 'b', ls='--')

ax1.hlines(0,-1000,1000,'k',':')
ax1.set_xlim(para_fit[1]-10*para_fit[2], para_fit[1]+10*para_fit[2])
ax1.set_ylim(-3*sigma, para_fit[0]+5*sigma)
ax1.set_ylabel('Flux density [mJy]')
ax1.legend(loc='upper left', frameon=False)

res = flux.value - Gauss(velo.value, para_fit[0], para_fit[1], para_fit[2])
ax2.step(velo, res, 'k', where='mid')
ax2.fill_between(velo, sigma_p, sigma_m, facecolor='k',hatch='/',linestyle=':',alpha=0.5)
ax2.hlines(0,-1000,1000,'k',':')
ax2.set_xlim(para_fit[1]-10*para_fit[2], para_fit[1]+10*para_fit[2])
ax2.set_ylim(-5*sigma, 5*sigma)

ax2.set_xlabel("Velocity [km/s]")
ax2.set_ylabel("Residual [mJy]")
#plt.savefig('/home/qyfei/Desktop/Codes/Result/clouds/region%2i_spec.pdf'%n, bbox_inches='tight', dpi=300)

## chi2 evaluation
z2 = (res/sigma)**2
chi2 = np.sum(z2)/(len(z2)-3)
print(chi2)

# %% 
# 
# Fit the spectra with double Gaussian profile with MCMC

from astropy.modeling import models
# define the profile and minimization algorithm

def Gauss_double(x_, a0_, m0_, s0_, a1_, m1_, s1_):
    x = x_
    a0, m0, s0 = a0_, m0_, s0_
    a1, m1, s1 = a1_, m1_, s1_
    gauss = models.Gaussian1D(a0, m0, s0) + models.Gaussian1D(a1, m1, s1)
    return gauss(x)

def log_likelihood(theta, x, y, yerr):
    a0, m0, s0, a1, m1, s1 = theta
    model = Gauss_double(x, a0, m0, s0, a1, m1, s1)
    sigma2 = yerr ** 2# + model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2)# + np.log(sigma2))

from scipy.optimize import minimize
x, y, yerr = velo.value, flux.value, sigma
np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([a_g, m_g, 20, a_g, m_g, 20]) + 0.1 * np.random.randn(6)
soln = minimize(nll, initial, args=(x, y, yerr), method='Nelder-Mead')
a0_ml, m0_ml, s0_ml, a1_ml, m1_ml, s1_ml = soln.x

print("Maximum likelihood estimates:")
print("a0 = {0:.3f}".format(a0_ml))
print("m0 = {0:.3f}".format(m0_ml))
print("s0 = {0:.3f}".format(s0_ml))
print("a1 = {0:.3f}".format(a1_ml))
print("m1 = {0:.3f}".format(m1_ml))
print("s1 = {0:.3f}".format(s1_ml))

def log_prior(para):
    a0, m0, s0, a1, m1, s1 = para
    if 0<a0<30 and -500<m0<500 and 0<s0<80 and 0<a1<30 and -500<m1<500 and 0<s1<80:
        return 0.0
    return -np.inf

def log_probability(para, x, y, yerr):
    lp = log_prior(para)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(para, x, y, yerr)

################
## Being mcmc ##
################
print("Being MCMC fitting:\n")
import emcee
from multiprocessing import Pool

pos = np.array([a0_ml, m0_ml, s0_ml, a1_ml, m1_ml, s1_ml]) + 1e-4 * np.random.randn(64, 6)
nwalkers, ndim = pos.shape
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(velo.value, flux.value, sigma), pool=pool)
    sampler.run_mcmc(pos, 5000, progress=True)

fig, axes = plt.subplots(6, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["A0", "m0", "sigma0", "A1", "m1", "sigma1"]
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

from IPython.display import display, Math
para_fit = np.zeros(6)
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    para_fit[i] = mcmc[1]
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))

inds = np.random.randint(len(flat_samples), size=100)

#############################
## Plot the fitting result ##
#############################
x_test = np.linspace(velo.value[0], velo.value[-1], 10000)

plt.figure(figsize=(12, 8))
grid=plt.GridSpec(6,1,wspace=0,hspace=0)
ax1=plt.subplot(grid[0:5])
ax2=plt.subplot(grid[5:])
ax1.step(velo, flux,'k',label='Spectrum', where='mid')
for ind in inds:
    sample = flat_samples[ind]
    ax1.plot(x_test, Gauss_double(x_test, sample[0], sample[1], sample[2], sample[3], sample[4], sample[5])
             , "red", alpha=0.1)
    #ax1.plot(x_test, Gauss(x_test, sample[0], sample[1], sample[2])
    #         , "yellow", alpha=0.1)
    #ax1.plot(x_test, Gauss(x_test, sample[3], sample[4], sample[5])
    #         , "green", alpha=0.1)
ax1.plot(x_test, Gauss_double(x_test, para_fit[0], para_fit[1], para_fit[2], para_fit[3], para_fit[4], para_fit[5]), 'r', lw=3, label='Fit')
ax1.fill_between(velo, sigma_p, sigma_m, facecolor='k',hatch='/',linestyle=':',alpha=0.5, label=r'1$\sigma$ noise')
w50_0 = 2*np.log(2)*para_fit[2]#+np.sqrt(2*np.log(2))*para_fit[3]
w50_1 = 2*np.log(2)*para_fit[5]
ax1.vlines(para_fit[1]-w50_0, -100,500, 'b', ls='--', label='$W_{50}$')
ax1.vlines(para_fit[1]+w50_0, -100,500, 'b', ls='--')
ax1.vlines(para_fit[4]+w50_1, -100,500, 'b', ls='--')
ax1.vlines(para_fit[4]-w50_1, -100,500, 'b', ls='--')

ax1.hlines(0,-1000,1000,'k',':')
ax1.set_xlim(para_fit[1]-10*para_fit[2], para_fit[4]+10*para_fit[5])
ax1.set_ylim(-3*sigma, a_g+5*sigma)
ax1.set_ylabel('Flux density [mJy]')
ax1.legend(loc='upper left', frameon=False)

res = flux.value - Gauss_double(velo.value, para_fit[0], para_fit[1], para_fit[2], para_fit[3], para_fit[4], para_fit[5])
ax2.step(velo, res, 'k', where='mid')
ax2.fill_between(velo, sigma_p, sigma_m, facecolor='k',hatch='/',linestyle=':',alpha=0.5)
ax2.hlines(0,-1000,1000,'k',':')
ax2.set_xlim(para_fit[1]-10*para_fit[2], para_fit[4]+10*para_fit[5])
ax2.set_ylim(-5*sigma, 5*sigma)

ax2.set_xlabel("Velocity [km/s]")
ax2.set_ylabel("Residual [mJy]")

#plt.savefig('spectrum/region%2i_spec.pdf'%n, bbox_inches='tight')

## chi2 evaluation
z2 = (res/sigma)**2
chi2 = np.sum(z2)/(len(z2)-6)
print(chi2)

# %%

region_x_L = np.array([-28, -30, -40, -43, -133, -108, -85])
region_y_L = np.array([-20, 0, -10, 5, -35, 42, 63])

region_x_R = np.array([30, 12, 8, 42, 55, 62, 193])
region_y_R = np.array([20, 45, -45, -58, -55, -48, 36])

region_x = np.append(region_x_L, region_x_R)
region_y = np.append(region_y_L, region_y_R)

transform = Affine2D()
transform.scale(pix_size, pix_size)
transform.translate(-pos_cen[0]*pix_size, -pos_cen[1]*pix_size)
#transform.translate(-100*pix_size, -100*pix_size)
transform.rotate(0.)  # radians

# Set up metadata dictionary
coord_meta = {}
coord_meta['name'] = 'RA (J2000)', 'DEC (J2000)'
coord_meta['type'] = 'longitude', 'latitude'
coord_meta['wrap'] = 180, None
coord_meta['unit'] = u.arcsec, u.arcsec
coord_meta['format_unit'] = None, None

fig = plt.figure(figsize=(8,10))

ax = WCSAxes(fig, [0.1,0.1,0.8,0.8], aspect='equal',
             transform=transform, coord_meta=coord_meta)
fig.add_axes(ax)
im = ax.imshow(mom0_rms, cmap='jet', origin='lower', norm=LogNorm())
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label(r'CO(2-1) [Jy/beam$\cdot$km/s]')
ax.contour(mom0_rms, mom0_level/3, colors=['k'])

#######################
## annotate the beam ##
#######################
rec = matplotlib.patches.Rectangle((pos_cen[0]-size, pos_cen[1]-size), 10, 10,
                                   angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax.add_artist(rec)
Beam = beam(hdu, pos_cen[0]-size+5, pos_cen[1]-size+5, 'k', pix_size)
ax.add_artist(Beam[0])
## Left part region
#ax.set_xlim(pos_cen[0]-size*3/4, pos_cen[0]+size*1/4)
#ax.set_ylim(pos_cen[1]-size/2, pos_cen[1]+size/2)

## Right part region
ax.set_xlim(pos_cen[0]-size, pos_cen[0]+size)
ax.set_ylim(pos_cen[1]-size, pos_cen[1]+size)


for i in range(len(region_x)):
    circ = matplotlib.patches.Circle((pos_cen[0]+region_x[i], pos_cen[1]+region_y[i]), rad, fill=False, edgecolor='C1', zorder=2, lw=1., linestyle='--')
    ax.add_artist(circ)
    ax.text(pos_cen[0]+region_x[i], pos_cen[1]+region_y[i], "%i"%i, c="r", fontsize=12, zorder=3)

#plt.savefig('/home/qyfei/Desktop/Codes/Analysis/spectrum/region_mom0_tot.pdf', bbox_inches='tight')

