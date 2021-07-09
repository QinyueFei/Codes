# %%
## Load module

import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import JSON
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colorbar as colorbar
from matplotlib.transforms import Affine2D
from astropy.visualization.wcsaxes import WCSAxes
from spectral_cube import SpectralCube
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from matplotlib.colors import LogNorm
import matplotlib

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

# %%
###############
## Load data ##
###############
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/working/"
file = "PG0050_CO21.combine.all.line.10km.mom0.fits"

hdu = fits.open(path+file)[0]
mom0 = hdu.data[0][0]
rms = sigma_clipped_stats(mom0)[-1]
mom0_level = np.array([-1,1,2,4,8,16,32,64])*2*rms

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
## Mark regions in mom0 map
pix_size = 0.05
size = 200
ring = 0.36 #units arcsec
rad = ring/2/pix_size #units pix


region_x = np.array([-124, -135, -132, -134, -136, -130, -124, -115, -108, -100, -78, -68, -37, -23, -10, -2, 15])
region_y = np.array([-82, -60, -38, -21, -9, 2, 15, 30, 40, 50, 68, 77, 89, 92, 94, 95, 95])

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
rec = matplotlib.patches.Rectangle((pos_cen[0]-size, pos_cen[1]-size), 10, 10, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax.add_artist(rec)
Beam = beam(hdu, pos_cen[0]-size+5, pos_cen[1]-size+5, 'k', pix_size)
ax.add_artist(Beam[0])
## Left part region
ax.set_xlim(pos_cen[0]-size*3/4, pos_cen[0]+size*1/4)
ax.set_ylim(pos_cen[1]-size/2, pos_cen[1]+size/2)

## Right part region
#ax.set_xlim(pos_cen[0]-size, pos_cen[0]+size)
#ax.set_ylim(pos_cen[1]-size, pos_cen[1]+size)


for i in range(len(region_x)):
    circ = matplotlib.patches.Circle((pos_cen[0]+region_x[i], pos_cen[1]+region_y[i]), rad, fill=False, edgecolor='C1', zorder=2, lw=1., linestyle='--')
    ax.add_artist(circ)
    ax.text(pos_cen[0]+region_x[i], pos_cen[1]+region_y[i], "%i"%(i+1), c="r", fontsize=12, zorder=3)

plt.savefig('/home/qyfei/Desktop/Codes/Result/clouds/region_clouds.pdf', bbox_inches='tight')

# %%
## Extract spectrum in each region

flux = []
sigma = []

yy,xx = np.indices([600, 597],dtype='float')

for i in range(len(region_x)):
    xpos, ypos = 299.37+region_x[i], 299.28+region_y[i] #center of region where extracting the spectrum
    ring = 0.36 #units arcsec
    rad = ring/2/0.05 #units pix
    radius = ((yy-ypos)**2+(xx-xpos)**2)**0.5
    mask = radius<=rad
    spectrum = CO21_cube*mask
    ##############
    ## Calculate the flux
    ##############
    flux.append(np.nansum(spectrum,axis=(1,2))/beam_area*1e3*u.Unit('mJy'))

    sigma.append(sigma_clipped_stats(list(flux[i][0:90].value)+list(flux[i][144:].value))[-1])
    #sigma_p = sigma
    #sigma_m = -sigma

# %%
# Fit the spectra with Gaussian profile
# minimization algorithm
# Define models
from astropy.modeling import models
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
    if 0<a<30 and -500<m<500 and 0<s<80:
        return 0.0
    return -np.inf

def log_probability(para, x, y, yerr):
    lp = log_prior(para)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(para, x, y, yerr)
# %%
## Fitting preparation
import emcee
from multiprocessing import Pool
from IPython.display import display, Math

para_A = []
para_m = []
para_s = []
para_f = []
chi2_tot = []
para = [para_A, para_m, para_s, para_f]

# %%
## Fitting
paths = "/home/qyfei/Desktop/Codes/Result/clouds/paras/"
for i in range(len(region_x)):
    print("Fit region %i spectrum:\n"%(i+1))

    f = flux[i]
    N = np.where(f == np.max(f))[0]
    m_g = velo[N[0]].value
    a_g = f[N[0]].value

    f_err = sigma[i]
    pos = np.array([a_g, m_g, 20]) + 1e-4 * np.random.randn(32, 3)
    nwalkers, ndim = pos.shape
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(velo.value, f.value, f_err), pool=pool)
        sampler.run_mcmc(pos, 5000, progress=True)
    samples = sampler.get_chain()
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

    para_fit = np.zeros(3)
    #labels = ["A", "m", "s"]
    for j in range(ndim):
        mcmc = np.percentile(flat_samples[:, j], [16, 50, 84])
        para_fit[j] = mcmc[1]
        q = np.diff(mcmc)
    #    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    #    txt = txt.format(mcmc[1], q[0], q[1], labels[j])
    #    display(Math(txt))
        para[j].append([mcmc[1], q[0], q[1]])

    from scipy.integrate import trapz
    x_test = np.linspace(velo.value[0], velo.value[-1], 1000)
    flux_model = np.zeros(len(flat_samples))
    for j in range(len(flat_samples)):
        f_model = Gauss(x_test, flat_samples[j][0], flat_samples[j][1], flat_samples[j][2])
        F_model = trapz(f_model, x_test)
        flux_model[j] = F_model

    FLUX = np.percentile(flux_model, [16, 50, 84])
    q = np.diff(FLUX)
    #txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    #txt = txt.format(FLUX[1], q[0], q[1], "flux")
    #display(Math(txt))
    para[3].append([FLUX[1], q[0], q[1]])

    inds = np.random.randint(len(flat_samples), size=100)

    plt.figure(figsize=(16, 8))
    grid=plt.GridSpec(6,1,wspace=0,hspace=0)
    ax1=plt.subplot(grid[0:5])
    ax2=plt.subplot(grid[5:])
    ax1.step(velo, f,'k',label='Spectrum', where='mid')
    for ind in inds:
        sample = flat_samples[ind]
        ax1.plot(x_test, Gauss(x_test, sample[0], sample[1], sample[2]), "r", alpha=0.1)
    ax1.plot(x_test, Gauss(x_test, para_fit[0], para_fit[1], para_fit[2]), 'r', label='Fit')
    ax1.fill_between(velo, -f_err, f_err, facecolor='k',hatch='/',linestyle=':',alpha=0.5, label=r'1$\sigma$ noise')
    w50 = 2*np.log(2)*para_fit[2]
    ax1.vlines(para_fit[1]-w50, -100,500, 'b', ls='--', label='$W_{50}$')
    ax1.vlines(para_fit[1]+w50, -100,500, 'b', ls='--')

    ax1.hlines(0,-1000,1000,'k',':')
    ax1.set_xlim(para_fit[1]-10*para_fit[2], para_fit[1]+10*para_fit[2])
    ax1.set_ylim(-3*f_err, para_fit[0]+5*f_err)
    ax1.set_ylabel('Flux density [mJy]')
    ax1.legend(loc='upper left', frameon=False)

    res = f.value - Gauss(velo.value, para_fit[0], para_fit[1], para_fit[2])
    ax2.step(velo, res, 'k', where='mid')
    ax2.fill_between(velo, -f_err, f_err, facecolor='k',hatch='/',linestyle=':',alpha=0.5)
    ax2.hlines(0,-1000,1000,'k',':')
    ax2.set_xlim(para_fit[1]-10*para_fit[2], para_fit[1]+10*para_fit[2])
    ax2.set_ylim(-5*f_err, 5*f_err)

    ax2.set_xlabel("Velocity [km/s]")
    ax2.set_ylabel("Residual [mJy]")
    plt.savefig(paths+'spec_region%2i.pdf'%(i+1), bbox_inches='tight')

    ## chi2 evaluation
    z2 = (res/f_err)**2
    chi2 = np.sum(z2)/(len(z2)-3)
    print(chi2)
    chi2_tot.append(chi2)

    np.savetxt(paths+"para_A.txt", np.array(para_A))
    np.savetxt(paths+"para_m.txt", np.array(para_m))
    np.savetxt(paths+"para_s.txt", np.array(para_s))
    np.savetxt(paths+"para_f.txt", np.array(para_f))
    np.savetxt(paths+"chi2.txt", np.array(chi2_tot))
# %%
## Output parameters 

para_A = np.loadtxt(paths+"para_A.txt")
para_f = np.loadtxt(paths+"para_f.txt")
para_m = np.loadtxt(paths+"para_m.txt")
para_s = np.loadtxt(paths+"para_s.txt")
chi2_fin = np.loadtxt(paths+"chi2.txt")

flux_fit = para_f[:,0]
flux_fit_emin = para_f[:,1]
flux_fit_emaj = para_f[:,2]

disp_fit = para_s[:,0]
disp_fit_emin = para_s[:,1]
disp_fit_emaj = para_s[:,2]

ampl_fit = para_A[:,0]
ampl_fit_emin = para_A[:,1]
ampl_fit_emaj = para_A[:,2]

# %%
## try to fit

def log_likelihood(theta, x, y, yerr):
    m, b = theta
    model = m * x + b
    sigma2 = yerr ** 2 
    return -0.5 * np.sum((y - model) ** 2 / sigma2)

x = np.delete(np.log10(flux_fit), 9)
y = np.delete(np.log10(disp_fit), 9)
xerr = np.delete(np.log10(flux_fit_emaj), 9)
yerr = np.delete(np.log10(disp_fit_emaj), 9)

from scipy.optimize import minimize

np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([0.4, 0.5]) + 0.1 * np.random.randn(2)
soln = minimize(nll, initial, args=(x, y, yerr))
m_ml, b_ml = soln.x

print("Maximum likelihood estimates:")
print("m = {0:.3f}".format(m_ml))
print("b = {0:.3f}".format(b_ml))

# %%
def log_prior(theta):
    m, b = theta
    if -10.0 < m < 10.0 and -10.0 < b < 10.0:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

import emcee

pos = soln.x + 1e-4 * np.random.randn(32, 2)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
sampler.run_mcmc(pos, 5000, progress=True)

# %%

fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["m", "b"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

# %%
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

from IPython.display import display, Math

for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))

# %%
## Larson's law?

plt.figure(figsize=(8,8))
ax = plt.subplot(111)
im = ax.scatter(flux_fit, disp_fit, c=chi2_fin, cmap='jet', zorder=3, s=100)
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label(r'$\chi^2$')
ax.errorbar(flux_fit, disp_fit, xerr = [flux_fit_emin, flux_fit_emaj], yerr = [disp_fit_emin, disp_fit_emaj], fmt = 'ko', mfc = 'none', ms = 10, capsize = 0)

for i in range(len(flux_fit)):
    ax.text(flux_fit[i]+1, disp_fit[i]+0.1, '%i'%(i+1), color='r', fontsize=15)

ax.set_xlabel("Flux [mJy$\cdot$km/s]")
ax.set_ylabel("$\sigma$ [km/s]")
ax.loglog()
ax.set_xlim(27, 120)
ax.set_ylim(7, 32)
ax.plot(x0, np.power(10, b_ml)*np.power(x0, m_ml), "k--")
ax.text(30, 20, "r=0.63$\pm$0.01")

inds = np.random.randint(len(flat_samples), size=100)
for ind in inds:
    sample = flat_samples[ind]
    ax.plot(x0, np.power(10, sample[1])*np.power(x0, sample[0]), "C1", alpha=0.1)

plt.savefig(paths+"fvss.pdf", bbox_inches="tight")
# %%

from scipy.stats import pearsonr
pearsonr(flux_fit, disp_fit)

# %%
x0 = np.linspace(0, 1000, 10000)
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x0, np.dot(np.vander(x0, 2), [m_ml, b_ml]), ":k", label="ML")
plt.legend(fontsize=14)
plt.xlim(1, 2.1)
plt.ylim(0, 2)
plt.xlabel("x")
plt.ylabel("y")
# %%
m_ml, b_ml
# %%
