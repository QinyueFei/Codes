# %%
## import
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
## Load data

path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/"
work = "Barolo_fit/"
name = "PG0050_CO21.combine.all.line.10km.sub.mom0.fits"

hdu = fits.open(path+work+name)[0]
mom0 = hdu.data[0][0]

bmaj = hdu.header['BMAJ']
bmin = hdu.header['BMIN']
delt = hdu.header['CDELT1']
CO_pix = np.pi*bmaj*bmin/(4*np.log(2)*delt**2)

sigma = sigma_clipped_stats(mom0)[-1]

# %%
## Plot moment 0 map
import matplotlib

N = np.where(mom0 == np.nanmax(mom0))
xpos = N[0][0]
ypos = N[1][0]
size = 100
pix_size = 0.05

mom0_level = np.array([-1,1,2,4,8,16,32])*2*sigma

transform = Affine2D()
transform.scale(pix_size, pix_size)
transform.translate(-xpos*pix_size, -ypos*pix_size)
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
im = ax.imshow(mom0, cmap='jet', origin='lower')
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label('CO(2-1) [Jy/beam km/s]')
ax.contour(mom0, mom0_level, colors=['k'])
#ax.contour(mom0_res, mom0_level, colors=['w'])
rec = matplotlib.patches.Rectangle((xpos-size, ypos-size), 10, 10,
                                   angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax.add_artist(rec)
Beam = beam(hdu, xpos-size+5,ypos-size+5, 'w', pix_size)
ax.add_artist(Beam[0])

ax.set_xlim(xpos-size,ypos+size)
ax.set_ylim(xpos-size,ypos+size)
# %%
## Build the model

from scipy.special import gammaincinv
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft

tran = 2*np.sqrt(2*np.log(2.0)) #convert FWHM to sigma
xstd = abs(hdu.header['BMAJ']/hdu.header['CDELT2'])/tran
ystd = abs(hdu.header['BMIN']/hdu.header['CDELT1'])/tran
pa = (hdu.header['BPA']+90)*u.deg #PA's definition

from scipy.signal import convolve as scipy_convolve
from astropy.convolution import convolve
kernel_CO = Gaussian2DKernel(x_stddev = xstd, y_stddev = ystd, theta = pa, mode='center')

from astropy.modeling.models import Sersic2D

def Disk2D(x, y, x0, y0, I_e, R_e, n_, ellip_, theta_):
    #y, x = np.mgrid[:y_, :x_]
    #note this function returns the model after convolving
    
    amplitude = I_e
    r_eff = R_e
    n = n_
    x_0, y_0 = x0, y0
    ellip = ellip_
    theta = theta_
    
    function = Sersic2D(amplitude, r_eff, n, x_0, y_0, ellip, theta)
    I_r = function(x, y)
    #f1 = convolve(f, kernel)
    
    Ir = scipy_convolve(I_r, kernel_CO, mode='same', method='fft')
    
    return Ir

from astropy.modeling.models import Gaussian2D
def Gauss2D(x, y, x0_, y0_, I0_, xstd_, ystd_, phi_):
    # define the gaussian profile after convolving
    x0, y0 = x0_, y0_
    I0, x_std, y_std, phi = I0_, xstd_, ystd_, phi_
    function = Gaussian2D(I0, x0, y0, x_std, y_std, phi)
    gauss = function(x, y)
    Ig = scipy_convolve(gauss, kernel_CO, mode='same', method='fft')
    return Ig

# %%
## Fit the observation with maximum likelihood algorithm

size = 100
f_mom0 = mom0[xpos-size:xpos+size, ypos-size:ypos+size]# * mask
#f_mom0_mask = f_mom0 * mask
f_err = sigma

def log_likelihood(para, x, y, z, zerr):
    x0, y0, Ie, Re, n, ellip, theta, x0_g, y0_g, I0, xstd0, ystd0, phi = para
    
    model0 = Disk2D(x, y, x0, y0, Ie, Re, n, ellip, theta)
    model1 = Gauss2D(x, y, x0_g, y0_g, I0, xstd0, ystd0, phi)
    model = model0 + model1
    sigma2 = zerr ** 2# + model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((z - model) ** 2 / sigma2 + np.log(sigma2))

x, y = np.mgrid[:2*size, :2*size]
x0, y0, Ie, Re, n, ellip, theta = size, size, 0.5, 25, 0.4, 0.15, 0.5
x0_g, y0_g, I0, xstd, ystd, phi = size, size, 27, 1.2, 4.2, 4.2

print("Begin Maximum likelyhood fitting:")
np.random.seed(42)
from scipy.optimize import minimize
nll = lambda *args: -log_likelihood(*args)
initial = np.array([x0, y0, Ie, Re, n, ellip, theta, x0_g, y0_g, I0, xstd, ystd, phi]) + 0.1 * np.random.randn(13)
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

level = np.array([-1,1,2,4,8,16])*3*f_err
f_ml0 = Disk2D(x, y, x0_ml, y0_ml, Ie_ml, Re_ml, n_ml, ellip_ml, theta_ml)
f_ml1 = Gauss2D(x, y, x0_g_ml, y0_g_ml, I0_ml, xstd_ml, ystd_ml, phi_ml)
f_ml = f_ml0 + f_ml1
f_res = f_mom0 - f_ml

# %%
## Plot the fitting result

plt.figure(figsize=(15,5))
ax0 = plt.subplot(131)
ax0.imshow(f_mom0, cmap='jet', origin='lower')
ax0.contour(f_mom0, mom0_level, colors=['k'])
ax1 = plt.subplot(132)
ax1.imshow(f_ml, cmap='jet', origin='lower')
ax1.contour(f_ml, mom0_level, colors=['k'])
ax2 = plt.subplot(133)
ax2.imshow(f_res, cmap='jet', origin='lower')
ax2.contour(f_res, mom0_level, colors=['k'])
#plt.savefig('CO_fit_ml.pdf', bbox_inches='tight', dpi=300)

# %%
## MCMC fitting process

def log_prior(para):
    x0, y0, Ie, Re, n, ellip, theta = para[:7]
    x0_g, y0_g, I0, xstd, ystd, phi = para[7:]
    if 80<x0<120 and 80<y0<120 and 1e-5<Ie<20. and 0.<Re<100. and 0.0<n<10.0 and 0<theta<2*np.pi and 80<x0_g<120 and 80<y0_g<120 and 1e-5<Ie<50. and 0.<xstd<=30. and 0.0<ystd<=30.0 and 0<phi<=2*np.pi:
        return 0.0
    return -np.inf

def log_probability(para, x, y, f, ferr):
    lp = log_prior(para)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(para, x, y, f, ferr)
print("\n")

print("Begin mcmc fitting:")
from multiprocessing import Pool
import emcee
pos = soln.x + 1e-4 * np.random.randn(50, 13)
nwalkers, ndim = pos.shape
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, f_mom0, f_err), pool=pool)
    sampler.run_mcmc(pos, 5000, progress=True)
    

# %%
## Check fitting result

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
#plt.savefig('Result/corner.png', bbox_inches='tight', dpi=300)

# %%
## Output parameters

from IPython.display import display, Math

for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))

# %%
## Output fitting model

f_outer = Disk2D(x, y, para_out[0], para_out[1], para_out[2], para_out[3], para_out[4], para_out[5], para_out[6])
f_inner = Gauss2D(x, y, para_out[7], para_out[8], para_out[9], para_out[10], para_out[11], para_out[12])
f_model = f_outer + f_inner
f_total_res = f_mom0 - f_model
print("\n")
print("Save the result:")

# %%
## Print the final fitting results
plt.figure(figsize=(15,6))
plt.rcParams['xtick.major.bottom'] = False
plt.rcParams['ytick.major.left'] = False
ax0 = plt.subplot(131)
im0 = ax0.imshow(f_mom0, cmap='jet', origin='lower')
ax0.contour(f_mom0, mom0_level, colors=['k'])
cp0,kw0 = colorbar.make_axes(ax0, pad=0.003, aspect=18, location='top')
cb0 = plt.colorbar(im0, cax=cp0, orientation='horizontal', ticklocation='top')
cb0.ax.set_xlabel(r'OBSERVATION')
ax1 = plt.subplot(132)
im1 = ax1.imshow(f_model, cmap='jet', origin='lower')
ax1.contour(f_model, mom0_level, colors=['k'])
cp1,kw1 = colorbar.make_axes(ax1, pad=0.003, aspect=18, location='top')
cb1 = plt.colorbar(im1, cax=cp1, orientation='horizontal', ticklocation='top')
cb1.ax.set_xlabel(r'MODEL')
ax2 = plt.subplot(133)
im2 = ax2.imshow(f_total_res, cmap='jet', origin='lower')
ax2.contour(f_total_res, mom0_level, colors=['k'])
cp2,kw2 = colorbar.make_axes(ax2, pad=0.003, aspect=18, location='top')
cb2 = plt.colorbar(im2, cax=cp2, orientation='horizontal', ticklocation='top')
cb2.ax.set_xlabel(r'RESIDUAL')

rec = matplotlib.patches.Rectangle((-1, -1), 10, 10,
                                   angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax0.add_artist(rec)
Beam = beam(hdu, 4., 4., 'w', pix_size)
ax0.add_artist(Beam[0])

plt.savefig('/home/qyfei/Desktop/Codes/Result/mom_fitting/CO21_fit.pdf', bbox_inches='tight')

# %%
