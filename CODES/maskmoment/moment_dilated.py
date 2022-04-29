# %%
import sys

from sklearn.metrics import r2_score
from maskmoment import *
from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import warnings

from spectral_cube import SpectralCube
warnings.filterwarnings("ignore")
import astropy.units as u
import astropy.constants as c

# %%
def quadplot(dir, name, extmask=None):
    basename = dir+"/"+name
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,12))
    mom0 = fits.getdata(basename+'.mom0.fits.gz')
    ax1.imshow(mom0,origin='lower',cmap='CMRmap')
    ax1.set_title(name+' - Moment 0',fontsize='x-large')
    mom1 = fits.getdata(basename+'.mom1.fits.gz')
    ax2.imshow(mom1,origin='lower',cmap='jet')
    ax2.set_title(name+' - Moment 1',fontsize='x-large')
    mom2 = fits.getdata(basename+'.mom2.fits.gz')
    ax3.imshow(mom2,origin='lower',cmap='CMRmap')
    ax3.set_title(name+' - Moment 2',fontsize='x-large')
    if extmask is None:
        mask = np.sum(fits.getdata(basename+'.mask.fits.gz'),axis=0)
    else:
        mask = np.sum(fits.getdata(extmask),axis=0)
    ax4.imshow(mask,origin='lower',cmap='CMRmap_r')
    ax4.set_title('Projected Mask',fontsize='x-large')
    plt.subplots_adjust(hspace=0.15,wspace=0.15)
    plt.show()
    return
# %%
# dir = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG0923"
# file = "/PG0923+126_CO21.final.image.fits"
# newfile = "/PG0923+126_CO21.final.image.vel.fits"

# dir = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG1244/CO32/Re_Re_calibration/product"
# file = "/PG1244+026_CO32_bin_clean.fits"
# newfile = "/PG1244+026_CO32_bin_clean_vel.fits"

# dir = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line"
# file = "/PG0050_CO21-combine-line-10km-mosaic.fits"
# newfile = "/PG0050_CO21-combine-line-10km-mosaic-vel.fits"

cube = SpectralCube.read(dir+file)

newcube = cube.with_spectral_unit(u.Unit("km/s"), velocity_convention='radio', rest_value=345.80*u.GHz)
newcube.write(dir+newfile, format='fits')  

# %%
dir = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG0923/data"
newfile = "/PG0923+129_CO21.final.image.vel.fits"

# dir = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG1244/CO32/Re_Re_calibration/product"
# newfile = "/PG1244+026_CO32_bin_clean_vel.fits"

name = "PG0923"
maskmoment(img_fits=dir+newfile, 
           #gain_fits='NGC4047.co.smo7gain.fits.gz',
           snr_hi=3.5, snr_lo=2, minbeam=1, snr_hi_minch=2, snr_lo_minch=2, nguard=[0, 0], 
           outname=name+'.dilmsk', to_kelvin=False, output_2d_mask=True)

# %%
quadplot(dir, name+'.dilmsk')

# %%
## Example 3: Dilated smooth-and-mask.  Expand from the 3.5$\sigma$ to 2$\sigma$ contour of the smoothed cube.

dir = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG1244/CO32/Re_Re_calibration/product"
newfile = "/PG1244+026_CO32_bin_clean_vel.fits"
name = "PG1244_CO32"

maskmoment(img_fits=dir+newfile, 
        #    rms_fits='NGC4047.dilmsk.ecube.fits.gz',
           snr_hi=5, snr_lo=3, fwhm=1, vsm=None, snr_lo_minch=3, minbeam=2,
           outname=name+'.dilsmomsk', output_2d_mask=True)

quadplot(dir, name+'.dilsmomsk')

# %%
## Ellipse
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
from astropy.convolution import convolve

def kernel(hdu):
    tran = 2*np.sqrt(2*np.log(2.0)) #convert FWHM to sigma
    xstd = abs(hdu.header['BMAJ']/hdu.header['CDELT2'])/tran
    ystd = abs(hdu.header['BMIN']/hdu.header['CDELT1'])/tran
    pa = np.rad2deg(hdu.header['BPA'])+90 #PA's definition

    kernel_CO = Gaussian2DKernel(x_stddev = xstd, y_stddev = ystd, theta = pa, mode='center')
    return kernel_CO

def ellipse(hdu, x, y, x_0, y_0, q, theta, C0):
    # kernel_CO = kernel(hdu)
    # pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    # size = 200
    # x_0, y_0 = size - dx0_/pix_size, size - dy0_/pix_size
    theta = np.radians(theta + 90)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    x_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
    x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
    ell = (abs(x_maj) ** (C0+2) + abs((x_min)/q) ** (C0+2)) ** (1/(C0+2))
    # ell_conv = scipy_convolve(ell, kernel_CO, mode='same', method='fft')
    return ell

def ellipse_conv(hdu, x, y, dx0_, dy0_, q, theta, C0):
    kernel_CO = kernel(hdu)
    pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    size = 200
    x_0, y_0 = size - dx0_/pix_size, size - dy0_/pix_size
    theta = np.radians(theta + 90)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    x_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
    x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
    ell = ((x_maj) ** (C0+2) + (x_min/q) ** (C0+2)) ** (1/(C0+2))
    ell_conv = convolve(ell, kernel_CO)#, mode='same', method='fft')
    return ell_conv

def Fourier_pert(hdu, x, y, x_0, y_0, q, m, am, phim):
    # pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    # size = 200
    # x_0, y_0 = size - dx0_/pix_size, size - dy0_/pix_size
    theta = np.arctan2((y-y_0), (x-x_0)/q) + np.radians(90)
    phim = np.radians(phim + 90)
    pert = am * np.cos(m*(theta+phim))
    return pert

def Fourier_conv(hdu, x, y, x_0, y_0, q, C0, m, am, phim):
    kernel_CO = kernel(hdu)

    # pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    # size = 200
    theta = np.arctan2((y-y_0), (x-x_0)/q) + np.radians(90)
    phim = np.radians(phim + 90)

    elli = ellipse(hdu, x, y, x_0, y_0, q, theta, C0)
    mode = am * np.cos(m*(theta+phim))
    r = elli * (1 + mode)
    r_conv = convolve(r, kernel_CO)#, mode='same', method='fft')
    return r_conv

def bending(hdu, x, y, x_0, y_0, m, am, rs):
    pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    size = 200
    x, y = x - x_0, y - y_0
    y_new = y + am * (x/rs) ** m
    return y_new, x

def rotation(x, y, x_0, y_0, theta_out, r_in, r_out, inc, PA, alpha):
    PA = np.radians(PA + 90)
    theta_out = np.radians(theta_out + 90)
    cos_theta, sin_theta, ratio = np.cos(PA), np.sin(PA), np.cos(np.radians(inc))
    x_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
    x_min = (-(x - x_0) * sin_theta + (y - y_0) * cos_theta) * ratio
    r = np.sqrt((x_maj - x_0) ** 2 + (x_min - y_0) ** 2)
    amp = (0.5*(r/r_out+1))**alpha
    return theta_out * np.tanh(r)*amp

# %%
hdu = fits.open(dir+newfile)[0]
yy, xx = np.mgrid[:400, :400]

q, C0 = 1., 0.
m, am, phim, rs = 3, 0.05, 45, 5
yy1, xx1 = bending(hdu, xx, yy, 200, 200, m, am, rs)
testf = Fourier_conv(hdu, xx1, yy1, 0, 0, q, C0, m, am, phim)

# %%
plt.figure(figsize=(10, 8))
plt.imshow(testf, origin='lower')
plt.colorbar()
plt.contour(testf, levels=np.arange(0, 80, 10), colors=['k'])

# %%
x_0, y_0, theta_out, r_in, r_out, inc, PA, alpha = 200, 200, 45, 0, 50, 0, 0, 0
test_rot = rotation(xx, yy, x_0, y_0, theta_out, r_in, r_out, inc, PA, alpha)

# %%
plt.imshow(test_rot, origin='lower')
plt.colorbar()
# %%
