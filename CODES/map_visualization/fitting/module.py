import numpy as np
import astropy.units as u
from scipy.special import gammaincinv
from scipy.signal import convolve as scipy_convolve
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft
from map_visualization.moment0 import load_mom0

path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
file = "PG0050_CO21-combine-line-10km-mosaic-mom0.fits"
#mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file)

def kernel(hdu):
    tran = 2*np.sqrt(2*np.log(2.0)) #convert FWHM to sigma
    xstd = abs(hdu.header['BMAJ']/hdu.header['CDELT2'])/tran
    ystd = abs(hdu.header['BMIN']/hdu.header['CDELT1'])/tran
    pa = (hdu.header['BPA']+90)*u.deg #PA's definition

    from scipy.signal import convolve as scipy_convolve
    from astropy.convolution import convolve
    kernel_CO = Gaussian2DKernel(x_stddev = xstd, y_stddev = ystd, theta = pa, mode='center')
    return kernel_CO

def noise(hdu, x, y, intr):
    kernel_CO = kernel(hdu)
    #pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    I_n = intr * np.random.randn(len(x), len(y))
    func = scipy_convolve(I_n, kernel_CO, mode='same', method='fft')
    return func

def Disk2D(hdu, x, y, dx0, dy0, I_e, R_e, n_, ellip_, theta_):    
    ## This function is used to describe a galaxy whose morphology can be well described by a Sersic profile
    ## dx0, dy0 are offset from the galaxy center, in units of arcsec
    ## I_e is the intensity at effective radius/amplitude, in units of Jy/beam km/s
    ## R_e is the effective radius, in units of arcsec
    ## n is the Sersic index, dimensionless
    ## ellip is the ellipcity and is defined as e=1-b/a, where a and b are major- and minor-axis
    ## theta is the position angle, increases anti-clockwise from x-axis 
    from astropy.modeling.models import Sersic2D
    kernel_CO = kernel(hdu)
    pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    amplitude = I_e
    r_eff = R_e/pix_size
    n = n_
    size = 100
    x_0, y_0 = size - dx0/pix_size, size - dy0/pix_size
    ellip = ellip_
    theta = -np.radians(theta_)
    function = Sersic2D(amplitude, r_eff, n, x_0, y_0, ellip, theta)
    I_r = function(x, y)
    Ir = scipy_convolve(I_r, kernel_CO, mode='same', method='fft')
    return Ir

def Expon2D(hdu, x, y, dx0, dy0, I_e, R_e, ellip_, theta_):
    from astropy.modeling.models import Sersic2D
    kernel_CO = kernel(hdu)
    pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    amplitude = I_e
    r_eff = R_e/pix_size
    n = 1
    size = 100
    x_0, y_0 = size - dx0/pix_size, size - dy0/pix_size
    ellip = ellip_
    theta = theta_
    function = Sersic2D(amplitude, r_eff, n, x_0, y_0, ellip, theta)
    I_r = function(x, y)    
    Ir = scipy_convolve(I_r, kernel_CO, mode='same', method='fft')
    return Ir

def Gauss2D(hdu, x, y, dx0_, dy0_, I0_, xstd_, ystd_, phi_):
    # define the gaussian profile after convolving
    from astropy.modeling.models import Gaussian2D
    kernel_CO = kernel(hdu)
    pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    size = 100
    x0, y0 = size - dx0_/pix_size, size - dy0_/pix_size
    I0, x_std, y_std, phi = I0_, xstd_/pix_size, ystd_/pix_size, -np.radians(phi_)
    function = Gaussian2D(I0, x0, y0, x_std, y_std, phi)
    gauss = function(x, y)
    Ig = scipy_convolve(gauss, kernel_CO, mode='same', method='fft')
    return Ig

