# %%
import numpy as np
import astropy.units as u
from scipy.special import gammaincinv
from scipy.signal import convolve as scipy_convolve
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft
from map_visualization.moment0 import load_mom0
from astropy.modeling.models import Sersic2D, Gaussian2D

# path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
# file = "PG0050_CO21-combine-line-10km-mosaic-mom0.fits"
#mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file)

def kernel(hdu):
    tran = 2*np.sqrt(2*np.log(2.0)) #convert FWHM to sigma
    xstd = abs(hdu.header['BMAJ']/hdu.header['CDELT2'])/tran
    ystd = abs(hdu.header['BMIN']/hdu.header['CDELT1'])/tran
    pa = (hdu.header['BPA']+90)*u.deg #PA's definition
    kernel_CO = Gaussian2DKernel(x_stddev = xstd, y_stddev = ystd, theta = pa, mode='center')
    return kernel_CO

def noise(hdu, x, y, intr):
    kernel_CO = kernel(hdu)
    #pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    I_n = intr * np.random.randn(len(x), len(y))
    func = scipy_convolve(I_n, kernel_CO, mode='same', method='fft')
    return func

def Disk2D(hdu, x, y, x0, y0, I_e, R_e, n_, ellip_, theta_):    
    ## This function is used to describe a galaxy whose morphology can be well described by a Sersic profile
    ## dx0, dy0 are offset from the galaxy center, in units of arcsec
    ## I_e is the intensity at effective radius/amplitude, in units of Jy/beam km/s
    ## R_e is the effective radius, in units of arcsec
    ## n is the Sersic index, dimensionless
    ## ellip is the ellipcity and is defined as e=1-b/a, where a and b are major- and minor-axis
    ## theta is the position angle, increases anti-clockwise from x-axis 
    kernel_CO = kernel(hdu)
    pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    amplitude = I_e
    r_eff = R_e/pix_size
    n = n_
    x_0, y_0 = x0, y0#size - dx0/pix_size, size - dy0/pix_size
    ellip = ellip_
    theta = -np.radians(theta_)
    function = Sersic2D(amplitude, r_eff, n, x_0, y_0, ellip, theta)
    I_r = function(x, y)
    Ir = scipy_convolve(I_r, kernel_CO, mode='same', method='fft')
    return Ir

def Expon2D(hdu, x, y, x0, y0, I_e, R_e, ellip_, theta_):
    kernel_CO = kernel(hdu)
    pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    amplitude = I_e
    r_eff = R_e/pix_size
    n = 1
    x_0, y_0 = x0, y0#size - dx0/pix_size, size - dy0/pix_size
    ellip = ellip_
    theta = theta_
    function = Sersic2D(amplitude, r_eff, n, x_0, y_0, ellip, theta)
    I_r = function(x, y)    
    Ir = scipy_convolve(I_r, kernel_CO, mode='same', method='fft')
    return Ir

def Gauss2D(hdu, x, y, x0_, y0_, I0_, xstd_, ystd_, phi_):
    # define the gaussian profile after convolving
    kernel_CO = kernel(hdu)
    pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    x0, y0 = x0_, y0_#size - dx0_/pix_size, size - dy0_/pix_size
    I0, x_std, y_std, phi = I0_, xstd_/pix_size, ystd_/pix_size, -np.radians(phi_)
    function = Gaussian2D(I0, x0, y0, x_std, y_std, phi)
    gauss = function(x, y)
    Ig = scipy_convolve(gauss, kernel_CO, mode='same', method='fft')
    return Ig

def truncated_Disk2D_ring(hdu, x, y, x0_, y0_, I_e, R_e, n_, ellip_, theta_, r_break_, dr_soft_, q_, theta_PA):
    # This model is used to describe a ring structure embedded in a Sersic disk
    # The first three parameters describe the models size, structure from observations
    # dx0_, dy0_ represents the offset between fitting center and observed center
    # I_e, R_e, ellip_, theta_ are parameters for Sersic profile
    # r_break, dr_soft, q_, theta_PA are parameters for truncated function
    kernel_CO = kernel(hdu)
    pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    amplitude = I_e
    r_eff = R_e/pix_size
    n = n_
    x_0, y_0 = x0_, y0_#size - dx0_/pix_size, size - dy0_/pix_size
    ellip = ellip_
    theta = -np.radians(theta_)
    function = Sersic2D(amplitude, r_eff, n, x_0, y_0, ellip, theta)
    I_r = function(x, y)

    thetaPA = -np.radians(theta_PA)
    r_break, dr_soft = r_break_/pix_size, dr_soft_/pix_size
    a, b  = r_break, q_ * r_break
    cos_thetaPA, sin_thetaPA = np.cos(thetaPA), np.sin(thetaPA)
    x_maj = (x - x_0) * cos_thetaPA + (y - y_0) * sin_thetaPA
    x_min = -(x - x_0) * sin_thetaPA + (y - y_0) * cos_thetaPA
    z = np.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)
    B = 2.65 - 4.98 * r_break/dr_soft
    trunc = 0.5 * (np.tanh((2 - B) * z + B) + 1)

    model_final = I_r * trunc
    model_conv = scipy_convolve(model_final, kernel_CO, mode='same', method='fft')
    return model_conv

def truncated_Disk2D_ring_out(hdu, x, y, x0_, y0_, I_e, R_e, n_, ellip_, theta_, r_break_, dr_soft_, q_, theta_PA):
    # This model is used to describe a ring structure embedded in a Sersic disk
    # The first three parameters describe the models size, structure from observations
    # dx0_, dy0_ represents the offset between fitting center and observed center
    # I_e, R_e, ellip_, theta_ are parameters for Sersic profile
    # r_break, dr_soft, q_, theta_PA are parameters for truncated function
    kernel_CO = kernel(hdu)
    pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    amplitude = I_e
    r_eff = R_e/pix_size
    n = n_
    x_0, y_0 = x0_, y0_#size - dx0_/pix_size, size - dy0_/pix_size
    ellip = ellip_
    theta = -np.radians(theta_)
    function = Sersic2D(amplitude, r_eff, n, x_0, y_0, ellip, theta)
    I_r = function(x, y)

    thetaPA = -np.radians(theta_PA)
    r_break, dr_soft = r_break_/pix_size, dr_soft_/pix_size
    a, b  = r_break, q_ * r_break
    cos_thetaPA, sin_thetaPA = np.cos(thetaPA), np.sin(thetaPA)
    x_maj = (x - x_0) * cos_thetaPA + (y - y_0) * sin_thetaPA
    x_min = -(x - x_0) * sin_thetaPA + (y - y_0) * cos_thetaPA
    z = np.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)
    B = 2.65 - 4.98 * r_break/dr_soft
    trunc = 1 - 0.5 * (np.tanh((2 - B) * z + B) + 1)

    model_final = I_r * trunc
    model_conv = scipy_convolve(model_final, kernel_CO, mode='same', method='fft')
    return model_conv



def Sersic_ring(hdu, x, y, x0_, y0_, I_e, R_e, n_, ellip_, theta_, rin_, rsin_, qin_, PAin_, rout_, rsout_, qout_, PAout_):
    # This model generates a ring-like structure for a Sersic model with combination of two truncated functions, one in and one out
    # The first three parameters describe the models size, structure from observations
    # dx0_, dy0_ represents the offset between fitting center and observed center
    # I_e, R_e, ellip_, theta_ are parameters for Sersic profile
    # r_break, dr_soft, q_, theta_PA are parameters for truncated function
    kernel_CO = kernel(hdu)
    pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    amplitude = I_e
    r_eff = R_e/pix_size
    n = n_
    # size = len(x)/2
    x_0, y_0 = x0_, y0_#size - dx0_/pix_size, size - dy0_/pix_size
    ellip = ellip_
    theta = -np.radians(theta_)
    function = Sersic2D(amplitude, r_eff, n, x_0, y_0, ellip, theta)
    I_r = function(x, y)

    PA_in = -np.radians(PAin_)
    r_in, rsoft_in = rin_/pix_size, rsin_/pix_size
    a_in, b_in  = r_in, qin_ * r_in
    cos_thetaPA_in, sin_thetaPA_in = np.cos(PA_in), np.sin(PA_in)
    x_maj_in = (x - x_0) * cos_thetaPA_in + (y - y_0) * sin_thetaPA_in
    x_min_in = -(x - x_0) * sin_thetaPA_in + (y - y_0) * cos_thetaPA_in
    z_in = np.sqrt((x_maj_in / a_in) ** 2 + (x_min_in / b_in) ** 2)
    B_in = 2.65 - 4.98 * r_in/rsoft_in
    trunc_in = 0.5 * (np.tanh((2 - B_in) * z_in + B_in) + 1)

    PA_out = -np.radians(PAout_)
    r_out, rsoft_out = rout_/pix_size, rsout_/pix_size
    a_out, b_out  = r_out, qout_ * r_out
    cos_thetaPA_out, sin_thetaPA_out = np.cos(PA_out), np.sin(PA_out)
    x_maj_out = (x - x_0) * cos_thetaPA_out + (y - y_0) * sin_thetaPA_out
    x_min_out = -(x - x_0) * sin_thetaPA_out + (y - y_0) * cos_thetaPA_out
    z_out = np.sqrt((x_maj_out / a_out) ** 2 + (x_min_out / b_out) ** 2)
    B_out = 2.65 - 4.98 * r_out/rsoft_out
    trunc_out = 1 - 0.5 * (np.tanh((2 - B_out) * z_out + B_out) + 1)

    # model_in = I_r*trunc_in
    # model_out = I_r*trunc_out
    model_final = I_r * trunc_in * trunc_out

    # model_in_conv = scipy_convolve(model_in, kernel_CO, mode='same', method='fft')
    # model_out_conv = scipy_convolve(model_out, kernel_CO, mode='same', method='fft')

    model_conv = scipy_convolve(model_final, kernel_CO, mode='same', method='fft')
    return model_conv#, model_in_conv, model_out_conv

# %%

# test, test_in, test_out = Sersic_ring(hdu, x, y, 0, 0, 10, 10, 1.0, 0.5, 60, 
#                                         5, 0.5, 0.5, 60, 
#                                         7, 0.5, 0.5, 60)
# Disk = Disk2D(hdu, x, y, 0, 0, 10, 10, 1.0, 0.5, 60)
# # test = truncated_Disk2D_ring_out(hdu, x, y, 0, 0, 10, 10, 1.0, 0.5, 60, 7, 0.5, 0.5, 60)
# # %%
# levels = np.array([10, 20, 30, 40])

# plt.figure(figsize=(10, 8))
# plt.imshow(test, origin='lower', cmap='jet')
# plt.colorbar()
# plt.contour(test, levels=levels, colors='w')
# plt.contour(Disk, levels=levels, colors='k')

# %%
