# %%
from astropy.io import fits
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt
import numpy as np
from spectral_cube import SpectralCube
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models
from scipy.optimize import minimize

# %%
object = "PG0923"
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/"+object+"/data"
file = "/PG0923+129_CO21.final.image.fits"

cube_hdu = fits.open(path+file)[0]
cube_data = cube_hdu.data[0]
cube_hdr = cube_hdu.header

cube = SpectralCube.read(path+file)
CO21_cube = cube.with_spectral_unit(unit='km/s', rest_value=230.58*u.GHz, velocity_convention='radio') 
velo = CO21_cube.spectral_axis.value

# %%
yy, xx = np.mgrid[:len(cube_data[0]), :len(cube_data[0][0])]
bmaj, bmin, pixel = cube_hdr["BMAJ"], cube_hdr["BMIN"], cube_hdr["CDELT1"]
beam2pixels = np.pi*bmaj*bmin/(4*np.log(2))/pixel**2
beamradius  = round(np.sqrt(beam2pixels/np.pi))

# %%
# cube_data_new = np.zeros((len(cube_data), len(cube_data[0]), len(cube_data[0][0])))
# logL = np.zeros((len(cube_data[0]), len(cube_data[0][0])))

# for ypos in range(len(cube_data[0])):
#     for xpos in range(len(cube_data[0][0])):
#         region_round = np.sqrt((yy-ypos)**2 + (xx-xpos)**2)

#         for i in np.array([0, 1, 2]):
#             mask_round = (region_round<=(beamradius+i))
#             mask_spectrum = np.nansum(cube_data*mask_round, axis=(1, 2))/len(np.where(mask_round)[0])
#             # mask_spectrum

#             rms_spectrum = sigma_clipped_stats(mask_spectrum, sigma=3)[-1]
#             if np.nanmax(mask_spectrum)>=3*rms_spectrum:
#                 detection = True
#                 cube_data_new[:, ypos, xpos] = mask_spectrum
#                 break
#             else:
#                 detection = False
#                 cube_data_new[:, ypos, xpos] = np.nan
    
# %%
# fits.writeto(path+"PG0923_CO21_new.fits", np.array([cube_data_new]), header=cube_hdr)
# %%
ypos, xpos = 89, 90
region_round = np.sqrt((yy-ypos)**2 + (xx-xpos)**2)
mask_round = (region_round<=(beamradius))
mask_spectrum = np.nansum(cube_data*mask_round, axis=(1, 2))/len(np.where(mask_round)[0])
nvctr = np.where(mask_spectrum == np.nanmax(mask_spectrum))
rms_spectrum = sigma_clipped_stats(mask_spectrum, sigma=3)[-1]

# %%
from cube_analysis.Gaussians import *
para1, BIC1 = Single_Gaussian_fit(velo, mask_spectrum, rms_spectrum)
# para1_mcmc, BIC1_mcmc = Single_Gaussian_mcmc(velo, mask_spectrum, rms_spectrum)

para2, BIC2 = Double_Gaussian_fit(velo, mask_spectrum, rms_spectrum)
# samples2_mcmc, para2_mcmc, BIC2_mcmc = Double_Gaussian_mcmc(velo, mask_spectrum, rms_spectrum, para2)

para3, BIC3 = Triple_Gaussian_fit(velo, mask_spectrum, rms_spectrum)
# samples3_mcmc, para3_mcmc, BIC3_mcmc = Triple_Gaussian_mcmc(velo, mask_spectrum, rms_spectrum, para3)

print(BIC1, BIC2, BIC3)#, BIC1_mcmc, BIC2_mcmc, BIC3_mcmc)

# %%
plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=20)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)
plt.figure(figsize=(15, 6))
ax = plt.subplot(111)
ax.step(velo, mask_spectrum, 'k', where='mid')
ax.errorbar(velo, mask_spectrum, yerr=rms_spectrum, fmt='k.', mfc='none', capsize=0)
ax.plot(velo, Gauss(velo, para1[0], para1[1], para1[2]), 'r:')
ax.plot(velo, Gauss(velo, para2[0], para2[1], para2[2]) + Gauss(velo, para2[3], para2[4], para2[5]), 'g--')
ax.plot(velo, Gauss(velo, para3[0], para3[1], para3[2]) + Gauss(velo, para3[3], para3[4], para3[5]) + Gauss(velo, para3[6], para3[7], para3[8]), 'b')
plt.show()
# %%

# %%
