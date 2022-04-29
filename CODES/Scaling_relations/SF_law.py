# %%
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
    Beam = matplotlib.patches.Ellipse((xpos,ypos), Bmaj, Bmin, 90+bpa, edgecolor='k', facecolor='gray', fill=True, zorder=3)    
    return Beam, Bmaj, Bmin

# %%
## Load data

path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/"
work = "working/SF_law/"

Kennicutt_98_SB = np.loadtxt(path+work+'Kennicutt_98_SB.txt', dtype='str')
Kennicutt_gas_den_SB = np.power(10, np.array(Kennicutt_98_SB[:, 2], dtype='float'))
Kennicutt_SFR_den_SB = np.power(10, np.array(Kennicutt_98_SB[:, 3], dtype='float'))*0.67
Kennicutt_98_D = np.loadtxt(path+work+'Kennicutt_98_D.txt', dtype='str')
Kennicutt_gas_den_D = np.power(10, np.array(Kennicutt_98_D[:, 3], dtype='float'))
Kennicutt_SFR_den_D = np.power(10, np.array(Kennicutt_98_D[:, 5], dtype='float'))*0.67

Genzel_10 = np.loadtxt(path+work+'Genzel_10.txt', dtype='str')
Genzel_gas_den_SB = np.power(10, np.array(Genzel_10[25:, 11], dtype='float'))/2
Genzel_SFR_den_SB = np.power(10, np.array(Genzel_10[25:, 13], dtype='float'))*0.67/0.63/2
Genzel_gas_den_D = np.power(10, np.array(Genzel_10[:25, 11], dtype='float'))/2
Genzel_SFR_den_D = np.power(10, np.array(Genzel_10[:25, 13], dtype='float'))*0.67/0.63/2

Tacconi_10_D = np.loadtxt(path+work+'Tacconi_10.txt', dtype='str')
Tacconi_R12 = np.array(Tacconi_10_D[:,3], dtype='float')*1e3
Tacconi_gas_den_D = np.array(Tacconi_10_D[:,8], dtype='float')*np.power(10,10)/(np.pi*(2*Tacconi_R12)**2)
Tacconi_SFR_den_D = np.array(Tacconi_10_D[:,4], dtype='float')/(np.pi*(2*Tacconi_R12/1e3)**2)*1.06

# %%
## PG0050 center

f_cont_area = 0.994
S_CO_area = 6.07
from astropy.cosmology import Planck15
DL = Planck15.luminosity_distance(0.061)

## Area
redshift = 0.06115
## Deconvolution
FWHM_maj = 0.174*u.arcsec/Planck15.arcsec_per_kpc_proper(redshift)#*2.355
FWHM_min = 0.100*u.arcsec/Planck15.arcsec_per_kpc_proper(redshift)#*2.355
FWHM_err = 0.02*u.arcsec/Planck15.arcsec_per_kpc_proper(redshift)#*2.355
## Convolution
FWHM_maj = 0.359*u.arcsec/Planck15.arcsec_per_kpc_proper(redshift)#*2.355
FWHM_min = 0.298*u.arcsec/Planck15.arcsec_per_kpc_proper(redshift)#*2.355
FWHM_err = 0.06*u.arcsec/Planck15.arcsec_per_kpc_proper(redshift)#*2.355

#disp_maj = FWHM_maj/2.355*2*u.arcsec/Planck15.arcsec_per_kpc_proper(redshift)
#disp_min = FWHM_min/2.355*2*u.arcsec/Planck15.arcsec_per_kpc_proper(redshift)
#area = np.pi*disp_maj*disp_min/np.cos(41*u.deg)

area = np.pi*FWHM_maj*FWHM_min/(4*np.log(2))/np.cos(41*u.deg)*9
area_err1 = (area - np.pi*(FWHM_maj-FWHM_err)*(FWHM_min-FWHM_err)/(4*np.log(2))/np.cos(41*u.deg))*9
area_err2 = (area - np.pi*(FWHM_maj+FWHM_err)*(FWHM_min+FWHM_err)/(4*np.log(2))/np.cos(41*u.deg))*9

#redshift = 0.06115
#area = (0.05*u.arcsec/cosmo.arcsec_per_kpc_proper(redshift))**2*134/np.cos(41*u.deg)
#DL = cosmo.luminosity_distance(redshift)

nu_obs = 217.232*u.GHz
L_CO_area = np.sum(3.25e7*(S_CO_area)*(DL.value)**2/(1+redshift)**3/(nu_obs/u.GHz)**2)

M_gas_traditional = 3.1*L_CO_area*u.M_sun/0.9
M_H2_density_traditional = (M_gas_traditional/area).to('M_sun/pc^2').value
M_H2_density_err1_traditional = (1.55/3.1*M_gas_traditional/area).to('M_sun/pc^2').value
M_H2_density_err2_traditional = (3.1/3.1*M_gas_traditional/area).to('M_sun/pc^2').value

M_gas_true = 1.27/1.4*L_CO_area*u.M_sun/0.9
M_H2_density_true = (M_gas_true/area).to('M_sun/pc^2').value
M_H2_density_err1_true = -((1.27-0.71)*M_gas_true/area).to('M_sun/pc^2').value + M_H2_density_true
M_H2_density_err2_true = ((1.27+0.83)/3.1*M_gas_true/area).to('M_sun/pc^2').value - M_H2_density_true

SFR_density = (14.20/area*u.Unit('M_sun/yr')).value
SFR_density_err = (0.20/area*u.Unit('M_sun/yr')).value


# %%
## KS-law from fitting

gas_den_x = np.logspace(-3,7,10000)
## Different depletion timescale
SFR_den_dep = gas_den_x/1e0 
SFR_den_dep1 = gas_den_x/1e1 
SFR_den_dep2 = gas_den_x/1e2 
SFR_den_dep3 = gas_den_x/1e3 
## fitting result
SFR_den_Shangguan = np.power(gas_den_x, 1.01)*np.power(10, -2.69)
SFR_den_Bigiel = np.power(gas_den_x/10, 0.96)*np.power(10, -2.06) *1.06
SFR_den_Sanchez = np.power(gas_den_x, 0.98)*np.power(10, -9.01+6)*0.67


plt.figure(figsize=(8,8))
plt.rcParams['xtick.minor.visible'] = True
#plt.plot(gas_den_x, SFR_den_dep, 'r--')
plt.plot(gas_den_x, SFR_den_dep1, 'k:') # depletion timescale 10 Myr
plt.text(1e3, 1e2+50, r'$\tau_\mathrm{dep}=10\mathrm{Myr}$', rotation=35, fontsize=20, zorder=4)
plt.plot(gas_den_x, SFR_den_dep2, 'k:')
plt.text(2e3, 2e1+10, r'$\tau_\mathrm{dep}=100\mathrm{Myr}$', rotation=35, fontsize=20, zorder=4)
plt.plot(gas_den_x, SFR_den_dep3, 'k:')
plt.text(4e3, 4+2, r'$\tau_\mathrm{dep}=1\mathrm{Gyr}$', rotation=35, fontsize=20, zorder=4)

lines1 = [0,0,0]
labels1 = ['Shangguan+2020','Bigiel+2008','Sanchez+2021']

lines1[0], = plt.plot(gas_den_x, SFR_den_Shangguan, 'c--')
lines1[1], = plt.plot(gas_den_x, SFR_den_Bigiel, 'y--')
lines1[2], = plt.plot(gas_den_x, SFR_den_Sanchez, 'm--')
points = [0,0,0,0,0]
labels2 = ['Kennicutt+1998 starburst','Kennicutt+1998 disk','Genzel+2010 starburst','Genzel+2010 disk','Tacconi+2010 disk']
plt.errorbar(Kennicutt_gas_den_SB, Kennicutt_SFR_den_SB, fmt='k*', mfc='none', ms=10, mew=1, label="Kennicutt98 SBs")
plt.errorbar(Kennicutt_gas_den_D, Kennicutt_SFR_den_D, fmt='ko', mfc='none', ms=10, mew=1, label="Kennicutt98 SFGs")
plt.errorbar(Genzel_gas_den_SB, Genzel_SFR_den_SB, fmt='b*', mfc='none', ms=10, mew=1, label="Genzel10 SBs")
plt.errorbar(Genzel_gas_den_D, Genzel_SFR_den_D, fmt='bo', mfc='none', ms=10, mew=1, label="Genzel10 SFGs")
plt.errorbar(Tacconi_gas_den_D, Tacconi_SFR_den_D, fmt='yo', mfc='none', ms=10, mew=1, label="Tacconi10 SFGs")

# plt.errorbar(np.array([M_H2_density_traditional]), np.array([SFR_density]), xerr=[np.array([M_H2_density_err1_traditional]), np.array([M_H2_density_err2_traditional])], yerr=np.array([SFR_density_err]), fmt='C1s', mfc='none', ms=15, capsize=8, mew=1, zorder=3, label=r"MW-like $\alpha_\mathrm{CO}$")
# plt.errorbar(np.array([M_H2_density_true]), np.array([SFR_density]), xerr=[-np.array([M_H2_density_err1_true]), np.array([M_H2_density_err2_true])], yerr=np.array([SFR_density_err]), fmt='rs', mfc='none', ms=15, capsize=8, mew=1, zorder=3, label=r"ULIRG-like $\alpha_\mathrm{CO}$")

plt.errorbar(np.array([Sigma_H2.value]), np.array([Sigma_SFR.value]), xerr=[np.array([-e1Sigma_H2.value]), np.array([e2Sigma_H2.value])], yerr = np.array([eSigma_SFR.value]), fmt='rs', mfc='none', ms=6, capsize=3, mew=0.5, lw=0.5, zorder=3, label="This work")
plt.errorbar(np.array([Sigma_H2_tra.value]), np.array([Sigma_SFR.value]), xerr=[np.array([e1Sigma_H2_tra.value]), np.array([e2Sigma_H2_tra.value])], yerr = np.array([eSigma_SFR.value]), fmt='r<', mfc='none', ms=6, capsize=3, mew=0.5, lw=0.5, zorder=3)


plt.errorbar(np.array([Sigma_H2_peak.value]), np.array([Sigma_SFR_peak.value]), xerr=[np.array([-e1Sigma_H2_peak.value]), np.array([e2Sigma_H2_peak.value])], yerr = np.array([eSigma_SFR_peak.value]), fmt='C1s', mfc='none', ms=6, capsize=3, mew=0.5, lw=0.5, zorder=3)
plt.errorbar(np.array([Sigma_H2_tra_peak.value]), np.array([Sigma_SFR_peak.value]), xerr=[np.array([e1Sigma_H2_tra_peak.value]), np.array([e2Sigma_H2_tra_peak.value])], yerr = np.array([eSigma_SFR_peak.value]), fmt='C1<', mfc='none', ms=6, capsize=3, mew=0.5, lw=0.5, zorder=3)


plt.xlabel(r'$\Sigma_\mathrm{mol}$ [$\mathrm{M_\odot\cdot pc^{-2}}$]')
plt.ylabel(r'$\Sigma_\mathrm{SFR}$ [$\mathrm{M_\odot\cdot kpc^{-2}\cdot yr^{-1}}$]')

plt.loglog()
plt.xlim(1e0, 9e4)
plt.ylim(1e-3, 5e3)
l1 = plt.legend(lines1, labels1, loc="upper left", frameon=False, fontsize=15)
#l2 = plt.legend(points, labels2, loc="upper right", frameon=False, fontsize=15)
plt.legend(loc="lower right", frameon=False, fontsize=15)
plt.gca().add_artist(l1)

# plt.savefig("/home/qyfei/Desktop/Results/Result/PG0050/SF_law_mean_peak.pdf", bbox_inches='tight', dpi=300)

# %%
asymmetric_error = [M_H2_density_err1, M_H2_density_err2]
plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
ax.errorbar(np.array([M_H2_density]), np.array([SFR_density]), yerr=np.array([SFR_density_err, SFR_density_err*2]), fmt='rs', mfc='none', ms=15, capsize=8, mew=1, zorder=3, label="This work")

# %%
# %%
x = np.arange(0,10,10)
y = x**2
yerr1 = 0.1*y
yerr2 = 0.3*y
x[0] = M_H2_density
yerr1 = np.array([M_H2_density_err1])
yerr2 = np.array([M_H2_density_err2])

plt.errorbar(x, y, xerr=[yerr1, yerr2], fmt='ko', mfc='none')
# %%
x
# %%
