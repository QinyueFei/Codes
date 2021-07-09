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

f_cont_area = 1.15
S_CO_area = 15.97
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.8, Om0=0.308)

## Area
redshift = 0.06115
major_dec = 2.922*0.05*u.arcsec/cosmo.arcsec_per_kpc_proper(redshift)*2.355
minor_dec = 1.533*0.05*u.arcsec/cosmo.arcsec_per_kpc_proper(redshift)*2.355
area0 = np.pi*major_dec*minor_dec/np.cos(38*u.deg)

redshift = 0.06115
area = (0.05*u.arcsec/cosmo.arcsec_per_kpc_proper(redshift))**2*134/np.cos(38*u.deg)
DL = cosmo.luminosity_distance(redshift)

nu_obs = 217.232*u.GHz
L_CO_area = np.sum(3.25e7*(S_CO_area)*(DL/u.Mpc)**2/(1+redshift)**3/(nu_obs/u.GHz)**2)

M_gas = 3.1*L_CO_area*u.M_sun/0.9

M_H2_density = (M_gas/area).to('M_sun/pc^2').value
M_H2_density_err = (10**0.30*u.Msun/area).to('M_sun/pc^2').value
SFR_density = (9.33/area*u.Unit('M_sun/yr')).value
SFR_density_err = (1.47/area*u.Unit('M_sun/yr')).value


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
plt.rcParams['xtick.major.bottom'] = True
plt.rcParams['ytick.major.left'] = True
#plt.rcParams['ytick.minor.visible'] = True
#plt.plot(gas_den_x, SFR_den_dep, 'r--')
plt.plot(gas_den_x, SFR_den_dep1, 'k:') # depletion timescale 10 Myr
plt.text(1e3, 1e2+50, r'$\tau_\mathrm{dep}=10\mathrm{Myr}$', rotation=35, fontsize=15)
plt.plot(gas_den_x, SFR_den_dep2, 'k:')
plt.text(2e3, 2e1+10, r'$\tau_\mathrm{dep}=100\mathrm{Myr}$', rotation=35, fontsize=15)
plt.plot(gas_den_x, SFR_den_dep3, 'k:')
plt.text(4e3, 4+2, r'$\tau_\mathrm{dep}=1\mathrm{Gyr}$', rotation=35, fontsize=15)


lines1 = [0,0,0]
labels1 = ['Shangguan+2020','Bigiel+2008','Sanchez+2021']

lines1[0], = plt.plot(gas_den_x, SFR_den_Shangguan, 'c--')
lines1[1], = plt.plot(gas_den_x, SFR_den_Bigiel, 'y--')
lines1[2], = plt.plot(gas_den_x, SFR_den_Sanchez, 'm--')
points = [0,0,0,0,0]
labels2 = ['Kennicutt+1998 starburst','Kennicutt+1998 disk','Genzel+2010 starburst','Genzel+2010 disk','Tacconi+2010 disk']
points[0], = plt.plot(Kennicutt_gas_den_SB, Kennicutt_SFR_den_SB, 'k*')
points[1], = plt.plot(Kennicutt_gas_den_D, Kennicutt_SFR_den_D, 'ko')
points[2], = plt.plot(Genzel_gas_den_SB, Genzel_SFR_den_SB, 'b*')
points[3], = plt.plot(Genzel_gas_den_D, Genzel_SFR_den_D, 'bo')
points[4], = plt.plot(Tacconi_gas_den_D, Tacconi_SFR_den_D, 'yo')

plt.errorbar(M_H2_density, SFR_density, yerr=SFR_density_err, fmt='C1o', mfc='none', ms=10, capsize=5, label="PG0050 center")


plt.xlabel(r'$\Sigma_\mathrm{H_2}$ [$\mathrm{M_\odot\cdot pc^{-2}}$]')
plt.ylabel(r'$\Sigma_\mathrm{SFR}$ [$\mathrm{M_\odot\cdot kpc^{-2}\cdot yr^{-1}}$]')

plt.loglog()
plt.xlim(1, 9e4)
plt.ylim(1e-3, 5e3)
l1 = plt.legend(lines1, labels1, loc="upper left", frameon=False)
plt.legend(loc="lower right", frameon=False)
plt.gca().add_artist(l1)

plt.savefig('/home/qyfei/Desktop/Codes/Result/SF_law_comp.pdf', bbox_inches='tight', dpi=300)


# %%
