# %%
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15
import astropy.units as u
import astropy.constants as c

plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

# %%
## Taper results
FWHMs = np.sqrt(8*np.log(2))*np.array([77.02, 93.93, 111.14])
eFWHMs1 = np.sqrt(8*np.log(2))*np.array([11.85, 4.81, 6.37])
eFWHMs2 = np.sqrt(8*np.log(2))*np.array([13.90, 5.28, 6.81])

FWHM = np.array([0.77, 0.41, 0.38])
eFWHM = np.array([0.18, 0.11, 0.20])
redshifts = np.array([0.309, 0.330, 0.421])
FWHMr = FWHM*u.arcsec/Planck15.arcsec_per_kpc_proper(redshifts)
eFWHMr = eFWHM*u.arcsec/Planck15.arcsec_per_kpc_proper(redshifts)

vcir = 0.75*FWHMs*u.Unit("km/s")
vcir_low = 0.75*(FWHMs-eFWHMs1)*u.Unit("km/s")
vcir_up = 0.75*(FWHMs+eFWHMs2)*u.Unit("km/s")
Mdyn = (vcir**2*FWHMr/c.G).to("M_sun")
Mdyn_low = (vcir_low**2*(FWHMr-eFWHMr)/c.G).to("M_sun")
Mdyn_up = (vcir_up**2*(FWHMr+eFWHMr)/c.G).to("M_sun")

eMdyn1 = Mdyn-Mdyn_low
eMdyn2 = Mdyn_up - Mdyn

# %%
## Original results
FWHMs = np.sqrt(8*np.log(2))*np.array([94.21, 89.00, 113.03])
eFWHMs1 = np.sqrt(8*np.log(2))*np.array([10.76, 4.07, 6.19])
eFWHMs2 = np.sqrt(8*np.log(2))*np.array([12.15, 4.27, 6.68])

FWHM = np.array([0.46, 0.42, 0.37])
eFWHM = np.array([0.17, 0.15, 0.10])
redshifts = np.array([0.309, 0.330, 0.421])
FWHMr = FWHM*u.arcsec/Planck15.arcsec_per_kpc_proper(redshifts)
eFWHMr = eFWHM*u.arcsec/Planck15.arcsec_per_kpc_proper(redshifts)

vcir = 0.75*FWHMs*u.Unit("km/s")
vcir_low = 0.75*(FWHMs-eFWHMs1)*u.Unit("km/s")
vcir_up = 0.75*(FWHMs+eFWHMs2)*u.Unit("km/s")
Mdyn = (vcir**2*FWHMr/c.G).to("M_sun")
Mdyn_low = (vcir_low**2*(FWHMr-eFWHMr)/c.G).to("M_sun")
Mdyn_up = (vcir_up**2*(FWHMr+eFWHMr)/c.G).to("M_sun")

eMdyn1 = Mdyn-Mdyn_low
eMdyn2 = Mdyn_up - Mdyn

# %%
MBH = np.array([20.53, 25.40, 68.55])*1e7
eMBH = np.array([6.66, 8.08, 26.57])*1e7

# %%
plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
ax.errorbar(MBH, Mdyn.value, xerr=eMBH, yerr=[eMdyn1.value, eMdyn2.value], fmt='ks', mfc='none', ms=10, mew=1, elinewidth=1, capsize=5)

plt.loglog()
# %%
