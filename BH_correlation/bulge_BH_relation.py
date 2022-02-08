# %%
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import Planck15
# %%

bulge = np.loadtxt('/home/qyfei/Desktop/Codes/BH_correlation/ar4-bulges.dat.txt', dtype='str')
log_Mbulge_c = np.array(bulge[:21,16], 'float')
elog_Mbulge_c = np.array(bulge[:21,18], 'float')

log_Mbulge_p = np.array(bulge[21:,16], 'float')
elog_Mbulge_p = np.array(bulge[21:,18], 'float')

log_Mbh_c = np.log10(np.array(bulge[:21,19], 'float')*np.power(10, np.array(bulge[:21,25], 'float')))
minlog_Mbh_c = np.log10(np.array(bulge[:21,20], 'float')*np.power(10, np.array(bulge[:21,25], 'float')))
maxlog_Mbh_c = np.log10(np.array(bulge[:21,22], 'float')*np.power(10, np.array(bulge[:21,25], 'float')))
eminlog_Mbh_c = log_Mbh_c - minlog_Mbh_c
emaxlog_Mbh_c = - log_Mbh_c + maxlog_Mbh_c

log_Mbh_p = np.log10(np.array(bulge[21:,19], 'float')*np.power(10, np.array(bulge[21:,25], 'float')))
minlog_Mbh_p = np.log10(np.array(bulge[21:,20], 'float')*np.power(10, np.array(bulge[21:,25], 'float')))
maxlog_Mbh_p = np.log10(np.array(bulge[21:,22], 'float')*np.power(10, np.array(bulge[21:,25], 'float')))
eminlog_Mbh_p = log_Mbh_p - minlog_Mbh_p
emaxlog_Mbh_p = - log_Mbh_p + maxlog_Mbh_p

ellip = np.loadtxt('/home/qyfei/Desktop/Codes/BH_correlation/ar4-ellipticals.dat.txt', dtype='str')

log_Mellip = np.array(ellip[:,10], 'float')
elog_Mellip = np.array(ellip[:,12], 'float')

log_Mbh_e = np.log10(np.array(ellip[:,13], 'float')*np.power(10, np.array(ellip[:,19], 'float')))
minlog_Mbh_e = np.log10(np.array(ellip[:,14], 'float')*np.power(10, np.array(ellip[:,19], 'float')))
maxlog_Mbh_e = np.log10(np.array(ellip[:,16], 'float')*np.power(10, np.array(ellip[:,19], 'float')))
eminlog_Mbh_e = log_Mbh_e - minlog_Mbh_e
emaxlog_Mbh_e = - log_Mbh_e + maxlog_Mbh_e

Mbulge_c = np.power(10, log_Mbulge_c)
Mbh_c = np.power(10, log_Mbh_c)
eMbulge_c = (1 - np.power(10, - elog_Mbulge_c))*Mbulge_c
emin_Mbh_c = (1 - np.power(10, -eminlog_Mbh_c))*Mbh_c
emax_Mbh_c = (np.power(10, emaxlog_Mbh_c) - 1)*Mbh_c

Mellip = np.power(10, log_Mellip)
Mbh_e = np.power(10, log_Mbh_e)
eMellip = (1 - np.power(10, - elog_Mellip))*Mellip
emin_Mbh_e = (1 - np.power(10, -eminlog_Mbh_e))*Mbh_e
emax_Mbh_e = (np.power(10, emaxlog_Mbh_e) - 1)*Mbh_e

PG_bh = np.loadtxt('/home/qyfei/Desktop/Codes/BH_correlation/PG_bh.txt', dtype='str')
log_Mbh_PG = np.array(PG_bh[:,14], 'float')
PG_host = np.loadtxt('/home/qyfei/Desktop/Codes/BH_correlation/PG_bulge.txt', dtype='str')
log_Mbulge_PG = np.array(PG_host[:,9], 'float')
elog_Mbulge_PG = np.array(PG_host[:,10], 'float')

Mbulge_PG = np.power(10, log_Mbulge_PG)
Mbh_PG = np.power(10, log_Mbh_PG)
emin_Mbulge_PG = (1 - np.power(10, - elog_Mbulge_PG))*Mbulge_PG
emax_Mbulge_PG = (np.power(10, elog_Mbulge_PG) - 1)*Mbulge_PG

# %%
## IR QSOs from Tan+19
Mbh_Tan = np.array([2.1, 3.2, 15.7, 0.4, 1.0, 0.8, 0.5, 1.0])*1e7
Mdyn_Tan = np.array([4.1, 1.1, 4.3, 1.6, 6.9, 0.9, 0.7, 1.3])*1e10
emin_Mdyn_Tan = Mdyn_Tan - (np.array([4.1, 1.1, 4.3, 1.6, 6.9, 0.9, 0.7, 1.3])
                 -np.array([0.7, 0.2, 0.8, 0.2, 1.8, 0.2, 0.2, 0.3]))*1e10
emax_Mdyn_Tan = (np.array([4.1, 1.1, 4.3, 1.6, 6.9, 0.9, 0.7, 1.3])
                 +np.array([0.7, 0.2, 0.8, 0.2, 1.8, 0.2, 0.2, 0.3]))*1e10 - Mdyn_Tan

# %%
plt.figure(figsize=(12,12))
plt.errorbar(Mbulge_c, Mbh_c, xerr=eMbulge_c, yerr=[emin_Mbh_c, emax_Mbh_c], 
            fmt='ro', ms=8, capsize=0, label='Classical bulges')
#plt.errorbar(log_Mbulge_p, log_Mbh_p, xerr=elog_Mbulge_p, yerr=[eminlog_Mbh_p, emaxlog_Mbh_p], 
#            fmt='bo', ms=4, capsize=0, label='pseudobulge')
plt.errorbar(Mellip, Mbh_e, xerr=eMellip, yerr=[emin_Mbh_e, emax_Mbh_e], 
            fmt='ko', ms=8, capsize=0, label='Ellipticals')
plt.errorbar(Mbulge_PG, Mbh_PG, xerr=[emin_Mbulge_PG, emax_Mbulge_PG], 
            fmt='ks', mfc='none', ms=8, capsize=0, label='PG quasars')

plt.loglog()
plt.xlim(10**8.5, 10**12.4)
plt.ylim(10**6, 10**10.5)
plt.xlabel('$M_\mathrm{bulge}/M_\odot$')
plt.ylabel('$M_\mathrm{BH}/M_\odot$')

plt.legend()

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

## BH mass
MBH = np.array([20.53, 25.40, 68.55])*1e7
eMBH = np.array([6.66, 8.08, 26.57])*1e7

# %%
plt.figure(figsize=(12,12))
ax = plt.subplot(111)
ax.errorbar(Mbulge_c, Mbh_c, xerr=eMbulge_c, yerr=[emin_Mbh_c, emax_Mbh_c], 
            fmt='ro', ms=8, capsize=0, zorder=3, label='Classical bulges')
ax.errorbar(Mellip, Mbh_e, xerr=eMellip, yerr=[emin_Mbh_e, emax_Mbh_e], 
            fmt='ko', ms=8, capsize=0, zorder=3, label='Ellipticals')

ax.errorbar(10**11, 1.7e10, xerr=(1-10**(-0.09))*1e11, yerr=1.7e10-1.4e10, 
            fmt='ro', ms=8, capsize=0, zorder=3, alpha=0.5)
ax.errorbar(10**9.64, 6e8, xerr=(1-10**(-0.1))*10**9.64, yerr=6e8-4e8, 
            fmt='ko', ms=8, capsize=0, zorder=3, alpha=0.5)
ax.errorbar(Mbulge_PG, Mbh_PG,
            fmt='bs', mfc='none', ms=8, capsize=0, label='PG quasars')
ax.errorbar(Mdyn_Tan, Mbh_Tan, xerr=[emin_Mdyn_Tan, emax_Mdyn_Tan],
            fmt='co', mfc='none', ms=8, capsize=0, zorder=3, label='Type 1 AGN in ULIRGs')
            
x0 = np.logspace(6, 13, 1000)
y0 = 0.49*(x0/1e11)**1.16*1e9
y01 = (0.49-0.05)*(x0/1e11)**(1.16-0.08)*1e9
y02 = (0.49+0.06)*(x0/1e11)**(1.16+0.08)*1e9

#ax.plot(x0, np.power(10, -4.622)*np.power(x0, 1.210), 'k')
ax.plot(x0, y0, 'k')
ax.fill_between(x0, y0/2, y0*2, color='k', alpha=0.3)

i = np.deg2rad(40)
Mdyn = (vcir**2*FWHMr/c.G).to("M_sun")/np.sin(i)**2
Mdyn_low = (vcir_low**2*(FWHMr-eFWHMr)/c.G).to("M_sun")/np.sin(i)**2
Mdyn_up = (vcir_up**2*(FWHMr+eFWHMr)/c.G).to("M_sun")/np.sin(i)**2
eMdyn1 = Mdyn-Mdyn_low
eMdyn2 = Mdyn_up - Mdyn
ax.errorbar(Mdyn.value, MBH, xerr=[eMdyn1.value, eMdyn2.value], yerr=eMBH, fmt='y*', mfc='gold', ms=30, mew=1, elinewidth=1, capsize=10, zorder=5)

inc = np.deg2rad(np.array([60, 20, 10, 5]))
for i in inc:
    Mdyn = (vcir**2*FWHMr/c.G).to("M_sun")/np.sin(i)**2
    Mdyn_low = (vcir_low**2*(FWHMr-eFWHMr)/c.G).to("M_sun")/np.sin(i)**2
    Mdyn_up = (vcir_up**2*(FWHMr+eFWHMr)/c.G).to("M_sun")/np.sin(i)**2
    eMdyn1 = Mdyn-Mdyn_low
    eMdyn2 = Mdyn_up - Mdyn
    ax.errorbar(Mdyn.value, MBH, xerr=[eMdyn1.value, eMdyn2.value], yerr=eMBH, fmt='k+', mfc='none', ms=0, mew=1, elinewidth=1, capsize=0, zorder=5, alpha=0.4)

#plt.plot(x0, y0, 'k:', label='Kormendy & Ho 2013')
ax.loglog()
ax.set_xlim(10**8.5, 10**12.4)
ax.set_ylim(10**6, 10**10.5)
ax.set_xlabel('$M_\mathrm{bulge}$ [$M_\odot$]')
ax.set_ylabel('$M_\mathrm{BH}$ [$M_\odot$]')
plt.legend(frameon=False)

#plt.savefig("/home/qyfei/Desktop/Results/NOEMA_detection/pre/bulge_BH_relation_origin.pdf", bbox_inches="tight", dpi=300)

# %%
