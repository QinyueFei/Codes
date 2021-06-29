"""
Plot correlation between SFR and gas inflow rate
"""

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import astropy.constants as const
from sigma_sf import sigma_sf

# Data from Schmidt+ 2016; circular velocities and values of Sigma_g
# and Sigma_sfr are from Leroy+ 2013
galnames_schmidt16 = ['NGC 2403', 'NGC 2841', 'NGC 2903', 'NGC 3198',
                      'NGC 3521', 'NGC 3621', 'NGC 5055', 'NGC 6946',
                      'NGC 7331', 'NGC 0925']
sfr_schmidt16 = np.array([0.38, 0.74, 0.44, 0.93, 2.1, 2.09,
                          2.12, 3.24, 2.99, 0.56]) * u.Msun/u.yr
mdot_in_schmidt16 = np.array([1.2, 0.0, 0.54, 1.02, 0.91, 0.0,
                              0.0, 13.12, 1.03, 0.0]) * u.Msun/u.yr
vphi_schmidt16 = np.array([120., 310., 210., 150.,
                           229., 0, 200., 190.,
                           260., 140.]) * u.km/u.s
Sigma_g_schmidt16 = np.array([10., 4.6, 12., 8.4,
                                22., 0, 18., 37.,
                                16., 10.]) * u.Msun/u.pc**2
Sigma_sfr_schmidt16 = np.array([3.3, 1.4, 5.7, 2.3,
                                7.8, 0, 10., 21.,
                                4.4, 1.3]) * 1e-3 * u.Msun/u.pc**2/u.Myr
fsf_schmidt16 = Sigma_sfr_schmidt16 / (Sigma_g_schmidt16 / (2e3*u.Myr))
fsf_schmidt16[fsf_schmidt16 > 1] = 1.0

# Fiducial parameter values that we won't alter
sigma_wnm = 5.4
sigma_mol = 0.2
phimp = 1.4
epsff = 0.015
tsfmax = 2.0*u.Gyr
Qmin = 1.0
phia = 1.0
eta = 1.5
phiQ = 2.0
pmstar = 3000.*u.km/u.s

# Four sets of parameters: dwarf-like, spiral-like, ULIRG-like, hi-z
fsf = np.array([0.2, 0.5, 1.0, 1.0])
vphi = np.array([60., 220., 300., 200.])*u.km/u.s
fgP = np.array([0.9, 0.5, 1.0, 0.7])
fgQ = np.array([0.9, 0.5, 1.0, 0.7])
torb = np.array([100., 200., 5., 200.])*u.Myr
beta = np.array([0.5, 0.0, 0.5, 0.0])
phia = np.array([1., 1., 3., 3.])
th_names = ['Local dwarf', 'Local spiral', 'ULIRG', 'High-$z$']

# Generate model predictions
sfr_fid = []
mdot_fid = []
sfr_nofb = []
mdot_nofb = []
sigma_vec = np.logspace(0, 3, 1000)
for fsf_, vphi_, fgP_, fgQ_, torb_, beta_, phia_ in \
    zip(fsf, vphi, fgP, fgQ, torb, beta, phia):

    # Fiducial model
    sigma_th = sigma_wnm*(1.0 - fsf_) + sigma_mol*fsf_
    ssf = sigma_sf(sigma_th = sigma_th,
                   fsf = fsf_, beta=beta_,
                   torb=torb_.to(u.Myr))*u.km/u.s
    sigma = ssf * sigma_vec
    phint = 1.0 - (sigma_th*(u.km/u.s)/sigma)**2
    sfr = np.sqrt(2.0/(1.0+beta_))*phia_*fsf_*fgQ_*vphi_**2*sigma / \
          (np.pi*const.G*Qmin) * np.maximum(
              np.sqrt(2.0*(1.0+beta_)/(3.0*fgP_*phimp)) *
              8.0*epsff*fgQ_/Qmin, torb_/tsfmax)
    mdot = 4.0*(1.0+beta_)*eta*phiQ*phint**1.5 / \
           ((1.0-beta_)*const.G*Qmin**2) * fgQ_**2 * sigma**3 * \
           (1.0 - ssf/sigma)
    sfr_fid.append(sfr)
    mdot_fid.append(mdot)

    # No feedback model
    sigma = sigma_vec * sigma_th*u.km/u.s
    phint = 1.0 - (sigma_th*u.km/u.s/sigma)**2
    sfr = np.sqrt(2.0/(1.0+beta_))*phia_*fsf_*fgQ_*vphi_**2*sigma / \
          (np.pi*const.G*Qmin) * np.maximum(
              np.sqrt(2.0*(1.0+beta_)/(3.0*fgP_*phimp)) *
              8.0*epsff*fgQ_/Qmin, torb_/tsfmax)
    mdot = 4.0*(1.0+beta_)*eta*phiQ*phint**1.5 / \
           ((1.0-beta_)*const.G*Qmin**2) * fgQ_**2 * sigma**3
    sfr_nofb.append(sfr)
    mdot_nofb.append(mdot)
    

# Plot
fig = plt.figure(1, figsize=(4,3.5))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.clf()
try:
    # Depends on matplotlib version
    plt.rc('xtick', direction='in', top=True)
    plt.rc('ytick', direction='in', right=True)
except:
    pass

for i in range(len(sfr_fid)):
    plt.plot(sfr_fid[i].to(u.Msun/u.yr),
             mdot_fid[i].to(u.Msun/u.yr),
             label=th_names[i])
p,=plt.plot([1e-2,1e2], [1e-2,1e2], 'k--', label=r'$\dot{M} = \dot{M}_*$')
p,=plt.plot(sfr_schmidt16.to(u.Msun/u.yr),
            mdot_in_schmidt16.to(u.Msun/u.yr), 'bo',
            label='Schmidt+ 2016')

# Legend
plt.legend(loc='upper left', prop={"size":8})

# Adjust axes
plt.xlim([1e-2, 1e2])
plt.ylim([1e-2, 1e2])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\dot{M}_*$ [$M_\odot$ yr$^{-1}$]')
plt.ylabel(r'Inflow rate = $\dot{M}$ [$M_\odot$ yr$^{-1}$]')
plt.subplots_adjust(left=0.18, bottom=0.15)

# Save
plt.savefig('inflow_sfr.pdf')



fig = plt.figure(2, figsize=(4,3.5))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.clf()
try:
    # Depends on matplotlib version
    plt.rc('xtick', direction='in', top=True)
    plt.rc('ytick', direction='in', right=True)
except:
    pass

for i in range(4):
    plt.plot(sfr_fid[i].to(u.Msun/u.yr) /
             (vphi[i].to(u.km/u.s)**2 * fsf[i]),
             mdot_fid[i].to(u.Msun/u.yr),
             label=th_names[i])
p,=plt.plot(sfr_schmidt16.to(u.Msun/u.yr) /
            (vphi_schmidt16.to(u.km/u.s)**2 * fsf_schmidt16),
            mdot_in_schmidt16.to(u.Msun/u.yr), 'bo',
            label='Schmidt+ 2016')

# Legend
plt.legend(loc='upper left', prop={"size":8})


# Adjust axes
plt.xlim([5e-6, 5e-4])
plt.ylim([1e-1, 1e2])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\dot{M}_*/f_{\mathrm{sf}} v_\phi^2$'
           ' [$M_\odot$ yr$^{-1}$ / (km s$^{-1}$)$^2$]')
plt.ylabel(r'Inflow rate = $\dot{M}$ [$M_\odot$ yr$^{-1}$]')
plt.subplots_adjust(left=0.18, bottom=0.15)

# Save
plt.savefig('inflow_sfr_mod.pdf')
