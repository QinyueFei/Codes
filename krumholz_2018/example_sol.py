"""
This script plots some example solutions to the fiducial model
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sigma_sf import sigma_sf
from kmtnew import fH2KMTnew

# Units and constants
msun = 1.99e33
pc = 3.09e18
kpc = 1e3*pc
yr = 365.25*24.*3600.
Myr = 1e6*yr
Gyr = 1e9*yr
kmps = 1e5
kB = 1.38e-16
mH = 1.67e-24
G = 6.67e-8

# Fiducial parameters that we're not going to alter from case to case
Q = 1.0
phi_mp = 1.4
eta = 1.5
phi_Q = 2.0
epsff = 0.015
t_sfmax = 2.0*Gyr
fc = 1.
pmstar = 3000.0*kmps
sigma_th_wnm = 5.4*kmps
sigma_th_mol = 0.2*kmps

# Set of models
models = []
models.append(
    { 'name'     : 'Local dwarf',
      'r_in'     : 0.1*kpc,
      'r_out'    : 5*kpc,
      'sigma_g'  : 6.0*kmps,
      'vphi_out' : 60*kmps,
      'beta'     : 0.5,
      'fgQ'      : 0.5,
      'fgP'      : 0.5,
      'Z'        : 0.2
    })
models.append(
    { 'name'     : 'Local spiral',
      'r_in'     : 0.1*kpc,
      'r_out'    : 10*kpc,
      'sigma_g'  : 10.0*kmps,
      'vphi_out' : 200*kmps,
      'beta'     : 0.0,
      'fgQ'      : 0.5,
      'fgP'      : 0.5,
      'Z'        : 1.0
    })
models.append(
    { 'name'     : 'ULIRG',
      'r_in'     : 0.1*kpc,
      'r_out'    : 1*kpc,
      'sigma_g'  : 60.0*kmps,
      'vphi_out' : 250*kmps,
      'beta'     : 0.5,
      'fgQ'      : 0.5,
      'fgP'      : 0.5,
      'Z'        : 1.0
    })
models.append(
    { 'name'     : 'High-$z$',
      'r_in'     : 0.1*kpc,
      'r_out'    : 5*kpc,
      'sigma_g'  : 40.0*kmps,
      'vphi_out' : 200*kmps,
      'beta'     : 0.0,
      'fgQ'      : 0.5,
      'fgP'      : 0.5,
      'Z'        : 1.0
    })


# For each model, generate the solution
for m in models:

    # Solve on radial grid
    m['r'] = np.logspace(np.log10(m['r_in']),
                         np.log10(m['r_out']), 200)

    # Get angular velocity and gas surface density
    m['Omega'] = m['vphi_out'] / m['r'] * \
                 (m['r'] / m['r_out'])**m['beta']
    m['Sigma_g'] = np.sqrt(2.0*(m['beta']+1))*m['Omega'] * \
                   m['sigma_g'] / (np.pi * G * Q/m['fgQ'])

    # Get fsf from KMT+ model
    vphi = m['vphi_out'] * (m['r']/m['r_out'])**m['beta']
    rho_min = vphi**2 * (2.0*m['beta']+1) / (4.0*np.pi*G*m['r']**2)
    m['fsf'] = fH2KMTnew(m['Sigma_g'], 2.0*rho_min, m['Z'], cfac=fc)

    # Get sigma_sf
    sigma_th = m['fsf']*sigma_th_mol + (1.0-m['fsf'])*sigma_th_wnm
    m['sigma_th'] = sigma_th
    m['sigma_sf1'] = sigma_sf(sigma_th = sigma_th/kmps,
                              fsf = m['fsf'],
                              epsff = epsff, fgP = m['fgP'],
                              fgQ = m['fgQ'], eta = eta,
                              phimp = phi_mp, phiQ = phi_Q,
                              pmstar = pmstar/kmps,
                              Q = Q,
                              beta = m['beta'],
                              tsfmax = t_sfmax/Myr,
                              torb = 2.0*np.pi/m['Omega']/Myr)*kmps
    phi_nt = 1.0 - sigma_th**2 / m['sigma_g']**2
    m['phi_nt'] = phi_nt
    m['sigma_sf2'] = sigma_sf(sigma_th = sigma_th/kmps,
                              fsf = m['fsf'],
                              epsff = epsff, fgP = m['fgP'],
                              fgQ = m['fgQ'], eta = eta,
                              phimp = phi_mp, phiQ = phi_Q,
                              pmstar = pmstar/kmps,
                              Q = Q,
                              beta = m['beta'],
                              tsfmax = t_sfmax/Myr,
                              torb = 2.0*np.pi/m['Omega']/Myr,
                              phint = phi_nt)*kmps
    

    # Get star formation rate
    m['Sigma_sfr'] = m['fsf'] * m['Sigma_g'] * \
                     np.maximum(4.0*epsff*m['fgQ']/(np.pi*Q) *
                                np.sqrt(2.0*(1.0+m['beta']) /
                                        (3.0*phi_mp*m['fgP'])) *
                                m['Omega'], 1.0/t_sfmax)

    # Get mass inflow rate
    m['Mdot'] = 4.0*(1.0+m['beta']) * eta * phi_Q * phi_nt**1.5 * \
                m['fgQ']**2 * m['sigma_g']**3 * \
                (1.0 - m['sigma_sf2']/m['sigma_g']) / \
                ((1.0 - m['beta']) * G * Q**2)

# Plot
fig = plt.figure(1, figsize=(7,5))
plt.clf()
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

# Gas surface density
for i, m in enumerate(models):
    ax=plt.subplot(4,4,i+1)
    ax.plot(m['r']/kpc, np.log10(m['Sigma_g']/(msun/pc**2)))
    ax.set_xlim([0, m['r'][-1]/kpc])
    ax.set_ylim([0,4])
    ax.set_title(m['name'])
    if i != 0:
        ax.tick_params(labelleft='off')
    ax.tick_params(labelbottom='off')
    ax.set_xticks([0,m['r'][-1]/kpc/2, m['r'][-1]/kpc])
ax=plt.subplot(4,4,1)
ax.set_ylabel('$\log\,\Sigma_{\mathrm{g}}$\n[$M_\odot$ pc$^{-2}$]',
              labelpad=12)

# SFR surface density
for i, m in enumerate(models):
    ax=plt.subplot(4,4,i+5)
    ax.plot(m['r']/kpc, np.log10(m['Sigma_sfr']/(msun/pc**2/Myr)))
    ax.set_xlim([0, m['r'][-1]/kpc])
    ax.set_ylim([-5,2])
    if i != 0:
        ax.tick_params(labelleft='off')
    ax.tick_params(labelbottom='off')
    ax.set_xticks([0,m['r'][-1]/kpc/2, m['r'][-1]/kpc])
    ax.set_yticks([-4, -2, 0])
ax=plt.subplot(4,4,5)
ax.set_ylabel('$\log\,\dot{\Sigma}_{*}$\n[$M_\odot$ pc$^{-2}$ Myr$^{-1}$]')

# sigma_g / sigma_sf
for i, m in enumerate(models):
    ax=plt.subplot(4,4,i+9)
    ax.plot(m['r']/kpc, np.log10(m['sigma_g']/m['sigma_sf2']))
    ax.set_xlim([0, m['r'][-1]/kpc])
    ax.set_ylim([-0.5,1.5])
    if i != 0:
        ax.tick_params(labelleft='off')
    ax.tick_params(labelbottom='off')
    ax.set_xticks([0,m['r'][-1]/kpc/2, m['r'][-1]/kpc])
ax=plt.subplot(4,4,9)
ax.set_ylabel('$\log\,\sigma_{\mathrm{g}}/\sigma_{\mathrm{sf}}$',
              labelpad=28)

# Mdot
for i, m in enumerate(models):
    ax=plt.subplot(4,4,i+13)
    ax.plot(m['r']/kpc, np.log10(m['Mdot']/(msun/yr)))
    ax.set_xlim([0, m['r'][-1]/kpc])
    ax.set_ylim([-2.5,3.5])
    if i != 0:
        ax.tick_params(labelleft='off')
    ax.set_xlabel('$r$ [kpc]')
    ax.set_xticks([0,m['r'][-1]/kpc/2, m['r'][-1]/kpc])
    #if i != 0:
    #    ax.set_xticklabels(['', '{:3.1f}'.format(m['r'][-1]/kpc/2),
    #                        '{:3.1f}'.format(m['r'][-1]/kpc)])
ax=plt.subplot(4,4,13)
ax.set_ylabel('$\log\,\dot{M}$\n[$M_\odot$ yr$^{-1}$]')

plt.subplots_adjust(hspace=0, wspace=0.2, top=0.92, right=0.95, left=0.12)

plt.savefig('example_sol.pdf')
