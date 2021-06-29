"""
Plot relationship between star formation and velocity dispersion
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii as asctab
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
import os.path as osp
import matplotlib.cm as cm
from sigma_sf import sigma_sf

# SFR to Ha conversion factor, Kennicutt & Evans (2012)
Ha_per_SFR = 10.**41.27*(u.erg/u.s)/(u.Msun/u.yr)

# Thermal + turbulent correction for Ha velocity dispersions 
sigma_Ha = 15.0*u.km/u.s

# Location of data
datadir = osp.join('data', 'kinematic')

# Data sets

# GHASP survey, Epinat+ 2008
data_epinat08 = asctab.read(osp.join(datadir, 'epinat08.txt'))
idx = data_epinat08['sigma'] != 99.0
sigma_epinat08 = data_epinat08['sigma'][idx] * u.km/u.s
lHa_epinat08 = data_epinat08['Ha_flux'][idx] * 1e-16*u.W/u.m**2 * \
               4.0*np.pi*(data_epinat08['D'][idx] * u.Mpc)**2
sfr_epinat08 = lHa_epinat08 / Ha_per_SFR

# Epinat+ 2009
data_epinat09 = asctab.read(osp.join(datadir, 'epinat09.txt'))
sigma_epinat09 = np.sqrt(
    (data_epinat09['sigma'] * u.km/u.s)**2 - sigma_Ha**2)
sfr_epinat09 = data_epinat09['SFR'] * u.Msun/u.yr

# Law+ 2009
data_law09 = asctab.read(osp.join(datadir, 'law09.txt'))
sigma_law09 = np.sqrt(
    (data_law09['sigma'] * u.km/u.s)**2 - sigma_Ha**2)
sfr_law09 = data_law09['SFR'] * u.Msun/u.yr

# SINS survey, Cresci+ 2009
# This is commented out because it has been superseded by the updated
# SINS-KMOS3D sample imported below
data_cresci09 = asctab.read(osp.join(datadir, 'cresci09.txt'), 
                            fill_values=[('...'), ('-999')])
sfr_cresci09 = data_cresci09['SFR']*u.Msun/u.yr
sigma_cresci09 = np.sqrt((data_cresci09['vdisp']*u.km/u.s)**2 - sigma_Ha**2)
fg_cresci09 = data_cresci09['fgas']

# Lemoine-Busserolle+ 2010
data_lb10 = asctab.read(osp.join(datadir, 'lemoine-busserolle10.txt'))
sigma_lb10 = np.sqrt(
    (data_lb10['sigma'] * u.km/u.s)**2 - sigma_Ha**2)
sfr_lb10 = data_lb10['SFR'] * u.Msun/u.yr

# Jones+ 2010, lensed galaxies
data_jones10 = asctab.read(osp.join(datadir, 'jones10.txt'))
sigma_jones10 = np.sqrt(
    (data_jones10['sigma'] * u.km/u.s)**2 - sigma_Ha**2)
sfr_jones10 = data_jones10['SFR'] * u.Msun/u.yr

# WiggleZ survey, Wisnioski+ 2011
data_wis11 = asctab.read(osp.join(datadir, 'wisnioski11.txt'))
sigma_wis11 = np.sqrt(
    (data_wis11['sigma'] * u.km/u.s)**2 - sigma_Ha**2)
sfr_wis11 = data_wis11['LHa'] * 1e41 * u.erg / u.s / Ha_per_SFR

# Ianjamasimanana+ 2012
data_ianj12 = asctab.read(osp.join(datadir, 'ianjamasimanana12.txt'),
                          fill_values=[('...'), ('-999')])
sfr_ianj12 = data_ianj12['SFR']*u.Msun/u.yr
sigma_ianj12 = data_ianj12['vdisp']*u.km/u.s
Mstar_ianj12 = 10.**data_ianj12['logM*']*u.Msun
MHI_ianj12 = 10.**data_ianj12['logMHI']*u.Msun
MH2_ianj12 = 10.**data_ianj12['logMH2']*u.Msun
fg_ianj12 = (MHI_ianj12+MH2_ianj12) / \
            (MHI_ianj12+MH2_ianj12+Mstar_ianj12)

# Stilp+ 2013
data_stilp13 = asctab.read(osp.join(datadir, 'stilp13.txt'))
sigma_stilp13 = data_stilp13['sigma'] * u.km/u.s
sfr_stilp13 = data_stilp13['SFR'] * u.Msun/u.yr

# DYNAMO survey, Green+ 2014
data_green14 = asctab.read(osp.join(datadir, 'green14.txt'))
sigma_green14 = np.sqrt(
    (data_green14['sigma'] * u.km/u.s)**2 - sigma_Ha**2)
sfr_green14 = data_green14['SFRHa'] * u.Msun/u.yr

# Moiseev+ 2015; note: already corrected for thermal broadening with a
# constant 9.1 km/s thermal velocity dispersion, but not turbulent
# broadening; we take out Moiseev's correction and then apply ours so
# that we treat these data consistently with the rest of them
data_moiseev15 = asctab.read(osp.join(datadir, 'moiseev15.csv'))
lHa_moiseev15 = 10.**data_moiseev15['log_LHa']*u.erg/u.s
sfr_moiseev15 = lHa_moiseev15 / Ha_per_SFR
sigma_moiseev15 = np.sqrt(
    (data_moiseev15['sigma']*u.km/u.s)**2
    + (9.1*u.km/u.s)**2
    - sigma_Ha**2)

# Di Teodoro+ 2016
data_di_teodoro16 = asctab.read(osp.join(datadir, 'di-teodoro16.txt'))
sigma_di_teodoro16 = np.sqrt(
    (data_di_teodoro16['sigma'] * u.km/u.s)**2 - sigma_Ha**2)
sfr_di_teodoro16 = data_di_teodoro16['SFR'] * u.Msun/u.yr

# Varidel+ 2016
data_varidel16 = asctab.read(osp.join(datadir, 'varidel16.txt'))
sigma_varidel16 = np.sqrt(
    (data_varidel16['sigma'] * u.km/u.s)**2 - sigma_Ha**2)
sfr_varidel16 = data_varidel16['SFR'] * u.Msun/u.yr

# SINS survey, Wuyts+ 2016
sinsdat = asctab.read(osp.join(datadir, 'wuyts16.csv'))
sigma_wuyts16 = np.sqrt(
    (np.array(sinsdat['col11'][2:], dtype=float) * u.km/u.s)**2
    - sigma_Ha**2)
sfr_wuyts16 = np.array(sinsdat['col25'][2:], dtype=float) * u.Msun/u.yr
fg_wuyts16 = np.array(sinsdat['col23'][2:], dtype=float)

# KROSS data from Johnson+ 2017 in prep
try:
    hdulist = fits.open(osp.join(datadir, 'johnson17.fits'))
    sigma_johnson17 = np.sqrt(
        (hdulist[1].data['SIGMA0'] * u.km/u.s)**2 - sigma_Ha**2)
    sfr_johnson17 = hdulist[1].data['SFR'] * u.Msun/u.yr
except:
    print("Apologies: you're seeing this message because "
          "the KROSS team have not yet submitted their "
          "paper, and have asked us to hold off on posting their "
          "tabulated results until they do. They anticipate "
          "submitting in June - July 2017, and the repository will "
          "be updated to include their data once they give us "
          "the go-ahead. Until then this code will produce the "
          "figure shown in the paper with the KROSS data "
          "removed. A copy of the plot in the submitted paper "
          "including the KROSS data is available from "
          "mark.krumholz@anu.edu.au.") 

# ULIRG compilation
data_ulirg = asctab.read(osp.join(datadir, 'ulirgs.txt'))
sigma_ulirg = data_ulirg['sigma'] * u.km/u.s
sfr_ulirg = data_ulirg['SFR'] * u.Msun/u.yr


# Plotting limits
sfrlim = [2e-4, 1e3]
sigmalim = [0, 175]


# Fiducial parameter values that we won't alter
sigma_wnm = 5.4
sigma_mol = 0.2
phimp = 1.4
epsff = 0.015
tsfmax = 2.0*u.Gyr
Qmin = 1.0
eta = 1.5
phiQ = 2.0
pmstar = 3000.*u.km/u.s

# Four sets of parameters: dwarf-like, spiral-like, ULIRG-like, hi-z
fsf = np.array([0.2, 0.5, 1.0, 1.0])
vphi = np.array([100., 220., 300., 200.])*u.km/u.s
fgP = np.array([0.9, 0.5, 1.0, 0.7])
fgQ = np.array([0.9, 0.5, 1.0, 0.7])
torb = np.array([100., 200., 5., 200.])*u.Myr
beta = np.array([0.5, 0.0, 0.5, 0.0])
phia = np.array([1., 1., 2., 3.])
sfr_cut = np.array([0.5, 5., -1., -1.])*u.Msun/u.yr
th_names = ['Local dwarf', 'Local spiral', 'ULIRG', 'High-$z$']

# Theoretical models
Qvec = np.logspace(4, 0, 10000)*Qmin
sigma_vec = np.linspace(1.0, 100.0, 10000)
sigma_fid = []
sfr_fid = []
sigma_nofb = []
sfr_nofb = []
sigma_fixed_eff = []
sfr_fixed_eff = []
sigma_fixed_Q = []
sfr_fixed_Q = []
for fsf_, vphi_, fgP_, fgQ_, torb_, beta_, phia_ in \
    zip(fsf, vphi, fgP, fgQ, torb, beta, phia):
    
    # Fiducial model
    sigma_th = sigma_wnm*(1.0 - fsf_) + sigma_mol*fsf_
    ssf = sigma_sf(sigma_th = sigma_th,
                   fsf = fsf_, beta=beta_,
                   torb=torb_.to(u.Myr))*u.km/u.s
    sigma1 = np.ones(Qvec.shape)*ssf
    sfr1 = np.sqrt(2.0/(1.0+beta_))*phia_*fsf_*fgQ_*vphi_**2*ssf / \
           (np.pi*const.G*Qvec) * np.maximum(
               np.sqrt(2.0*(1.0+beta_)/(3.0*fgP_*phimp)) *
               8.0*epsff*fgQ_/Qvec, torb_/tsfmax)
    sigma2 = sigma_vec*ssf
    sfr2 = np.sqrt(2.0/(1.0+beta_))*phia_*fsf_*fgQ_*vphi_**2*sigma2 / \
           (np.pi*const.G*Qmin) * np.maximum(
               np.sqrt(2.0*(1.0+beta_)/(3.0*fgP_*phimp)) *
               8.0*epsff*fgQ_/Qmin, torb_/tsfmax)
    # Note: some footwork is needed because astropy units does not
    # properly support numpy concatenation; we have to manually delete
    # the units, then reinsert them 
    sigma = np.concatenate((np.array(sigma1.to(u.km/u.s)),
                            np.array(sigma2.to(u.km/u.s)))) * \
                            u.km/u.s
    sfr = np.concatenate((np.array(sfr1.to(u.Msun/u.yr)),
                          np.array(sfr2.to(u.Msun/u.yr)))) * \
                          u.Msun/u.yr
    sigma_fid.append(sigma)
    sfr_fid.append(sfr)
    
    # No feedback model
    sigma1_nofb = ssf/sigma_vec[::-1]
    sfr1_nofb = np.sqrt(2.0/(1.0+beta_))*phia_*fsf_*fgQ_*vphi_**2* \
                sigma1_nofb / \
                (np.pi*const.G*Qmin) * np.maximum(
                    np.sqrt(2.0*(1.0+beta_)/(3.0*fgP_*phimp)) *
                    8.0*epsff*fgQ_/Qmin, torb_/tsfmax)
    sigma = np.concatenate((np.array(sigma1_nofb.to(u.km/u.s)),
                            np.array(sigma2.to(u.km/u.s)))) * \
                            u.km/u.s
    sfr = np.concatenate((np.array(sfr1_nofb.to(u.Msun/u.yr)),
                          np.array(sfr2.to(u.Msun/u.yr)))) * \
                          u.Msun/u.yr
    sigma_nofb.append(sigma)
    sfr_nofb.append(sfr)

    # No transport, fixed epsilon_ff model
    sigma = ssf * np.ones(sigma_vec.size)
    sfr = np.logspace(np.log10(sfrlim[0]), np.log10(sfrlim[1]),
                      sigma_vec.size)*u.Msun/u.yr
    sigma_fixed_eff.append(sigma)
    sfr_fixed_eff.append(sfr)

    # No transport, fixed Q model
    sigma = sigma_nofb[-1]
    phint = 1.0 - (sigma_th*u.km/u.s/sigma)**2
    phint[phint < 0] = 0.0
    sfr = 4.0*eta*np.sqrt(phimp*phint**3)*phiQ*phia_ / \
          (const.G*Qmin**2*pmstar) * fgQ_**2/fgP_ * \
          vphi_**2 * sigma**2
    sigma_fixed_Q.append(sigma)
    sfr_fixed_Q.append(sfr)


# Color and symbol scheme
cmaps = [cm.cool, cm.autumn]
symbols = ['o', '^', 's', '*', 'h', 'D', '8']
colors = ['#1f77bf', '#ff7f0e', '#2ca02c', '#d62728']

# Make plot
fig = plt.figure(1, figsize=(6.5,5.5))
plt.clf()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
try:
    # Depends on matplotlib version
    plt.rc('xtick', direction='in', top=True)
    plt.rc('ytick', direction='in', right=True)
except:
    pass

# Overall axes
ax = fig.add_subplot(1,1,1)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(top='off', bottom='off', left='off', right='off')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlabel(r'$\dot{M}_*$ [$M_\odot$ yr$^{-1}$]', labelpad=18)
ax.set_ylabel(r'$\sigma_{\mathrm{g}}$ [km s$^{-1}$]', labelpad=18)

##############################################

# Fiducial model + data
ax = fig.add_subplot(2,2,1)
p_near = []
lab_near = []
p_far = []
lab_far = []

# Fiducial model
p_theory = []
lab_theory = th_names
for i in range(len(sfr_fid)):
    if sfr_cut[i] > 0*u.Msun/u.yr:
        idx = np.argmax(sfr_fid[i] > sfr_cut[i])
        p,=plt.plot(sfr_fid[i][:idx].to(u.Msun/u.yr),
                    sigma_fid[i][:idx].to(u.km/u.s),
                    lw=3, color=colors[i], zorder=3)
        p_theory.append(p)
        idx1 = idx
        for j in range(40):
            idx2 = np.argmax(sfr_fid[i] > 10.**((j+1)/80.)*sfr_cut[i])
            plt.plot(sfr_fid[i][idx1:idx2].to(u.Msun/u.yr),
                     sigma_fid[i][idx1:idx2].to(u.km/u.s),
                     zorder=3,
                     color=colors[i],
                     lw=3.0*(1.0-(j+1)/40.))
            idx1 = idx2
    else:
        idx = np.argmin(sfr_fid[i] < -sfr_cut[i])
        p,=plt.plot(sfr_fid[i][idx:].to(u.Msun/u.yr),
                    sigma_fid[i][idx:].to(u.km/u.s),
                    lw=3, color=colors[i], zorder=3)
        p_theory.append(p)
        idx1 = idx
        for j in range(40):
            idx2 = np.argmin(sfr_fid[i] < -10.**(-(j+1)/80.)*sfr_cut[i])
            plt.plot(sfr_fid[i][idx2:idx1].to(u.Msun/u.yr),
                     sigma_fid[i][idx2:idx1].to(u.km/u.s),
                     zorder=3,
                     color=colors[i],
                     lw=3.0*(1.0-(j+1)/40.))
            idx1 = idx2

# High-z
try:
    p,=plt.plot(sfr_johnson17.to(u.Msun/u.yr),
                sigma_johnson17.to(u.km/u.s),
                marker = symbols[6],
                mfc = cmaps[1](6./6.),
                mec = 'k',
                ls = 'None')
    p_far.append(p)
    lab_far.append(r'KROSS')
except:
    pass
p,=plt.plot(sfr_wuyts16.to(u.Msun/u.yr),
            sigma_wuyts16.to(u.km/u.s),
            marker = symbols[5],
            mfc = cmaps[1](5./6.),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'SINS')
p,=plt.plot(sfr_di_teodoro16.to(u.Msun/u.yr),
            sigma_di_teodoro16.to(u.km/u.s),
            marker = symbols[4],
            mfc = cmaps[1](4./6.),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'Di Teodoro+ 2016')
p,=plt.plot(sfr_wis11.to(u.Msun/u.yr),
            sigma_wis11.to(u.km/u.s),
            marker = symbols[3],
            mfc = cmaps[1](3./6.),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'WiggleZ')
p,=plt.plot(sfr_jones10.to(u.Msun/u.yr),
            sigma_jones10.to(u.km/u.s),
            marker = symbols[2],
            mfc = cmaps[1](2./6.),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'Jones+ 2010')
p,=plt.plot(sfr_law09.to(u.Msun/u.yr),
            sigma_law09.to(u.km/u.s),
            marker = symbols[1],
            mfc = cmaps[1](1./6.),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'Law+ 2009')
#p,=plt.plot(sfr_cresci09.to(u.Msun/u.yr),
#            sigma_cresci09.to(u.km/u.s),
#            marker = symbols[1],
#            mfc = cmaps[1](1./6.),
#            mec = 'k',
#            ls = 'None')
#p_far.append(p)
#lab_far.append(r'SINS')
p,=plt.plot(sfr_epinat09.to(u.Msun/u.yr),
            sigma_epinat09.to(u.km/u.s),
            marker = symbols[0],
            mfc = cmaps[1](0.0),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'Epinat+ 2009')

# Nearby galaxies (z <~ 0.1)
p,=plt.plot(sfr_epinat08.to(u.Msun/u.yr),
            sigma_epinat08.to(u.km/u.s),
            marker = symbols[0],
            mfc = cmaps[0](0.0),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'GHASP (H$\alpha$)')
p,=plt.plot(sfr_green14.to(u.Msun/u.yr),
            sigma_green14.to(u.km/u.s),
            marker = symbols[1],
            mfc = cmaps[0](1./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'DYNAMO (H$\alpha$)')
p,=plt.plot(sfr_moiseev15.to(u.Msun/u.yr),
            sigma_moiseev15.to(u.km/u.s),
            marker = symbols[2],
            mfc = cmaps[0](2./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'Moiseev+ 2015 (H$\alpha$)')
p,=plt.plot(sfr_varidel16.to(u.Msun/u.yr),
            sigma_varidel16.to(u.km/u.s),
            marker = symbols[3],
            mfc = cmaps[0](3./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'Varidel+ 2016 (H$\alpha$)')
p,=plt.plot(sfr_ianj12.to(u.Msun/u.yr),
            sigma_ianj12.to(u.km/u.s),
            marker = symbols[4],
            mfc = cmaps[0](4./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'THINGS (H~\textsc{i})')
p,=plt.plot(sfr_stilp13.to(u.Msun/u.yr),
            sigma_stilp13.to(u.km/u.s),
            marker = symbols[5],
            mfc = cmaps[0](5./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'Stilp+ 2013 (H~\textsc{i})')
p,=plt.plot(sfr_ulirg.to(u.Msun/u.yr),
            sigma_ulirg.to(u.km/u.s),
            marker = symbols[6],
            mfc = cmaps[0](6./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'ULIRGs (CO, HCN)')

# Legends
leg1=plt.legend(p_near, lab_near, loc='center left',
                prop={"size":7}, title='Local Galaxies',
                numpoints=1)
leg1.get_title().set_fontsize('7')

# Adjust axes, add labels
plt.text(5e-4, 160, 'Transport+feedback',
         horizontalalignment='left',
         verticalalignment='top',
         fontdict={'size' : 12},
         bbox={'edgecolor' : 'black',
               'facecolor' : 'wheat',
               'linewidth' : 1,
               'boxstyle' : 'round' })
plt.xscale('log')
plt.xlim(sfrlim)
plt.ylim(sigmalim)
plt.setp(ax.get_xticklabels(), visible=False)


##############################################

# No feedback model + data
ax = fig.add_subplot(2,2,2)
p_near = []
lab_near = []
p_far = []
lab_far = []

# No feedback model
p_theory = []
lab_theory = th_names
for i in range(len(sfr_nofb)):
    if sfr_cut[i] > 0*u.Msun/u.yr:
        idx = np.argmax(sfr_nofb[i] > sfr_cut[i])
        p,=plt.plot(sfr_nofb[i][:idx].to(u.Msun/u.yr),
                    sigma_nofb[i][:idx].to(u.km/u.s),
                    lw=3, color=colors[i], zorder=3)
        p_theory.append(p)
        idx1 = idx
        for j in range(40):
            idx2 = np.argmax(sfr_nofb[i] > 10.**((j+1)/80.)*sfr_cut[i])
            plt.plot(sfr_nofb[i][idx1:idx2].to(u.Msun/u.yr),
                     sigma_nofb[i][idx1:idx2].to(u.km/u.s),
                     zorder=3,
                     color=colors[i],
                     lw=3.0*(1.0-(j+1)/40.))
            idx1 = idx2
    else:
        idx = np.argmin(sfr_nofb[i] < -sfr_cut[i])
        p,=plt.plot(sfr_nofb[i][idx:].to(u.Msun/u.yr),
                    sigma_nofb[i][idx:].to(u.km/u.s),
                    lw=3, color=colors[i], zorder=3)
        p_theory.append(p)
        idx1 = idx
        for j in range(40):
            idx2 = np.argmin(sfr_nofb[i] < -10.**(-(j+1)/80.)*sfr_cut[i])
            plt.plot(sfr_nofb[i][idx2:idx1].to(u.Msun/u.yr),
                     sigma_nofb[i][idx2:idx1].to(u.km/u.s),
                     zorder=3,
                     color=colors[i],
                     lw=3.0*(1.0-(j+1)/40.))
            idx1 = idx2

# High-z
try:
    p,=plt.plot(sfr_johnson17.to(u.Msun/u.yr),
                sigma_johnson17.to(u.km/u.s),
                marker = symbols[6],
                mfc = cmaps[1](6./6.),
                mec = 'k',
                ls = 'None')
    p_far.append(p)
    lab_far.append(r'KROSS')
except:
    pass
p,=plt.plot(sfr_wuyts16.to(u.Msun/u.yr),
            sigma_wuyts16.to(u.km/u.s),
            marker = symbols[5],
            mfc = cmaps[1](5./6.),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'SINS')
p,=plt.plot(sfr_di_teodoro16.to(u.Msun/u.yr),
            sigma_di_teodoro16.to(u.km/u.s),
            marker = symbols[4],
            mfc = cmaps[1](4./6.),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'Di Teodoro+ 2016')
p,=plt.plot(sfr_wis11.to(u.Msun/u.yr),
            sigma_wis11.to(u.km/u.s),
            marker = symbols[3],
            mfc = cmaps[1](3./6.),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'WiggleZ')
p,=plt.plot(sfr_jones10.to(u.Msun/u.yr),
            sigma_jones10.to(u.km/u.s),
            marker = symbols[2],
            mfc = cmaps[1](2./6.),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'Jones+ 2010')
p,=plt.plot(sfr_law09.to(u.Msun/u.yr),
            sigma_law09.to(u.km/u.s),
            marker = symbols[1],
            mfc = cmaps[1](1./6.),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'Law+ 2009')
#p,=plt.plot(sfr_cresci09.to(u.Msun/u.yr),
#            sigma_cresci09.to(u.km/u.s),
#            marker = symbols[1],
#            mfc = cmaps[1](1./6.),
#            mec = 'k',
#            ls = 'None')
#p_far.append(p)
#lab_far.append(r'SINS')
p,=plt.plot(sfr_epinat09.to(u.Msun/u.yr),
            sigma_epinat09.to(u.km/u.s),
            marker = symbols[0],
            mfc = cmaps[1](0.0),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'Epinat+ 2009')

# Nearby galaxies (z <~ 0.1)
p,=plt.plot(sfr_epinat08.to(u.Msun/u.yr),
            sigma_epinat08.to(u.km/u.s),
            marker = symbols[0],
            mfc = cmaps[0](0.0),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'GHASP (H$\alpha$)')
p,=plt.plot(sfr_green14.to(u.Msun/u.yr),
            sigma_green14.to(u.km/u.s),
            marker = symbols[1],
            mfc = cmaps[0](1./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'DYNAMO (H$\alpha$)')
p,=plt.plot(sfr_moiseev15.to(u.Msun/u.yr),
            sigma_moiseev15.to(u.km/u.s),
            marker = symbols[2],
            mfc = cmaps[0](2./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'Moiseev+ 2015 (H$\alpha$)')
p,=plt.plot(sfr_varidel16.to(u.Msun/u.yr),
            sigma_varidel16.to(u.km/u.s),
            marker = symbols[3],
            mfc = cmaps[0](3./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'Varidel+ 2016 (H$\alpha$)')
p,=plt.plot(sfr_ianj12.to(u.Msun/u.yr),
            sigma_ianj12.to(u.km/u.s),
            marker = symbols[4],
            mfc = cmaps[0](4./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'THINGS (H~\textsc{i})')
p,=plt.plot(sfr_stilp13.to(u.Msun/u.yr),
            sigma_stilp13.to(u.km/u.s),
            marker = symbols[5],
            mfc = cmaps[0](5./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'Stilp+ 2013 (H~\textsc{i})')
p,=plt.plot(sfr_ulirg.to(u.Msun/u.yr),
            sigma_ulirg.to(u.km/u.s),
            marker = symbols[6],
            mfc = cmaps[0](6./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'ULIRGs (CO, HCN)')

# Legends
leg2=plt.legend(p_far[::-1], lab_far[::-1], loc='center left',
                prop={"size":7}, title=r'High-$z$ Galaxies (All H$\alpha$)',
                numpoints=1)
leg2.get_title().set_fontsize('7')

# Adjust axes
plt.text(5e-4, 160, 'No-feedback',
         horizontalalignment='left',
         verticalalignment='top',
         fontdict={'size' : 12},
         bbox={'edgecolor' : 'black',
               'facecolor' : 'wheat',
               'linewidth' : 1,
               'boxstyle' : 'round' })
plt.xscale('log')
plt.xlim(sfrlim)
plt.ylim(sigmalim)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)

##############################################

# No transport, fixed e_ff model + data
ax = fig.add_subplot(2,2,3)
p_near = []
lab_near = []
p_far = []
lab_far = []

# Model
p_theory = []
lab_theory = th_names
for i in range(len(sfr_fixed_eff)):
    if sfr_cut[i] > 0*u.Msun/u.yr:
        idx = np.argmax(sfr_fixed_eff[i] > sfr_cut[i])
        p,=plt.plot(sfr_fixed_eff[i][:idx].to(u.Msun/u.yr),
                    sigma_fixed_eff[i][:idx].to(u.km/u.s),
                    lw=3, color=colors[i], zorder=3)
        p_theory.append(p)
        idx1 = idx
        for j in range(40):
            idx2 = np.argmax(sfr_fixed_eff[i] > 10.**((j+1)/80.)*sfr_cut[i])
            plt.plot(sfr_fixed_eff[i][idx1:idx2].to(u.Msun/u.yr),
                     sigma_fixed_eff[i][idx1:idx2].to(u.km/u.s),
                     zorder=3,
                     color=colors[i],
                     lw=3.0*(1.0-(j+1)/40.))
            idx1 = idx2
    else:
        idx = np.argmin(sfr_fixed_eff[i] < -sfr_cut[i])
        p,=plt.plot(sfr_fixed_eff[i][idx:].to(u.Msun/u.yr),
                    sigma_fixed_eff[i][idx:].to(u.km/u.s),
                    lw=3, color=colors[i], zorder=3)
        p_theory.append(p)
        idx1 = idx
        for j in range(40):
            idx2 = np.argmin(sfr_fixed_eff[i] < -10.**(-(j+1)/80.)*sfr_cut[i])
            plt.plot(sfr_fixed_eff[i][idx2:idx1].to(u.Msun/u.yr),
                     sigma_fixed_eff[i][idx2:idx1].to(u.km/u.s),
                     zorder=3,
                     color=colors[i],
                     lw=3.0*(1.0-(j+1)/40.))
            idx1 = idx2

# High-z
try:
    p,=plt.plot(sfr_johnson17.to(u.Msun/u.yr),
                sigma_johnson17.to(u.km/u.s),
                marker = symbols[6],
                mfc = cmaps[1](6./6.),
                mec = 'k',
                ls = 'None')
    p_far.append(p)
    lab_far.append(r'KROSS')
except:
    pass
p,=plt.plot(sfr_wuyts16.to(u.Msun/u.yr),
            sigma_wuyts16.to(u.km/u.s),
            marker = symbols[5],
            mfc = cmaps[1](5./6.),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'SINS')
p,=plt.plot(sfr_di_teodoro16.to(u.Msun/u.yr),
            sigma_di_teodoro16.to(u.km/u.s),
            marker = symbols[4],
            mfc = cmaps[1](4./6.),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'Di Teodoro+ 2016')
p,=plt.plot(sfr_wis11.to(u.Msun/u.yr),
            sigma_wis11.to(u.km/u.s),
            marker = symbols[3],
            mfc = cmaps[1](3./6.),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'WiggleZ')
p,=plt.plot(sfr_jones10.to(u.Msun/u.yr),
            sigma_jones10.to(u.km/u.s),
            marker = symbols[2],
            mfc = cmaps[1](2./6.),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'Jones+ 2010')
p,=plt.plot(sfr_law09.to(u.Msun/u.yr),
            sigma_law09.to(u.km/u.s),
            marker = symbols[1],
            mfc = cmaps[1](1./6.),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'Law+ 2009')
#p,=plt.plot(sfr_cresci09.to(u.Msun/u.yr),
#            sigma_cresci09.to(u.km/u.s),
#            marker = symbols[1],
#            mfc = cmaps[1](1./6.),
#            mec = 'k',
#            ls = 'None')
#p_far.append(p)
#lab_far.append(r'SINS')
p,=plt.plot(sfr_epinat09.to(u.Msun/u.yr),
            sigma_epinat09.to(u.km/u.s),
            marker = symbols[0],
            mfc = cmaps[1](0.0),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'Epinat+ 2009')

# Nearby galaxies (z <~ 0.1)
p,=plt.plot(sfr_epinat08.to(u.Msun/u.yr),
            sigma_epinat08.to(u.km/u.s),
            marker = symbols[0],
            mfc = cmaps[0](0.0),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'GHASP (H$\alpha$)')
p,=plt.plot(sfr_green14.to(u.Msun/u.yr),
            sigma_green14.to(u.km/u.s),
            marker = symbols[1],
            mfc = cmaps[0](1./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'DYNAMO (H$\alpha$)')
p,=plt.plot(sfr_moiseev15.to(u.Msun/u.yr),
            sigma_moiseev15.to(u.km/u.s),
            marker = symbols[2],
            mfc = cmaps[0](2./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'Moiseev+ 2015 (H$\alpha$)')
p,=plt.plot(sfr_varidel16.to(u.Msun/u.yr),
            sigma_varidel16.to(u.km/u.s),
            marker = symbols[3],
            mfc = cmaps[0](3./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'Varidel+ 2016 (H$\alpha$)')
p,=plt.plot(sfr_ianj12.to(u.Msun/u.yr),
            sigma_ianj12.to(u.km/u.s),
            marker = symbols[4],
            mfc = cmaps[0](4./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'THINGS (H~\textsc{i})')
p,=plt.plot(sfr_stilp13.to(u.Msun/u.yr),
            sigma_stilp13.to(u.km/u.s),
            marker = symbols[5],
            mfc = cmaps[0](5./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'Stilp+ 2013 (H~\textsc{i})')
p,=plt.plot(sfr_ulirg.to(u.Msun/u.yr),
            sigma_ulirg.to(u.km/u.s),
            marker = symbols[6],
            mfc = cmaps[0](6./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'ULIRGs (CO, HCN)')

# Legends
leg3=plt.legend(p_theory, lab_theory, loc='center left',
                prop={"size":7}, title=r'Model')
leg3.get_title().set_fontsize('7')

# Adjust axes
plt.text(5e-4, 160, 'No-transport, fixed $\epsilon_{\mathrm{ff}}$',
         horizontalalignment='left',
         verticalalignment='top',
         fontdict={'size' : 12},
         bbox={'edgecolor' : 'black',
               'facecolor' : 'wheat',
               'linewidth' : 1,
               'boxstyle' : 'round' })
plt.xscale('log')
plt.xlim(sfrlim)
plt.ylim(sigmalim)
# Note: the pause statement is needed due to a matplotlib bug
pause(0.01)
for l in plt.gca().get_xticklabels():
    if l.get_text() == '$10^{3}$':
        plt.setp(l, visible=False)
for l in plt.gca().get_yticklabels():
    if l.get_text() == '$175$':
        plt.setp(l, visible=False)

##############################################

# No transport, fixed Q model + data
ax = fig.add_subplot(2,2,4)
p_near = []
lab_near = []
p_far = []
lab_far = []

# Model
p_theory = []
lab_theory = th_names
for i in range(len(sfr_fixed_Q)):
    if sfr_cut[i] > 0*u.Msun/u.yr:
        idx = np.argmax(sfr_fixed_Q[i] > sfr_cut[i])
        p,=plt.plot(sfr_fixed_Q[i][:idx].to(u.Msun/u.yr),
                    sigma_fixed_Q[i][:idx].to(u.km/u.s),
                    lw=3, color=colors[i], zorder=3)
        p_theory.append(p)
        idx1 = idx
        for j in range(40):
            idx2 = np.argmax(sfr_fixed_Q[i] > 10.**((j+1)/80.)*sfr_cut[i])
            plt.plot(sfr_fixed_Q[i][idx1:idx2].to(u.Msun/u.yr),
                     sigma_fixed_Q[i][idx1:idx2].to(u.km/u.s),
                     zorder=3,
                     color=colors[i],
                     lw=3.0*(1.0-(j+1)/40.))
            idx1 = idx2
    else:
        idx = np.argmin(sfr_fixed_Q[i] < -sfr_cut[i])
        p,=plt.plot(sfr_fixed_Q[i][idx:].to(u.Msun/u.yr),
                    sigma_fixed_Q[i][idx:].to(u.km/u.s),
                    lw=3, color=colors[i], zorder=3)
        p_theory.append(p)
        idx1 = idx
        for j in range(40):
            idx2 = np.argmin(sfr_fixed_Q[i] < -10.**(-(j+1)/80.)*sfr_cut[i])
            plt.plot(sfr_fixed_Q[i][idx2:idx1].to(u.Msun/u.yr),
                     sigma_fixed_Q[i][idx2:idx1].to(u.km/u.s),
                     zorder=3,
                     color=colors[i],
                     lw=3.0*(1.0-(j+1)/40.))
            idx1 = idx2

# High-z
try:
    p,=plt.plot(sfr_johnson17.to(u.Msun/u.yr),
                sigma_johnson17.to(u.km/u.s),
                marker = symbols[6],
                mfc = cmaps[1](6./6.),
                mec = 'k',
                ls = 'None')
    p_far.append(p)
    lab_far.append(r'KROSS')
except:
    pass
p,=plt.plot(sfr_wuyts16.to(u.Msun/u.yr),
            sigma_wuyts16.to(u.km/u.s),
            marker = symbols[5],
            mfc = cmaps[1](5./6.),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'SINS')
p,=plt.plot(sfr_di_teodoro16.to(u.Msun/u.yr),
            sigma_di_teodoro16.to(u.km/u.s),
            marker = symbols[4],
            mfc = cmaps[1](4./6.),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'Di Teodoro+ 2016')
p,=plt.plot(sfr_wis11.to(u.Msun/u.yr),
            sigma_wis11.to(u.km/u.s),
            marker = symbols[3],
            mfc = cmaps[1](3./6.),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'WiggleZ')
p,=plt.plot(sfr_jones10.to(u.Msun/u.yr),
            sigma_jones10.to(u.km/u.s),
            marker = symbols[2],
            mfc = cmaps[1](2./6.),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'Jones+ 2010')
p,=plt.plot(sfr_law09.to(u.Msun/u.yr),
            sigma_law09.to(u.km/u.s),
            marker = symbols[1],
            mfc = cmaps[1](1./6.),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'Law+ 2009')
#p,=plt.plot(sfr_cresci09.to(u.Msun/u.yr),
#            sigma_cresci09.to(u.km/u.s),
#            marker = symbols[1],
#            mfc = cmaps[1](1./6.),
#            mec = 'k',
#            ls = 'None')
#p_far.append(p)
#lab_far.append(r'SINS')
p,=plt.plot(sfr_epinat09.to(u.Msun/u.yr),
            sigma_epinat09.to(u.km/u.s),
            marker = symbols[0],
            mfc = cmaps[1](0.0),
            mec = 'k',
            ls = 'None')
p_far.append(p)
lab_far.append(r'Epinat+ 2009')

# Nearby galaxies (z <~ 0.1)
p,=plt.plot(sfr_epinat08.to(u.Msun/u.yr),
            sigma_epinat08.to(u.km/u.s),
            marker = symbols[0],
            mfc = cmaps[0](0.0),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'GHASP (H$\alpha$)')
p,=plt.plot(sfr_green14.to(u.Msun/u.yr),
            sigma_green14.to(u.km/u.s),
            marker = symbols[1],
            mfc = cmaps[0](1./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'DYNAMO (H$\alpha$)')
p,=plt.plot(sfr_moiseev15.to(u.Msun/u.yr),
            sigma_moiseev15.to(u.km/u.s),
            marker = symbols[2],
            mfc = cmaps[0](2./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'Moiseev+ 2015 (H$\alpha$)')
p,=plt.plot(sfr_varidel16.to(u.Msun/u.yr),
            sigma_varidel16.to(u.km/u.s),
            marker = symbols[3],
            mfc = cmaps[0](3./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'Varidel+ 2016 (H$\alpha$)')
p,=plt.plot(sfr_ianj12.to(u.Msun/u.yr),
            sigma_ianj12.to(u.km/u.s),
            marker = symbols[4],
            mfc = cmaps[0](4./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'THINGS (H~\textsc{i})')
p,=plt.plot(sfr_stilp13.to(u.Msun/u.yr),
            sigma_stilp13.to(u.km/u.s),
            marker = symbols[5],
            mfc = cmaps[0](5./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'Stilp+ 2013 (H~\textsc{i})')
p,=plt.plot(sfr_ulirg.to(u.Msun/u.yr),
            sigma_ulirg.to(u.km/u.s),
            marker = symbols[6],
            mfc = cmaps[0](6./6.),
            mec = 'k',
            ls = 'None')
p_near.append(p)
lab_near.append(r'ULIRGs (CO, HCN)')

# Adjust axes
plt.text(5e-4, 160, 'No-transport, fixed $Q$',
         horizontalalignment='left',
         verticalalignment='top',
         fontdict={'size' : 12},
         bbox={'edgecolor' : 'black',
               'facecolor' : 'wheat',
               'linewidth' : 1,
               'boxstyle' : 'round' })
plt.xscale('log')
plt.xlim(sfrlim)
plt.ylim(sigmalim)
plt.setp(ax.get_yticklabels(), visible=False)

# Adjust axes
plt.subplots_adjust(hspace=0, wspace=0, top=0.95, right=0.95)

# Save
plt.savefig('sfrvdisp.pdf')

