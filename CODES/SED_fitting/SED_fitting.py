# %%
from spectral_cube.io.core import normalize_cube_stokes
#from Analysis.detection import F
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import Planck15
plt.clf()
plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
# %%
def blackbody(T_, nu_):
    nu = nu_
    T = T_
    B = 2*c.h*nu**3/c.c**2/(np.exp(c.h*nu/(c.k_B*T))-1)
    return B.to('Jy')

def dust(nu_, Td_, b_):
    # modified blackbody radiation
    nu = nu_*u.GHz
    T = Td_*u.K
    b = b_
    #n = n_
    nu0 = (2.82*c.k_B/c.h*T).to('GHz')
    #kappa = 
    #M = 
    B = blackbody(T, nu) * (1-np.exp(-np.power((nu/nu0).si,b)))#*(A/S).si
    return (B.to('mJy')).value

def ff(nu_, EM_):
    # free-free radiation
    nu, Te = nu_*u.GHz, 1e4*u.K
    #n = n_
    # free-free emission
    EM = 10**EM_*u.Unit('cm^-5')
    #EM = 1e25*u.Unit('cm^-5')
    g_ff = np.log(np.exp(5.960-np.sqrt(3)/np.pi*np.log(nu.value*(Te.value/1e4)**(-3/2)))+np.exp(1))
    tau_ff = (4/3*(2*np.pi/3)**(1/2)*(c.e.gauss)**6*g_ff*EM/(c.m_e*c.k_B*Te)**(3/2)/c.c/nu**2).si
    B = blackbody(Te, nu)*(1-np.exp(-tau_ff))#*(A/S).si
    return B.to('mJy').value

def syn(nu_, a_, b_):
    # synchrotron
    nu, a, b = nu_, a_, b_#nu_*u.GHz, a_
    #B = 4.91/(1+0.06115)*u.mJy*(nu/1.46/u.GHz/(1+0.06115)).si**(-a)#*np.exp(-tau_ff)
    f = np.power(10, b)*np.power(nu, a)
    return f#B.to('mJy').value

def cont(nu_, Td_, b_, EM_, a_):
    Td, b, EM, a = Td_, b_, EM_, a_
    #n = n_
    nu = nu_
    B_d = dust(nu, Td, b)
    B_ff = ff(nu, EM)
    B_syn = syn(nu, a)
    return B_d+B_ff+B_syn

def MBB(nu_, Td_, Md_, b_):
    nu = nu_*u.GHz
    T = Td_*u.K
    Md = np.power(10, Md_)*u.Msun
    b = b_
    kappa0 = 1.92*u.Unit('cm^2/g')
    nu0 = (c.c/350/u.um).to('GHz')
    #kappa0 = 1.92*u.Unit('cm^2/g')
    #nu0 = (c.c/350/u.um).to('GHz')
    
    epsilon = ((1)**2/DL**2*Md*kappa0*np.power(nu/nu0, b)).si.value
    B = (blackbody(T, nu) * epsilon).to('mJy')
    return B.value

# %%
## IR Estimation with SED fitting
## For F08238+0752

z = 0.330
#flux = np.array([1.515e-01, 1.803e-01, 2.445e-01, 7.888e-01, 0.229e-03])*(1+z)*1e3 # F08238+0752
#eflux = np.array([0.12*1.515e-01, 0.12*1.803e-01, 0.12*2.445e-01, 0.12*7.888e-01, 0.071e-03])*(1+z)*1e3

flux = np.array([1.143e-01, 1.235e-01, 0.3316, 0.5156, 0.856e-03])*(1+z)*1e3 # F08542+0752
eflux = np.array([0.12*1.143e-01, 0.12*1.235e-01, 0.12*0.3316, 0.12*0.5156, 0.129e-03])*(1+z)*1e3

#flux = np.array([1.110e-01, 2.365e-01, 2.094e-01, 6.905e-01, 0.136e-03])*(1+z)*1e3 # F13403-0038
#eflux = np.array([0.12*9.760e-02, 0.12*9.247e-02, 0.12*3.073e-01, 0.12*5.710e-01, 0.043e-03])*(1+z)*1e3 # F14167+4247

lamb = np.array([12, 25, 60, 100, 866.95*(1+z)])/(1+z)
nu = (c.c/lamb/u.um).to('GHz').value
uplims = np.array([1,1,0,0,0])

# %%
from astropy.cosmology import Planck15
DL = Planck15.luminosity_distance(z)
freq = np.logspace(1, 5, 1000)
f = ((c.c/lamb[4]/u.Unit('um')).to("GHz")).value
norm_est_NOEMA = flux[4]/dust(f, 47, 1.6)
flux_est_NOEMA = norm_est_NOEMA * dust(freq, 47, 1.6)#MBB(freq, 47, 8.2, 1.6)

f = ((c.c/lamb[3]/u.Unit('um')).to("GHz")).value
norm_est_IRAS = flux[3]/dust(f, 47, 1.6)
flux_est_IRAS = norm_est_IRAS * dust(freq, 47, 1.6)

plt.figure(figsize=(8,6))
ax = plt.subplot(111)
ax.errorbar(nu, flux, yerr = eflux, uplims=uplims, fmt='ko', mfc='none', ms=0, capsize=10)
ax.plot(freq, flux_est_NOEMA, 'k-', lw=3, alpha=0.7, label="NOEMA")
ax.plot(freq, flux_est_IRAS, 'k--', lw=2, alpha=0.5, label="IRAS")

ax.loglog()
ax.set_xlabel('rest frequency [GHz]')
ax.set_ylabel('rest flux [mJy]')
ax.set_xlim(1e2, 1e5)
ax.set_ylim(1e-1, 5e3)
ax.vlines(345.8, 1e-5, 1e5, color='b', linestyle='--')

plt.legend()

#plt.savefig('/home/qyfei/Desktop/Codes/Result/NOEMA_detection/SED_fitting/F14167+4247_SED.pdf', bbox_inches='tight', dpi=300)

# %%
FIR_wavelength = np.linspace(8, 1000, 10000)*u.um
FIR_frequency = (c.c/FIR_wavelength).to('GHz')

FIR_flux = norm_est_NOEMA * dust(FIR_frequency.value, 47, 1.6)
from scipy.integrate import trapz

L_FIR = trapz(FIR_flux*u.Unit('mJy'), FIR_frequency)*4*np.pi*DL**2

np.log10(-(L_FIR.to('L_sun')).value)
# %%
def log_likelihood(theta, x, y, yerr):
    Td, Md, b = theta
    model = MBB(x, Td, Md, b)
    
    sigma2 = yerr ** 2# + model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

# %%
from scipy.optimize import minimize

test_nu = nu[2:]#FIR_frequency.value
test_f = flux[2:]
test_ferr = eflux[2:]

np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([30, 8., 1.8]) + 0.1 * np.random.randn(3)
soln = minimize(nll, initial, args=(test_nu, test_f, test_ferr), method='Nelder-Mead')
Td_ml, Md_ml, b_ml = soln.x

print("Maximum likelihood estimates:")
print("Td = {0:.3f}".format(Td_ml))
print("Md = {0:.3f}".format(Md_ml))
#print("A = {0:.3f}".format(A_ml))
print("b = {0:.3f}".format(b_ml))
#print("norm = {0:.3f}".format(norm_ml))

# %%
def log_prior(theta):
    Td, Md, b = theta
    if 20.0<Td<100.0 and 5.0<Md<10.0 and 0.0<b<3.0:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

from multiprocessing import Pool
import emcee

pos = soln.x + 1e-4 * np.random.randn(50, 3)
nwalkers, ndim = pos.shape
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(test_nu, test_f, test_ferr), pool=pool)
    sampler.run_mcmc(pos, 5000, progress=True)

# %%
## Check fitting result
fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = [r"$T_d$", r"$M_d$", r"$\beta$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

# %%
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)

import corner

fig = corner.corner(
    flat_samples, labels=labels, truths=[Td_ml, Md_ml, b_ml]
)
#plt.savefig('Result/ot_corner.pdf', bbox_inches='tight', dpi=300)

# %%

from IPython.display import display, Math

for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))

# %%
test_x = freq
inds = np.random.randint(len(flat_samples), size=100)

plt.figure(figsize=(8,6))
ax = plt.subplot(111)

for ind in inds:
    sample = flat_samples[ind]
    #plt.plot(test_x, test_dust(test_x, sample[0], sample[1], sample[2]), "C1", alpha=0.1)
    ax.plot(test_x, MBB(test_x, sample[0], sample[1], sample[2]), "C1", alpha=0.1)

ax.errorbar(nu, flux, yerr = eflux, uplims=uplims, fmt='ko', mfc='none', ms=0, capsize=10, zorder=3)

ax.plot(freq, flux_est_NOEMA, 'k-.', lw=2, alpha=0.7, label="NOEMA")
ax.plot(freq, flux_est_IRAS, 'k--', lw=2, alpha=0.5, label="IRAS")
ax.plot(freq, MBB(freq, 46.046, 8.448, 1.650), 'k-', lw=3, alpha=0.9, label="FIT")

ax.loglog()
ax.set_xlabel('rest frequency [GHz]')
ax.set_ylabel('rest flux [mJy]')
ax.set_xlim(1e2, 1e5)
ax.set_ylim(1e-1, 5e3)
ax.vlines(345.8, 1e-5, 1e5, color='b', linestyle='--')

plt.legend()
#plt.savefig("/home/qyfei/Desktop/Codes/Result/NOEMA_detection/SED_fitting/F08542+1920_SED.pdf", bbox_inches='tight', dpi=300)

# %%
FIR_wavelength = np.linspace(8, 1000, 10000)*u.um
FIR_frequency = (c.c/FIR_wavelength).to('GHz')

FIR_flux = MBB(FIR_frequency.value, 46.046, 8.448, 1.650)

from scipy.integrate import trapz

L_FIR = trapz(FIR_flux*u.Unit('mJy'), FIR_frequency)*4*np.pi*DL**2

np.log10(-(L_FIR.to('L_sun')).value)

# %%
# PG0050
## Load spectral energy distribution

redshift = 0.06115
rest = 1+redshift

DL = Planck15.luminosity_distance(redshift)

nu_VLA = np.array([1.46, 4.8, 8.4, 15.0]) * rest
f_VLA = np.array([4.91, 2.41, 1.15, 1.06])/rest
ef_VLA = np.array([0.12, 0.12, 0.17, 0.31])/rest

# mm-submm, far-IR emission, Tan+2019, this work, Hughes+1993
nu_ALMA = np.array([108.644, 224.937]) * rest#*u.GHz#(c.c/(np.array([0.13,0.3,6,20])*u.cm)).to('GHz')
f_ALMA_total = np.array([0.492, 0.994])/rest#1.155*u.mJy
ef_ALMA = np.array([0.078, 0.025])#0.056*u.mJy

nu_JCMT = np.array([374.74, 666.21]) * rest
f_JCMT = np.array([18.4, 225])/rest
ef_JCMT = np.array([4.4, 75])

# middle infrared flux, detected by IRAS, Spinoglio+2002

lam_PACS = np.array([70, 100, 160, 250, 350, 500])*u.um
nu_PACS = (c.c/lam_PACS).to('GHz') * rest
f_PACS = np.array([2238.35, 2384.70, 1904.58, 755.94, 309.96, 127.40])/rest
ef_PACS = np.array([6.40, 5.77, 23.79, 14.02, 10.49, 11.210])

# VLA + JCMT + Herschel
test_nu = np.array([1.46, 4.8, 8.4, 15.0, 374.74, 666.21, 3375.78096297, 2109.86310186, 1350.31238519, 964.50884656,  675.15619259]) * rest
test_f = np.array([4.91, 2.41, 1.15, 1.06, 18.4, 225, 2384.70, 1904.58, 755.94, 309.96, 127.40]) / rest
test_ferr = np.array([0.12, 0.12, 0.17, 0.31, 4.4, 75, 5.77, 23.79, 14.02, 10.49, 11.210])

# Total flux, including two ALMA results
nu_rest = np.array([1.46, 4.8, 8.4, 15.0, 108.644, 224.937, 374.74, 666.21, 599.58, 856.55, 1199.17, 1873.70, 2997.92]) * rest
f_rest = np.array([4.91, 2.41, 1.15, 1.06, 0.492, 0.994, 18.4, 225, 127.40, 309.96, 755.94, 1904.58, 2384.70])/rest
ef_rest = np.array([0.12, 0.12, 0.17, 0.31, 0.078, 0.025, 4.4, 75, 11.21, 10.49, 14.02, 23.79, 5.77])

# %%
def log_likelihood(theta, x, y, yerr):
    a, b, Td, Md, beta = theta
    model = syn(x, a, b) + MBB(x, Td, Md, beta)
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2*np.pi*sigma2))

from scipy.optimize import minimize

np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([-0.67, 0.8, 30, 8, 1.6]) + 0.1 * np.random.randn(5)
soln = minimize(nll, initial, args=(test_nu, test_f, test_ferr), method="Nelder-Mead")
a_ml, b_ml, Td_ml, Md_ml, beta_ml = soln.x

print("Maximum likelihood estimates:")
print("a = {0:.3f}".format(a_ml))
print("b = {0:.3f}".format(b_ml))
#print("EM = {0:.3f}".format(EM_ml))
print("Td = {0:.3f}".format(Td_ml))
print("Md = {0:.3f}".format(Md_ml))
print("beta = {0:.3f}".format(beta_ml))

# %%
x_mm = np.logspace(0, 5, 10000)
plt.figure(figsize=(8, 8))
plt.errorbar(nu_ALMA, f_ALMA_total, yerr=ef_ALMA, fmt='ko', mfc='none', ms=10, capsize=5, label='ALMA')
plt.errorbar(nu_JCMT, f_JCMT, yerr=ef_JCMT, fmt='k>', mfc='none', ms=10, capsize=5, label='JCMT')
plt.errorbar(nu_VLA, f_VLA, yerr=ef_VLA, fmt='k<', mfc='none', ms=10, capsize=5, label='VLA')
plt.errorbar(nu_PACS.value, f_PACS, yerr=ef_PACS, fmt='ks', mfc='none', ms=10, capsize=5, label='Herschel')
plt.plot(x_mm, syn(x_mm, a_ml, b_ml) + MBB(x_mm, Td_ml, Md_ml, beta_ml), c='k', ls='-', lw=3, alpha=0.5, label='Herschel+JCMT+ALMA fit')

plt.errorbar(nu_ALMA[1], 1.50, yerr=0.20, fmt='k*', mfc='none', ms=10, capsize=5, label='ACA')
plt.loglog()
plt.xlim(1.1, 1.5e4)
plt.ylim(1.001e-1,5e4)
plt.xlabel('Frequency [GHz]')
plt.ylabel('Flux [mJy]')
#l1 = plt.legend(lines1, labels1, loc=(0.01, 0.72), fontsize=12, frameon=False)
#l2 = plt.legend(lines2, labels2, loc=(0.35, 0.72), fontsize=12, frameon=False)
plt.legend(loc='upper left', fontsize=12, frameon=False)

# %%
def log_prior(theta):
    a, b, Td, Md, beta = theta
    if -10<a<10 and -10<b<10 and 20.0<Td<100.0 and 5.0<Md<10.0 and 0.0<b<3.0: #0.1<A<10.0 and 
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

output_dir = "/home/qyfei/Desktop/Results/SED_fitting/PG0050/"
pos = soln.x + 1e-4 * np.random.randn(100, 5)
backname = "tutorial.h5"
nwalkers, ndim = pos.shape
backend = emcee.backends.HDFBackend(output_dir+backname)
backend.reset(nwalkers, ndim)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(test_nu, test_f, test_ferr), pool=pool, backend=backend)
    sampler.run_mcmc(pos, 1000, progress=True)

# %%
fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["a", "b", "Td", "Md", "beta"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

# %%
import h5py
import corner

f = h5py.File(output_dir+"tutorial.h5", "r")
accepted = f['mcmc']['accepted']
chain = f['mcmc']['chain']
log_prob = f['mcmc']['log_prob']

get_chain = np.reshape(chain[600:], (400*100, 5))

from IPython.display import display, Math
para_out = []
for i in range(len(get_chain[1])):
    para_out.append(np.percentile(get_chain[:,i], [50]))
    mcmc = (np.percentile(get_chain[:,i], [16, 50, 84]))
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))

# %%
fig = corner.corner(
    get_chain, labels=labels, truths=[para_out[0],para_out[1],para_out[2],para_out[3],para_out[4]]
)
#plt.savefig('Result/corner.pdf', bbox_inches='tight', dpi=300)

# %%
from IPython.display import display, Math
para = []

x_IR = np.linspace(1000, 8, 10000)*u.um
nu_IR = (c.c/x_IR).to('GHz')
f_IR = np.zeros(len(get_chain))
from scipy.integrate import trapz

for i in range(len(get_chain)):
    a, b, Td, Md, beta = get_chain[i]
    f_IR[i] = trapz(syn(nu_IR.value, a, b) + MBB(nu_IR.value, Td, Md, beta), nu_IR.value)

# %%
LIR = (np.percentile(f_IR, [16, 50, 84])*u.Unit('mJy')*u.GHz*4*np.pi*DL**2).to('erg/s')
LIR_ = np.diff(LIR)
logLIR = np.log10(LIR.value)
logLIRer = np.log10(LIR_.value)
txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
txt = txt.format(logLIR[1], logLIRer[0], logLIRer[1])
display(Math(txt))

# %%

for i in range(ndim):
    mcmc = np.percentile(get_chain[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))
    para.append(mcmc[1])

# %%

inds = np.random.randint(len(get_chain), size=100)

plt.figure(figsize=(8,8))
for ind in inds:
    sample = get_chain[ind]
    plt.plot(x_mm, syn(x_mm, sample[0], sample[1]), "r", alpha=0.1)
    plt.plot(x_mm, MBB(x_mm, sample[2], sample[3], sample[4]), "y", alpha=0.1)
    plt.plot(x_mm, syn(x_mm, sample[0], sample[1])+MBB(x_mm, sample[2], sample[3], sample[4]), "b", alpha=0.1)

plt.errorbar(nu_ALMA, f_ALMA_total, yerr=ef_ALMA, fmt='ko', mfc='none', ms=10, capsize=5, label='ALMA', zorder=3)
plt.errorbar(nu_JCMT, f_JCMT, yerr=ef_JCMT, fmt='k>', mfc='none', ms=10, capsize=5, label='JCMT', zorder=3)
plt.errorbar(nu_VLA, f_VLA, yerr=ef_VLA, fmt='k<', mfc='none', ms=10, capsize=5, label='VLA', zorder=3)
plt.errorbar(nu_PACS.value, f_PACS, yerr=ef_PACS, fmt='ks', mfc='none', ms=10, capsize=5, label='Herschel', zorder=3)
plt.errorbar(nu_ALMA[1], 1.50, yerr=0.20, fmt='k*', mfc='none', ms=10, capsize=5, label='ACA')

plt.plot(x_mm, syn(x_mm, para[0], para[1]), "r", lw=1, label='Syn')
plt.plot(x_mm, MBB(x_mm, para[2], para[3], para[4]), 'y', lw=1, label='MBB')
plt.plot(x_mm, syn(x_mm, para[0], para[1])+MBB(x_mm, para[2], para[3], para[4]), "b", label='Model')

plt.legend()
plt.xlim(1,2e4)
plt.ylim(1e-1, 5e3)
plt.loglog()
plt.xlabel("Rest Frequency [GHz]")
plt.ylabel("Rest Flux [mJy]")
#plt.savefig("/home/qyfei/Desktop/Results/SED_fitting/PG0050/SED_fitting.pdf", bbox_inches='tight', dpi=300)

# %%
SFR = 3e-44*LIR.value
txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
txt = txt.format(SFR[1], np.diff(SFR)[0], np.diff(SFR)[1], "SFR")
display(Math(txt))

# %%
f_exp = MBB(230.58, 31.72, 7.992, 1.901)
f_cen = 0.994
SFR_cen = f_cen/f_exp*SFR
SFR_cen

Sigma_cen = SFR_cen*u.Unit("M_sun/yr")/0.1818/u.Unit("kpc^2")

# %%
