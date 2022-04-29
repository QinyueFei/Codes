# %%
import astropy.constants as c
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.functional_models import Sersic1D
from astropy.stats import sigma_clipped_stats
from Physical_values.surface_density import *
from Dynamics.models import *
from scipy.misc import derivative

plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

# %%
# from Barolo_analyse.parameters import load_parameters
#from Dynamics.models import *
## Load the velocity profiles
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG0923/Barolo_fit/output/PG0923+129_well"
file = "/ringlog2.txt"
r_fit,rad_fit,vrot_fit,disp_fit,inc,pa,z0,xpos,ypos,vsys,vrad, evrot1_fit, evrot2_fit, edisp1_fit, edisp2_fit = np.genfromtxt(path+file,skip_header=1,usecols=(0,1,2,3,4,5,7,9,10,11,12,13,14,15,16),unpack=True) 

output_dir = "/home/qyfei/Desktop/Results/Dynamics/results"

## Load the asymmetric drift correction
asymdrift = np.loadtxt(path+"/asymdrift.txt")
vasy = asymdrift[:,1]

## Load the surface density 
RADIUS, SUM, MEAN, MEDIAN, STDDEV, MAD, NPIX, SURFDENS, ERR_SD, SURFDENS_FO, MSURFDENS, MSURFDENS2 = np.genfromtxt(path+"/densprof.txt", skip_header=1,usecols=(0,1,2,3,4,5,6,7,8,9,10,11),unpack=True)

path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG0923/Barolo_fit/output/PG0923+129"
vrot_fit, disp_fit, evrot, edisp = np.genfromtxt(path+"/vel_prof.txt", skip_header=0, usecols=(0,1,2,3),unpack=True)

L_fit = np.zeros(len(r_fit))
for i in range(len(r_fit)):
    L_fit[i] = np.nansum(SUM[:i+1])

# %%
## Estimate the rotation velocity contributed by molecular gas
redshift = 0.029
D_L = Planck15.luminosity_distance(redshift)
nu_obs = 230.58/(1+redshift)

LCO21 = 3.25e7 * L_fit * D_L.value**2/((1+redshift)**3*nu_obs**2)
LCO10 = LCO21/0.62
G = 4.302e-6
V_gas = np.sqrt(G*LCO10/r_fit)

# %%
from scipy.special import gammainc, gamma
convert = Planck15.arcsec_per_kpc_proper(0.029).value
log_Mb = 10.43
re = 1.01 / convert
n = 1.04

log_Md = np.log10(10**10.74 - 10**10.43)
log_Mtot = 10.74
rd = 12.39 / convert

log_Mdh = 13.5
log_fb = -1.75
con = 5.
a = 4.3

# %%
#v_b = V_b(r_test, Mb, re, n)
#v_d = V_d(r_test, Md, rd)
#v_dh = V_dh(r_test, Mdh, con)
#v_test = np.sqrt(v_b**2 + v_d**2 + v_dh**2)
#vcirc_fit = np.sqrt(vrot_fit[1:]**2 + vasy2[1:])#vasy[1:]**2)
#rse = 9.02/Planck15.arcsec_per_kpc_proper(0.061).value/1.68 #Convert to r_d
#sigma_s0 = np.power(10, 10.64)/(2*np.pi*(rse)**2)/1e6

# L_CO = np.power(10, 9.75)
redshift = 0.029
b_con = -0.101+0.026*redshift
a_con = 0.520+(0.905-0.520)*np.exp(-0.617*redshift**1.21)

# def sigmag(x):
#     L_bar = L0i*np.exp(-bni*((x/rei)**(1/ni)-1))
#     L_disk = L0o*np.exp(-bno*((x/reo)**(1/no)-1))
#     L_Gauss = A*np.exp(-x**2/(2*xstd**2))
#     return (L_disk + L_bar + L_Gauss)*alpha_CO

# def sigma_sd(x):
#     rse = rd/1.68
#     sigmas0 = np.power(10, log_Md)/(2*np.pi*(rse)**2)/1e6
#     return sigmas0*np.exp(-x/rse)

# def sigmat(x):
#     return sigmag(x) + sigma_sd(x)

## x: radius
## y: rotation velocity
## yerr: rms of rotation velocity
## z: velocity dispersion
## zerr: rms of velocity dispersion
## vasy: asymmetric drift correction

def log_likelihood(theta, x, y, yerr, vasy):
    log_Mb, re, n, log_Mtot, rd, log_Mdh, con = theta #Mb, re, n, Md, rd, 
    alpha_CO = 3.1 * 1.36
    # log_Mb = 10.43
    # log_Mtot = 10.74
    v_b = V_b(x, log_Mb, re, n)
    log_Md = np.log10(10**log_Mtot - 10**log_Mb)
    v_d = V_d(x, log_Md, rd)
    v_dh = V_dh(x, log_Mdh, con)
    v_g = V_gas[2:]*np.sqrt(alpha_CO)

    model2 = v_b**2 + v_d**2 + v_dh**2 + v_g**2 - vasy**2
    sigma2 = yerr ** 2
    model = np.sqrt(model2)
    lnL = -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2*np.pi*sigma2))
    return lnL

def log_prior(theta):
    log_Mb, re, n, log_Mtot, rd, log_Mdh, con = theta #Mb, re, n, Md, rd, 
    # log_Mb = 10.43
    # log_Mtot = 10.74
    M_baryon = 10**log_Mtot       # Total baryon mass
    fb = M_baryon/(M_baryon+10**log_Mdh)                       # The baryon fraction
    log_fb = np.log10(fb)
    log_con_theory = a_con + b_con*(log_Mdh - 12)       # Theoretical concentration
    log_con = np.log10(con)                         # Logrithmic concentration
    alpha_CO =3.1 * 1.36
    #if not (10.46<log_Mb<11.46 and 0.1<re<10.0 and 0.6<n<10.0 and 10.14<log_Md<11.14 and 5.<rd<15. and -2.25<log_fb<-1.3 and (log_con_theory-0.1)<log_con<(log_con_theory+0.1) and 0.0<alpha_CO<20): #
    ## constraints
    ## Prior knowledge
    ## logMb = 10.43\pm0.47, logMtot = 10.75\pm0.50
    ## reb = 

    if not (-2.25<log_fb<-1.3 and (log_con_theory-0.1)<log_con<(log_con_theory+0.1) and 0.<alpha_CO<20): 
        return -np.inf
    Mbmu = 10.43
    Mbsigma = 0.47
    lpMb = np.log(1./(np.sqrt(2*np.pi)*Mbsigma)) - 0.5*(log_Mb - Mbmu)**2/Mbsigma**2-np.log(Mbmu)
    remu = 0.607
    resigma = 0.030
    lpre = np.log(1./(np.sqrt(2*np.pi)*resigma)) - 0.5*(re - remu)**2/resigma**2-np.log(remu)
    nmu = 1.040
    nsigma = 0.050
    lpn = np.log(1./(np.sqrt(2*np.pi)*nsigma)) - 0.5*(n - nmu)**2/nsigma**2-np.log(nmu)
    Mtmu = 10.74
    Mtsigma = 0.50
    lpMt = np.log(1./(np.sqrt(2*np.pi)*Mtsigma)) - 0.5*(log_Mtot - Mtmu)**2/Mtsigma**2-np.log(Mtmu)
    rdmu = 7.441
    rdsigma = 0.228
    lpred = np.log(1./(np.sqrt(2*np.pi)*rdsigma)) - 0.5*(rd - rdmu)**2/rdsigma**2-np.log(rdmu)
    # alpha_CO_mu = 4.34
    # alpha_CO_sigma = 0.50
    # lp_alpha_CO = np.log(1./(np.sqrt(2*np.pi)*alpha_CO_sigma)) - 0.5*(alpha_CO - alpha_CO_mu)**2/alpha_CO_sigma**2-np.log(alpha_CO_mu)

    lp = lpre + lpn + lpred #+ lpMb + lpMt# +lp_alpha_CO#
    #return 0
    return lp

def log_probability(theta, x, y, yerr, vasy):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr, vasy)

# %%

log_Mb = 10.43
re = 0.607
n = 1.04
log_Mtot = 10.75
rd = 7.441
log_Mdh = 12.49
log_fb = -1.75
con = 7.24
# a = 4.58

r_fit_tot = r_fit[2:]
vrot_tot = vrot_fit[2:]
evrot_tot = evrot[2:]
vasy_tot = vasy[2:]


np.random.seed(42)
from scipy.optimize import minimize
nll = lambda *args: -log_likelihood(*args)
initial = np.array([log_Mb, re, n, log_Mtot, rd, log_Mdh, con]) + 0.1 * np.random.randn(7)#Mb, re, n, Md, rd, 
soln = minimize(nll, initial, args=(r_fit_tot, vrot_tot, evrot_tot, vasy_tot), method="Nelder-Mead")#, disp_tot, edisp_tot

print("Fitting result:")
log_Mb_ml, re_ml, n_ml, log_Mtot_ml, rd_ml, log_Mdh_ml, con_ml = soln.x #
print("log_Mb = {0:.3f}".format(log_Mb_ml))
print("re = {0:.3f}".format(re_ml))
print("n = {0:.3f}".format(n_ml))
print("log_Md = {0:.3f}".format(log_Mtot_ml))
print("rd = {0:.3f}".format(rd_ml))
print("log_Mdh = {0:.3f}".format(log_Mdh_ml))
print("con = {0:.3f}".format(con_ml))
# print("a = {0:.3f}".format(a_ml))

# %%
from multiprocessing import Pool
import emcee
#pos = [9.3, re, n, 9.5, rd, Mdh, con] + 1e-4 * np.random.randn(400, 7)
log_Mb = 10.28
re = 0.607
n = 1.04
log_Mtot = 11.21
rd = 7.44
log_Mdh = 12.95
log_fb = -1.75
con = 7.24
a = 7.74

r_fit_tot = r_fit[2:]
vrot_tot = vrot_fit[2:]
evrot_tot = evrot[2:]
vasy_tot = vasy[2:]

pos = [log_Mb, re, n, log_Mtot, rd, log_Mdh, con] + 1e-4 * np.random.randn(200, 7)

# Mb, re, n, Md, rd, 
#[Mb_ml, re_ml, n_ml, Md_ml, rd_ml, Mdh_ml] + 1e-4 * np.random.randn(32, 6)
nwalkers, ndim = pos.shape
output_dir = "/home/qyfei/Desktop/Results/Dynamics/results/PG0923/rotation_velocity/stellar_mass/B13/"

backname = "tutorial.h5"
backend = emcee.backends.HDFBackend(output_dir+backname)
backend.reset(nwalkers, ndim)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(r_fit_tot, vrot_tot, evrot_tot, vasy_tot), backend=backend) #, disp_tot, edisp_tot
    sampler.run_mcmc(pos, 500, progress=True)

# %%
#### Plot the fitting parameters space
fig, axes = plt.subplots(7, figsize=(10, 40), sharex=True)
samples = sampler.get_chain()
labels = ["logMb", "re", "n", "logMd", "rd", "logMdh", "con"] #Mb, re, n, Md, rd, 
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number")
#plt.savefig(output_dir+'step.png', bbox_inches='tight',dpi=300)

# %%

import h5py
import numpy as np
import matplotlib.pyplot as plt
import corner

output_dir = "/home/qyfei/Desktop/Results/Dynamics/results/PG0923/rotation_velocity/stellar_mass/B13/"

labels = ["\log M_b", "r_{e,b}", "n", "log M_{tot}", "r_{e,d}", "log M_{DM}","C"]#,"\\alpha_{CO}"]
#output_dir = "/home/qyfei/Desktop/Results/Dynamics/results/PG0050/rotation_velocity/CO_Ha_combine/no_constraints/"

with h5py.File(output_dir+"tutorial.h5", "r") as f:
    print(list(f.keys()))
f = h5py.File(output_dir+"tutorial.h5", "r")
accepted = f['mcmc']['accepted']
chain = f['mcmc']['chain']
log_prob = f['mcmc']['log_prob']
get_chain = np.reshape(chain[200:], (300*200, 7))

from IPython.display import display, Math
para_out = []
for i in range(len(get_chain[1])):
    #para_out.append(np.percentile(get_chain[:,i], [50])[0])
    mcmc = (np.percentile(get_chain[:,i], [16, 50, 84]))
    para_out.append(mcmc)
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))
    
# %%

log_Mb_distribution = get_chain[:,0]
log_Md_distribution = get_chain[:,3]
log_Mdh_distribution = get_chain[:,5]
con_distribution = get_chain[:,6]
# alpha_CO_distribution = get_chain[:,7]

# %%
para_out_determine = []
for i in range(len(get_chain[1])):
    para_distribution = get_chain[:, i]
    hists = plt.hist(para_distribution, align='mid', bins=20)

    nums = hists[0]
    vals_intr = hists[1]
    vals = hists[1][:-1] + np.diff(hists[1])/2

    N_max = np.where(nums == np.nanmax(nums))[0]
    FWHM_distribution = np.where(nums>=0.5*np.nanmax(nums))[0]
    N_lowlim, N_uplim = FWHM_distribution[0], FWHM_distribution[-1]

    para_map = vals[N_max][0]
    para_lowlim = vals[N_lowlim] - np.diff(hists[1])[0]/2
    para_uplim = vals[N_uplim] + np.diff(hists[1])[0]/2

    #plt.vlines(para_map, 0, 3e4, color='k')
    #plt.vlines(para_lowlim, 0, 3e4, color='k', ls=':')
    #plt.vlines(para_uplim, 0, 3e4, color='k', ls=':')
    #plt.ylim(0, 18000)

    para_vals = np.array([para_lowlim, para_map, para_uplim])
    para_out_determine.append(para_map)
    epara_vals = np.diff(para_vals)

    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(para_vals[1], epara_vals[0], epara_vals[1], labels[i])
    display(Math(txt))

# %%
labels = ["$\log M_\mathrm{b}$", "$r_\mathrm{e,b}$", "$n$", "$\log M_\mathrm{d}$", "$r_\mathrm{e,d}$", "$\log M_\mathrm{DM}$","c"]#,"$\\alpha_\mathrm{CO}$"]
#labels = ["$\log M_\mathrm{b}$", "$\log M_\mathrm{d}$", "$\log M_\mathrm{DM}$","c","$\\alpha_\mathrm{CO}$"]
nlabels = len(labels)
ns = np.array([0,1,2,3,4,5,6])
plt.rc('font', family='dejavuserif', size=15)

fig = corner.corner(get_chain[:, [0,1,2,3,4,5,6]], labels=labels, show_titles=True)

# Extract the axes
axes = np.array(fig.axes).reshape((nlabels, nlabels))

# Loop over the diagonal

for i in range(nlabels):
    ni = ns[i]
    ax = axes[i, i]
    ax.axvline(para_out[ni][1], color="k")
    ax.axvline(para_out[ni][0], color="k", ls=":")
    ax.axvline(para_out[ni][2], color="k", ls=":")
    
# Loop over the histograms
for yi in range(nlabels):
    for xi in range(yi):
        nyi, nxi = ns[yi], ns[xi]
        ax = axes[yi, xi]
        ax.axvline(para_out[nxi][1], color="k")
        ax.axvline(para_out[nxi][0], color="k", ls=":")
        ax.axvline(para_out[nxi][2], color="k", ls=":")
        
        ax.axhline(para_out[nyi][1], color="k")
        ax.axhline(para_out[nyi][0], color="k", ls=":")
        ax.axhline(para_out[nyi][2], color="k", ls=":")

        ax.plot(para_out[nxi][1], para_out[nyi][1], "ws")
        #ax.plot(paras_out[xi], paras_out[yi], "sr")

# plt.savefig(output_dir+'corner.pdf',bbox_inches='tight', dpi=300)

# %%
plt.rc('font', family='dejavuserif', size=25)

r_fit_tot = r_fit#[4:17]
vrot_tot = vrot_fit#[4:17]
evrot_tot = evrot#[4:17]
vasy_tot = vasy

parameters = np.array(para_out)[:,1]

alpha_CO = 3.1 * 1.36#parameters[5]
r_test = r_fit_tot#[7:17]
# asy0 = r_test/sigmag(r_test)*derivative(sigmag, r_test, dx=1e-8)   # Gas asymmetric drift correction
# asy1 = r_test/sigmat(r_test)*derivative(sigmat, r_test, dx=1e-8)   # Total asymmetric drift correction

plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
v_b = V_b(r_test, parameters[0], parameters[1], parameters[2]) #parameters[0], parameters[1], parameters[2]
log_Md = np.log10(10**parameters[3] - 10**parameters[0])
v_d = V_d(r_test, log_Md, parameters[4]) #parameters[3], parameters[4]
#log_Mdh = np.log10((10**parameters[0]+10**parameters[3]+parameters[7]*1.36*L_CO)*(1/10**parameters[5]-1))
v_dh = V_dh(r_test, parameters[5], parameters[6])
#v_g = V_gas*np.sqrt(parameters[7])


ax.plot(r_test, v_b, "C1", lw=2, label="Bulge")
ax.plot(r_test, v_d, "C2", lw=2, label="Disk")
ax.plot(r_test, v_dh, "C3", lw=2, label="DM")
ax.plot(r_test, V_gas[:]*np.sqrt(alpha_CO), "C4", lw=2, label="Gas")
#ax.plot(r_test, vs_in*np.sqrt(alpha_CO), "g:", lw=2)
#ax.plot(r_test, vs_out*np.sqrt(alpha_CO), "g--", lw=2)
#ax.plot(r_test, vs_p*np.sqrt(alpha_CO), "g-.", lw=2)

ax.plot(r_test, np.sqrt(v_b**2+v_d**2+v_dh**2+V_gas[:]**2*alpha_CO - vasy_tot**2), "Grey", lw=6, zorder=0, label='Total')

#ax.errorbar(rfit2D[5:], Vrot2D[5:], yerr=evrot2D[5:], fmt='ro', mfc='r', ms=8, mew=1, elinewidth=1, capsize=4, label=r'$\mathrm{H}\alpha$')
ax.errorbar(r_fit, vrot_tot, yerr=evrot, fmt='bo', mfc='b', ms=4, mew=1, elinewidth=1, capsize=5)
ax.fill_between(r_fit, vrot_fit-evrot, vrot_fit+evrot, color='b', alpha=0.3, zorder=3)

# ax.errorbar(r_fit[4:17], vrot_fit[4:17], yerr=[-evrot1_fit[4:17], evrot2_fit[4:17]], fmt='rs', mfc='r', ms=10, mew=1, elinewidth=1, capsize=5)

# ax.vlines(, 0, 400, color='k', ls=':')
ax.set_xlim(1e-4, r_fit[-1]+(r_fit[2]-r_fit[1])/2)
ax.set_ylim(-5, 335)
ax.grid()
ax.fill_between([0, 2*(r_fit[1]-r_fit[0])], [-500], [500], color='k', hatch='/', alpha=0.1)

ax.set_xlabel('Radius [kpc]')
ax.set_ylabel(r'$V_\mathrm{rot}$ [$\mathrm{km\,s^{-1}}$]')
plt.legend(fontsize=18, loc="lower right")
#plt.show()
#output_dir = "/home/qyfei/Desktop/Results/Dynamics/results/PG0050/rotation_velocity/CO_based/no_spiral/"

# plt.savefig(output_dir+'RC_fit_show.pdf', bbox_inches='tight', dpi=300)

# %%
output_dir = "/home/qyfei/Desktop/Results/Dynamics/results/PG0050/rotation_velocity/CO_based/"

documents = ["new_without_spiral/stellar_mass_constrained_one_sigma/", "new_without_spiral/stellar_mass_constrained_three_sigma/", "new_without_spiral/no_constraints/", "without_spiral/alpha_CO_constrained/", "without_spiral/alpha_CO_almost_fixed/"]

labels = ["\log M_b", "r_{e,b}", "n", "log M_d", "r_{e,d}", "log M_{DM}","C","\\alpha_{CO}"]
#output_dir = "/home/qyfei/Desktop/Results/Dynamics/results/PG0050/rotation_velocity/CO_Ha_combine/no_constraints/"
cases = ["A.1", "A.2", "C", "B.1", "B.2"]
colors = ["r", "C1", "C2", "grey", "black"]
linestyles = ["-", "-", "-.", "-", "-"]
hatches = ["/", "/", "/", "\\", "\\"]

bins = np.linspace(0., 6., 40)
dbins = np.diff(bins)[0]

plt.figure(figsize=(16, 8))
ax0, ax1 = plt.subplot(121), plt.subplot(122)

ax0.plot(r_test, v_b, "C1", lw=2, label="Bulge")
ax0.plot(r_test, v_d, "C2", lw=2, label="Disk")
ax0.plot(r_test, v_dh, "C3", lw=2, label="DM")
ax0.plot(r_test, v_g*np.sqrt(alpha_CO), "C4", lw=2, label="Gas")

ax0.plot(r_test, np.sqrt(v_b**2+v_d**2+v_dh**2+v_g**2*alpha_CO + disp_tot**2*(asy0+asy1)), "Grey", lw=6, zorder=0, label='Total')

ax0.errorbar(r_fit, vrot_fit, yerr=[-evrot1_fit, evrot2_fit], fmt='bo', mfc='b', ms=10, mew=1, elinewidth=1, capsize=5)
ax0.errorbar(r_fit[4:17], vrot_fit[4:17], yerr=[-evrot1_fit[4:17], evrot2_fit[4:17]], fmt='rs', mfc='r', ms=10, mew=1, elinewidth=1, capsize=5)

ax0.vlines(0.43, 0, 400, color='k', ls=':')
ax0.vlines(0.86, 0, 400, color='k', ls=':')
ax0.vlines(2.1, 0, 400, color='k', ls=':')

ax0.set_xlim(0, 3.0)
ax0.set_ylim(0, 360)
ax0.set_xlabel('Radius [kpc]')
ax0.set_ylabel(r'$V_\mathrm{rot}$ [$\mathrm{km\,s^{-1}}$]')

for i in range(4):
    #with h5py.File(output_dir+"tutorial.h5", "r") as f:
    #    print(list(f.keys()))
    f = h5py.File(output_dir+documents[i]+"tutorial.h5", "r")
    accepted = f['mcmc']['accepted']
    chain = f['mcmc']['chain']
    log_prob = f['mcmc']['log_prob']
    get_chain = np.reshape(chain[600:], (400*400, 8))
    alpha_CO_distribution = get_chain[:,7]
    alpha_CO_mean = np.percentile(alpha_CO_distribution, 50)
    #numbers, bins, plots = ax1.hist(alpha_CO_distribution, bins=20, density=True, lw=3, color=colors[i], histtype="step", label=cases[i])
    numbers = np.zeros(len(bins))
    for j in range(len(bins)):
        N = np.where((alpha_CO_distribution>=bins[j]-dbins) & (alpha_CO_distribution<bins[j]+dbins))
        #print(N)
        numbers[j] = len(N[0])
    
    ax1.plot(bins, numbers/np.nanmax(numbers), drawstyle = "steps", ls=linestyles[i], lw=3, color=colors[i], label=cases[i])
    ax1.vlines(alpha_CO_mean, 0, 2.0, color=colors[i], ls=":")

ax1.set_xlabel(r"$\alpha_\mathrm{CO}$ [$\mathrm{M_\odot\left(K\,km\,s^{-1}\,pc^2\right)^{-1}}$]")
ax1.set_ylabel("Normalized counts")
ax1.set_xlim(0, 4.8)
ax1.set_ylim(0, 1.1)

ax0.legend(loc="lower right")
ax1.legend()

#plt.savefig(output_dir+"RC_fit_alpha_CO_distribution_extra.pdf", bbox_inches="tight", dpi=300)

# %%
numbers, bins, plots = plt.hist(alpha_CO_distribution, bins=20)
# %%
