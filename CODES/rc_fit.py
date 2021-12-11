# %%
import numpy as np
import matplotlib.pyplot as plt
from Dynamics.models import *
from Barolo_analyse.parameters import load_parameters
from scipy.special import gammaincinv
# %%
plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

# %%
path = '/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/'
folder = "Barolo_fit/output/PG0050+124_best/"
file = "ringlog2.txt"
#path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/ULIRG/IRAS_13120-5453/Working/"
#folder = "/Barolo_fit/CO21/output/IRAS_13120-5453/"
#file = "ringlog2.txt"
r_fit,rad_fit,vrot_fit,disp_fit,inc,pa,z0,xpos,ypos,vsys,vrad, evrot1_fit, evrot2_fit, edisp1_fit, edisp2_fit = np.genfromtxt(path+folder+file,skip_header=1,usecols=(0,1,2,3,4,5,7,9,10,11,12,13,14,15,16),unpack=True) 
output_dir = "/home/qyfei/Desktop/Codes/CODES/Dynamics/results/PG0050/prior/"
parameters = load_parameters(path, folder, file)
r_fit, rad_fit, vrot_fit, evrot1_fit, evrot2_fit, vdisp_fit, edisp1_fit, edisp2_fit, inc_fit, pa_fit, vrad_fit, vsys_fit, vrad, xpos,ypos, x, y, y_perp, vcirc_fit, einc1_fit, einc2_fit, epa1_fit, epa2_fit = parameters
r_test = np.linspace(0, np.nanmax(r_fit), 10000)

asymdrift = np.loadtxt(path+folder+"asymdrift.txt")
vasy = asymdrift[:,1]

Mb = 10.96
re = 1.617
n = 1.69
Md = 10.24#np.log10(2*np.pi*np.power(10, rd)**2*denp)
rd = 10.78
Mdh = 12.
con = 5.

v_b = V_b(r_test, Mb, re, n)
v_d = V_d(r_test, Md, rd)
v_dh = V_dh(r_test, Mdh, con)
v_test = np.sqrt(v_b**2 + v_d**2 + v_dh**2)
vcirc_fit = np.sqrt(vrot_fit**2 + vasy**2)

def log_likelihood(theta, x, y, yerr):
    Mb, re, n, Md, rd, Mdh, con, a_in, a_out = theta #Mb, re, n, Md, rd, 
    v_b = V_b(x, Mb, re, n)
    v_d = V_d(x, Md, rd)
    #Mdh = np.log10((10**Mb+10**Md)*(1/fb-1))
    v_dh = V_dh(x, Mdh, con)
    v_g = np.sqrt(a_in*vs_in**2 + a_out*vs_out**2)
    model = np.sqrt(v_b**2 + v_d**2 + v_dh**2 + v_g**2)
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2*np.pi*sigma2))

def log_prior(theta):
    Mb, re, n, Md, rd, Mdh, con, a_in, a_out = theta #Mb, re, n, Md, rd, 
    if not (9.<Mb<14. and 0.1<re<10.0 and 0.6<n<10.0 and 8.<Md<14.0 and 5.<rd<15. and 11.<Mdh<15.0 and 1.<con<15. and 0.<a_in<10.0 and 0.<a_out<10.0): #
        return -np.inf
    Mdmu = 10.64
    Mdsigma = 0.50
    lp0 = np.log(1./(np.sqrt(2*np.pi)*Mdsigma)) - 0.5*(Md - Mdmu)**2/Mdsigma**2-np.log(Mdmu)
    rdmu = 11.18
    rdsigma = 0.46
    lp1 = np.log(1./(np.sqrt(2*np.pi)*rdsigma)) - 0.5*(rd - rdmu)**2/rdsigma**2-np.log(rdmu)
    lp = lp0 + lp1
    #return 0
    return lp

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)
# %%
Mb, re, n, Md, rd, Mdh, con = 10.96, 1.617, 1.69, 10.64, 10.968, 13, 8
a = 3.1
np.random.seed(42)
from scipy.optimize import minimize
nll = lambda *args: -log_likelihood(*args)
initial = np.array([Mb, re, n, Md, rd, Mdh, con, a1, a2]) + 0.1 * np.random.randn(9)#Mb, re, n, Md, rd, 
soln = minimize(nll, initial, args=(r_fit[1:], disp_fit[1:], vrot_fit[1:], evrot2_fit[1:]), method="Nelder-Mead")

print("Fitting result:")
Mb_ml, re_ml, n_ml, Md_ml, rd_ml, Mdh_ml, con_ml, a1_ml, a2_ml = soln.x #
print("Mb = {0:.3f}".format(Mb_ml))
print("re = {0:.3f}".format(re_ml))
print("n = {0:.3f}".format(n_ml))
print("Md = {0:.3f}".format(Md_ml))
print("rd = {0:.3f}".format(rd_ml))
print("Mdh = {0:.3f}".format(Mdh_ml))
print("con = {0:.3f}".format(con_ml))
print("a1 = {0:.3f}".format(a1_ml))
print("a2 = {0:.3f}".format(a2_ml))


# %%
from multiprocessing import Pool
import emcee

pos = [Mb, re, n, Md, rd, Mdh, con] + 1e-4 * np.random.randn(200, 7)
nwalkers, ndim = pos.shape

backname = "tutorial.h5"
backend = emcee.backends.HDFBackend(output_dir+backname)
backend.reset(nwalkers, ndim)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(r_fit[1:], vrot_fit[1:], evrot2_fit[1:]), backend=backend)
    sampler.run_mcmc(pos, 1000, progress=True)

#### Plot the fitting parameters space
fig, axes = plt.subplots(7, figsize=(10, 40), sharex=True)
samples = sampler.get_chain()
labels = ["Mb", "re", "n", "Md", "rd", "Mdh","con"] #Mb, re, n, Md, rd, 
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number")
plt.savefig(output_dir+'step.png', bbox_inches='tight',dpi=300)

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)
inds = np.random.randint(len(flat_samples), size=100)
para = np.zeros(ndim)
for i in range(ndim):
    para[i] = np.percentile(flat_samples[:, i], [50])

import corner
fig = corner.corner(
    flat_samples, labels=labels, truths=[para[0], para[1], para[2], para[3], para[4], para[5], para[6]] #, para[2], para[3], para[4], para[5], para[6]
)
plt.savefig(output_dir+'corner.png',bbox_inches='tight', dpi=300)

plt.figure(figsize=(12,8))
ax = plt.subplot(111)
for ind in inds:
    sample = flat_samples[ind]
    v_b = V_b(r_test, sample[0], sample[1], sample[2])#sample[0], sample[1], sample[2]
    v_d = V_d(r_test, sample[3], sample[4]) #sample[3], sample[4]
    #Mdh = np.log10((10**sample[0]+10**sample[3])*(1/sample[5]-1))
    v_dh = V_dh(r_test, sample[5], sample[6])
    #ax.plot(r_test, v_b, "r", alpha=0.05)
    #ax.plot(r_test, v_d, "b", alpha=0.05)
    #ax.plot(r_test, v_dh, "y", alpha=0.05)
    ax.plot(r_test, np.sqrt(v_b**2+v_d**2+v_dh**2), "C1", alpha=0.5)

from IPython.display import display, Math
output = []
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    output.append(mcmc)

np.savetxt(output_dir+'parameter.txt', np.array(output))
v_b = V_b(r_test, para[0], para[1], para[2]) #para[0], para[1], para[2]
v_d = V_d(r_test, para[3], para[4]) #para[3], para[4]
#Mdh = np.log10((10**para[0]+10**para[3])*(1/para[5]-1))
v_dh = V_dh(r_test, para[5], para[6])

#plt.figure(figsize=(8,6))
#ax = plt.subplot(111)
#v_b = V_b(r_test, Mb_ml, re_ml, n_ml)
#v_d = V_d(r_test, Md_ml, rd_ml)
#v_dh = V_dh(r_test, Mdh_ml, con_ml)

ax.plot(r_test, v_b, "red", lw=2, label="Bulge")
ax.plot(r_test, v_d, "blue", lw=2, label="Disk")
ax.plot(r_test, v_dh, "yellow", lw=2, label="DM")
ax.plot(r_test, np.sqrt(v_b**2+v_d**2+v_dh**2), "k", label='Total')
ax.errorbar(r_fit, vcirc_fit, yerr=[-evrot1_fit, evrot2_fit], fmt='bo', mfc='none', ms=10, mew=0.5, elinewidth=0.5, capsize=0)
ax.errorbar(r_fit, vrot_fit, yerr=[-evrot1_fit, evrot2_fit], fmt='co', mfc='none', ms=10, mew=0.5, elinewidth=0.5, capsize=0)

ax.set_xlim(0, r_fit[-1]+1)
ax.set_xlabel('radius [kpc]')
ax.set_ylabel(r'$V_\mathrm{circ}$ [$\mathrm{km\,s^{-1}}$]')
plt.legend()
#plt.show()
plt.savefig(output_dir+'mcmc_fit.pdf', bbox_inches='tight', dpi=300)

# %%
beam_area = 0.18899035 ## true value

## Inner disk component
L0i = 1096.342/0.9*1e6     #/0.18899035 ## convert to M_sun/kpc^2
rei = 0.322/Planck15.arcsec_per_kpc_proper(z).value#*(1-0.650)
ni = 1.828
bni = gammaincinv(2*ni, 0.5)

## Outer disk component
L0o = 102.096/0.62*1e6    #/0.18899035
reo = 1.329/Planck15.arcsec_per_kpc_proper(z).value
no = 0.456
bno = gammaincinv(2*no, 0.5)

## The stellar disk part
rd = 9.02/Planck15.arcsec_per_kpc_proper(0.061).value
rse = rd/1.68 #Convert to scale length
sigmas0 = np.power(10, 10.64)/(2*np.pi*(rse)**2)            #Convert to M_sun/kpc^2
## sigmas0*np.exp(-(r/rse)) the stellar disk surface density

from scipy.misc import derivative

# %%
## All of parameters of the galaxy are set to be free

def v_ac2(theta, x, y):
    ## x is the radius, 
    ## y is the velocity dispersion
    Md, rd, a = theta[3], theta[4], theta[7]
    rse = rd/1.68   # Convert to scale length in kpc
    sigmas0 = np.power(10, Md)/(2*np.pi*rse**2)

    L1 = L0i*np.exp(-bni*((x/rei)**(1/ni)-1))
    L2 = L0o*np.exp(-bno*((x/reo)**(1/no)-1))
    dL1 = -bni/ni/rei*(x/rei)**(1/ni-1)*L1
    dL2 = -bno/no/reo*(x/reo)**(1/no-1)*L2
    Sigma_disk = sigmas0*np.exp(-x/rse)
    dSigma_disk = -Sigma_disk/rse

    model_asyg = y**2*x/(L1 + L2)*(dL1+dL2)
    model_asyt = y**2*x/(a*L1 + a*L2 + Sigma_disk)*(a*dL1+a*dL2+dSigma_disk)
    return model_asyg + model_asyt


def log_likelihood(theta, x, z, zerr):
    # Here theta are parameters, 
    # x is the radius, 
    # y is the velocity dispersion at that radius  
    # z is the rotation velocity, 
    # zerr is the error of rotation velocity
    Mb, re, n, Md, rd, Mdh, con, a = theta

    v_b = V_b(x, Mb, re, n)
    v_d = V_d(x, Md, rd)
    v_dh = V_dh(x, Mdh, con)
    v_g = np.sqrt((vs_in[1:])**2*a + (vs_out[1:])**2*a)

    #v_asym = v_ac2(theta, x, y)

    model_vc2 = v_b**2 + v_d**2 + v_dh**2 + v_g**2
    model = np.sqrt(model_vc2)# + v_asym
    sigma2 = zerr ** 2
    return -0.5 * np.sum((z - model) ** 2 / sigma2 + np.log(2*np.pi*sigma2))

def log_prior(theta):
    Mb, re, n, Md, rd, Mdh, con, a = theta #Mb, re, n, Md, rd, 
    if not (8.<Mb<14. and 0.1<re<10.0 and 0.1<n<10.0 and 8.<Md<15.0 and 2.<rd<15. and 10.<Mdh<18.0 and 1.<con<15. and 0.<a<20.0): #
        return -np.inf
    
    Mbmu = 10.96
    Mbsigma = 0.50
    lpMb = np.log(1./(np.sqrt(2*np.pi)*Mbsigma)) - 0.5*(Mb - Mbmu)**2/Mbsigma**2-np.log(Mbmu)

    remu = 1.617
    resigma = 0.05
    lpre = np.log(1./(np.sqrt(2*np.pi)*resigma)) - 0.5*(re - remu)**2/resigma**2-np.log(remu)

    nmu = 1.69
    nsigma = 0.05
    lpn = np.log(1./(np.sqrt(2*np.pi)*nsigma)) - 0.5*(n - nmu)**2/nsigma**2-np.log(nmu)

    Mdmu = 10.64
    Mdsigma = 0.50
    lp0 = np.log(1./(np.sqrt(2*np.pi)*Mdsigma)) - 0.5*(Md - Mdmu)**2/Mdsigma**2-np.log(Mdmu)

    rdmu = 10.97
    rdsigma = 0.47
    lp1 = np.log(1./(np.sqrt(2*np.pi)*rdsigma)) - 0.5*(rd - rdmu)**2/rdsigma**2-np.log(rdmu)
    lp = lp0 + lp1 + lpre + lpn
    #lp = 0
    return lp

def log_probability(theta, x, z, zerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, z, zerr)

# %%
output_dir = '/home/qyfei/Desktop/Codes/CODES/Dynamics/results/PG0050/params_8_prior/'
# %%
from multiprocessing import Pool
import emcee
Mb, re, n, Md, rd, Mdh, con = 10.96, 1.617, 1.69, 10.64, 10.968, 14, 5
#a1, a2 = 0.8, 3.1
#Md, re, n, Md, rd, Mdh, con = 10.50, 1.617, 1.69, 10.50, 8.4, 15., 8.
a = 0.8
pos = [Mb, re, n, Md, rd, Mdh, con, a] + 1e-4 * np.random.randn(400, 8)

nwalkers, ndim = pos.shape

backname = "tutorial.h5"
backend = emcee.backends.HDFBackend(output_dir+backname)
backend.reset(nwalkers, ndim)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(r_fit[1:], vcirc[1:], evrot2_fit[1:]), backend=backend)
    sampler.run_mcmc(pos, 500, progress=True)

# %%
#### Plot the fitting parameters space
fig, axes = plt.subplots(8, figsize=(10, 40), sharex=True)
samples = sampler.get_chain()
labels = ["$\log M_b$", "$r_e$", "$n$", "$\log M_d$", "$r_{e,d}$", "$\log M_\mathrm{DM}$", "c", "$\\alpha_\mathrm{CO}$"] #Mb, re, n, Md, rd, 
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number")
#plt.savefig(output_dir+'step.pdf', bbox_inches='tight',dpi=300)

# %%
#flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
flat_samples = get_chain.get_chain(discard=100, thin=15, flat=True)

print(flat_samples.shape)
inds = np.random.randint(len(flat_samples), size=100)
para = np.zeros(ndim)
for i in range(ndim):
    para[i] = np.percentile(flat_samples[:, i], [50])
# %%
import corner
fig = corner.corner(
    flat_samples, labels=labels, truths=[para[0], para[1], para[2], para[3], para[4], para[5], para[6], para[7]] #, para[2], para[3], para[4], para[5], para[6]
)
#plt.savefig(output_dir+'corner.pdf',bbox_inches='tight', dpi=300)


# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import corner

f = h5py.File("/home/qyfei/Desktop/Codes/CODES/Dynamics/results/PG0050/params_8_prior/tutorial.h5", "r")
accepted = f['mcmc']['accepted']
chain = f['mcmc']['chain']
log_prob = f['mcmc']['log_prob']
number = 200
get_chain = np.reshape(chain[number:], (400*(500-number), 8))

para = []
for i in range(len(get_chain[1])):
    print(np.percentile(get_chain[:,i], [50]), np.diff(np.percentile(get_chain[:,i], [16, 50, 84])))
    para.append(np.percentile(get_chain[:,i], [50]))

# %%
## Output and plot the fitting results
#Mb, re, n, Md, rd, Mdh, con = 10.80, 1.617, 1.69, 10.80, 10.968, 13, 8
#a1, a2 = 0.5/beam_area, 0.5/beam_area
#para = [Mb, re, n, Md, rd, Mdh, con, a1, a2]

def v_ac2(para, x, y):
    ## x is the radius, 
    ## y is the velocity dispersion
    rse, a = para[4]/1.68, para[7]
    sigmas0 = 1/(2*np.pi*rse**2)*np.power(10, para[3])
    L1 = L0i*np.exp(-bni*((x/rei)**(1/ni)-1))
    L2 = L0o*np.exp(-bno*((x/reo)**(1/no)-1))
    dL1 = -bni/ni/rei*(x/rei)**(1/ni-1)*L1
    dL2 = -bno/no/reo*(x/reo)**(1/no-1)*L2
    Sigma_disk = sigmas0*np.exp(-x/rse)
    dSigma_disk = -1/rse*Sigma_disk
    model_asyg = y**2*x/(L1 + L2)*(dL1+dL2)
    model_asyt = y**2*x/(a*L1 + a*L2 + Sigma_disk)*(a*dL1+a*dL2+dSigma_disk)
    return model_asyg + model_asyt

r_test = r_fit

plt.figure(figsize=(8,8))
ax = plt.subplot(111)
v_b = V_b(r_test, para[0], para[1], para[2]) #para[0], para[1], para[2]
v_d = V_d(r_test,para[3], para[4]) #para[3], para[4]
v_dh = V_dh(r_test, para[5], para[6])
v_g1 = vs_in*np.sqrt(para[7])
v_g2 = vs_out*np.sqrt(para[7])
v_g = np.sqrt((v_g1)**2 + (v_g2)**2)
v_c = np.sqrt(v_b**2 + v_d**2 + v_dh**2 + v_g**2)
ax.plot(r_test, v_b, "red", lw=2, label="Bulge")
ax.plot(r_test, v_d, "blue", lw=2, label="Disk")
ax.plot(r_test, v_dh, "yellow", lw=2, label="DM")

ax.plot(r_test, v_g1, "green", ls=":", lw=2)
ax.plot(r_test, v_g2, "green", ls="--",lw=2)
ax.plot(r_test, v_g, "green", lw=2, label="Gas")
#ax.plot(r_test, np.sqrt(v_b**2+v_d**2+v_dh**2+v_g**2), "k", label='Total')

for i in range(len(get_chain[::1000])):
    para = get_chain[i]
    v_b = V_b(r_test, para[0], para[1], para[2]) #para[0], para[1], para[2]
    v_d = V_d(r_test,para[3], para[4]) #para[3], para[4]
    v_dh = V_dh(r_test, para[5], para[6])
    v_g1 = vs_in*np.sqrt(para[7])
    v_g2 = vs_out*np.sqrt(para[7])
    v_g = np.sqrt((v_g1)**2 + (v_g2)**2)
    v_c = np.sqrt(v_b**2 + v_d**2 + v_dh**2 + v_g**2)
    ax.plot(r_test, v_b, "red", alpha=0.1)
    ax.plot(r_test, v_d, "blue", alpha=0.1)
    ax.plot(r_test, v_dh, "yellow", alpha=0.1)

    ax.plot(r_test, v_g1, "green", alpha=0.1)
    ax.plot(r_test, v_g2, "green", alpha=0.1)
    ax.plot(r_test, v_g, "green", alpha=0.1)
    ax.plot(r_test, np.sqrt(v_b**2+v_d**2+v_dh**2+v_g**2), "C1", alpha=0.1, zorder=0)

#vad2 = v_ac2(para, r_test, disp_fit)
#ax.plot(r_test, np.sqrt(v_c**2 + vad2), lw=2, label="final")
ax.errorbar(r_fit, vcirc, yerr=[-evrot1_fit, evrot2_fit], fmt='bo', mfc='none', ms=10, mew=1., elinewidth=1., capsize=0, zorder=3)

ax.set_xlim(0, r_fit[-1]+0.15)
ax.set_ylim(0, 340)
ax.set_xlabel('radius [kpc]')
ax.set_ylabel(r'$V_\mathrm{circ}$ [$\mathrm{km\,s^{-1}}$]')
plt.legend(fontsize=18, loc="best")

#plt.show()
#plt.savefig(output_dir+'mcmc_fit.pdf', bbox_inches='tight', dpi=300)

# %%

# %%
fig = corner.corner(
    get_chain, labels=labels, truths=[para[0], para[1], para[2], para[3], para[4], para[5], para[6], para[7]] #para[4], para[5], para[6], para[7], para[8]
)
#plt.savefig(output_dir+"corner.pdf", bbox_inches="tight", dpi=300)
# %%
labels = ["M_b", "r_e", "n", "M_d", "r_\mathrm{e,d}", "M_\mathrm{DM}", "c", "\\alpha_\mathrm{CO}"]
from IPython.display import display, Math
para_out = []
for i in range(len(get_chain[1])):
    para_out.append(np.percentile(get_chain[:,i], [50])[0])
    mcmc = (np.percentile(get_chain[:,i], [16, 50, 84]))
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))

# %%



# %%
## Only consider those paramters that we don't know
## dark matter halo and conversion factor

def v_ac2(theta, x, y):
    ## x is the radius, 
    ## y is the velocity dispersion
    rse, a = rd/1.68, theta[2]
    sigmas0 = np.power(10, 10.64)/(2*np.pi*rse**2)
    L1 = L0i*np.exp(-bni*((x/rei)**(1/ni)-1))
    L2 = L0o*np.exp(-bno*((x/reo)**(1/no)-1))
    dL1 = -bni/ni/rei*(x/rei)**(1/ni-1)*L1
    dL2 = -bno/no/reo*(x/reo)**(1/no-1)*L2
    Sigma_disk = sigmas0*np.exp(-x/rse)
    dSigma_disk = -Sigma_disk/rse
    model_asyg = y**2*x/(a*L1 + a*L2)*(a*dL1+a*dL2)
    model_asyt = y**2*x/(a*L1 + a*L2 + Sigma_disk)*(a*dL1+a*dL2+dSigma_disk)

    return model_asyg + model_asyt


def log_likelihood(theta, x, z, zerr):
    # Here theta are parameters, 
    # x is the radius, 
    # y is the velocity dispersion at that radius  
    # z is the rotation velocity, 
    # zerr is the error of rotation velocity
    Mdh, con, a = theta

    v_b = V_b(x, Mb, re, n)
    v_d = V_d(x, Md, rd)
    v_dh = V_dh(x, Mdh, con)
    v_g = np.sqrt((vs_in[1:])**2*a + (vs_out[1:])**2*a)

    #v_asym = v_ac2(theta, x, y)

    model_vc2 = v_b**2 + v_d**2 + v_dh**2 + v_g**2
    model = np.sqrt(model_vc2)# + v_asym
    sigma2 = zerr ** 2
    return -0.5 * np.sum((z - model) ** 2 / sigma2 + np.log(2*np.pi*sigma2))

def log_prior(theta):
    Mdh, con, a = theta #Mb, re, n, Md, rd, 
    if not (10.<Mdh<18.0 and 1.<con<15. and 0.<a<20.0): #
        return -np.inf
    lp = 0
    return lp

def log_probability(theta, x, z, zerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, z, zerr)

# %%
output_dir = '/home/qyfei/Desktop/Codes/CODES/Dynamics/results/PG0050/params_4/'

from multiprocessing import Pool
import emcee
Mb, re, n, Md, rd, Mdh, con = 10.96, 1.617, 1.69, 10.64, 10.968, 13, 5
a = 0.8
pos = [Mdh, con, a] + 1e-4 * np.random.randn(200, 3)

nwalkers, ndim = pos.shape

backname = "tutorial.h5"
backend = emcee.backends.HDFBackend(output_dir+backname)
backend.reset(nwalkers, ndim)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(r_fit[1:], vcirc[1:], evrot2_fit[1:]), backend=backend)
    sampler.run_mcmc(pos, 1000, progress=True)

# %%
#### Plot the fitting parameters space
fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["Mdh", "con", "a"] #Mb, re, n, Md, rd, 
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number")
#plt.savefig(output_dir+'step.pdf', bbox_inches='tight',dpi=300)

# %%
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)
inds = np.random.randint(len(flat_samples), size=100)
para = np.zeros(ndim)
for i in range(ndim):
    para[i] = np.percentile(flat_samples[:, i], [50])

import corner
fig = corner.corner(
    flat_samples, labels=labels, truths=[para[0], para[1], para[2]] #, para[2], para[3], para[4], para[5], para[6]
)
#plt.savefig(output_dir+'corner.pdf',bbox_inches='tight', dpi=300)

# %%
## Output and plot the fitting results
#Mb, re, n, Md, rd, Mdh, con = 10.80, 1.617, 1.69, 10.80, 10.968, 13, 8
#a1, a2 = 0.5/beam_area, 0.5/beam_area
#para = [Mb, re, n, Md, rd, Mdh, con, a1, a2]

def v_ac2(para, x, y):
    ## x is the radius, 
    ## y is the velocity dispersion
    rse, a = rd/1.68, para[2]
    sigmas0 = 1/(2*np.pi*rse**2)*np.power(10, Md)
    L1 = L0i*np.exp(-bni*((x/rei)**(1/ni)-1))
    L2 = L0o*np.exp(-bno*((x/reo)**(1/no)-1))
    dL1 = -bni/ni/rei*(x/rei)**(1/ni-1)*L1
    dL2 = -bno/no/reo*(x/reo)**(1/no-1)*L2
    Sigma_disk = sigmas0*np.exp(-x/rse)
    dSigma_disk = -1/rse*Sigma_disk
    model_asyg = y**2*x/(L1 + L2)*(dL1+dL2)
    model_asyt = y**2*x/(a*L1 + a*L2 + Sigma_disk)*(a*dL1+a*dL2+dSigma_disk)
    return model_asyg + model_asyt

r_test = r_fit

plt.figure(figsize=(8,8))
ax = plt.subplot(111)
v_b = V_b(r_test, Mb, re, n) #para[0], para[1], para[2]
v_d = V_d(r_test, Md, rd) #para[3], para[4]
v_dh = V_dh(r_test, para[0], para[1])
v_g1 = vs_in*np.sqrt(para[2])
v_g2 = vs_out*np.sqrt(para[2])
v_g = np.sqrt((v_g1)**2 + (v_g2)**2)
v_c = np.sqrt(v_b**2 + v_d**2 + v_dh**2 + v_g**2)

ax.plot(r_test, v_b, "red", lw=2, label="Bulge")
ax.plot(r_test, v_d, "blue", lw=2, label="Disk")
ax.plot(r_test, v_dh, "yellow", lw=2, label="DM")

ax.plot(r_test, v_g1, "green", ls=":", lw=2)
ax.plot(r_test, v_g2, "green", ls="--",lw=2)
ax.plot(r_test, v_g, "green", lw=2, label="Gas")
ax.plot(r_test, np.sqrt(v_b**2+v_d**2+v_dh**2+v_g**2), "k", label='Total')

vad2 = v_ac2(para, r_test, disp_fit)
ax.plot(r_test, np.sqrt(v_c**2 + vad2), lw=2, label="final")
ax.errorbar(r_fit, vrot_fit, yerr=[-evrot1_fit, evrot2_fit], fmt='bo', mfc='none', ms=10, mew=1., elinewidth=1., capsize=0)

ax.set_xlim(0, r_fit[-1]+0.15)
ax.set_ylim(0, 340)
ax.set_xlabel('radius [kpc]')
ax.set_ylabel(r'$V_\mathrm{circ}$ [$\mathrm{km\,s^{-1}}$]')
plt.legend(fontsize=18, loc="lower right")

# %%
