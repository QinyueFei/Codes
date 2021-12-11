# %%
import astropy.constants as c
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.functional_models import Sersic1D
from astropy.stats import sigma_clipped_stats
from Physical_values.surface_density import *
from Dynamics.models import *

plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

radius = iso_rad(800, np.array([395, 399]), 0.05, 35, 41)
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
file = "PG0050_CO21-combine-line-10km-mosaic-mom0.fits"
file_rms = "PG0050_CO21-combine-line-10km-mosaic-mom0-pbcor.fits"
file_mom2 = "PG0050_CO21-combine-line-10km-mosaic-mom2.fits"

mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file_rms)
bmaj = hdu.header['BMAJ']
bmin = hdu.header['BMIN']
delt = hdu.header['CDELT1']
CO_pix = np.pi*bmaj*bmin/(4*np.log(2)*delt**2)

rad = r_fit[1:]#np.logspace(-1, 1, 50)
f_ring = []
f_ringrms = []
for i in range(len(rad)-1):
    mask = ((radius>=rad[i]) & (radius<rad[i+1]))
    N = len(np.where(mask)[0])
    mom0_ring = mom0*mask
    mom0_ring[mom0_ring == 0] = 'nan'
    f_ring.append(np.nansum(mom0_ring))
    f_ringrms.append(np.nanstd(mom0_ring))
f_ring = np.array(f_ring)
f_ringrms = np.array(f_ringrms)

f_rad = []
f_radrms = []
for i in range(len(rad)):
    mask = (radius<rad[i])
    N = len(np.where(mask)[0])
    mom0_ring = mom0*mask
    mom0_ring[mom0_ring == 0] = 'nan'
    f_rad.append(np.nansum(mom0_ring))
    f_radrms.append(np.nanstd(mom0_ring))
f_rad = np.array(f_rad)
f_radrms = np.array(f_radrms)

# %%
mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file)

from astropy.cosmology import Planck15
z = 0.061
DL = Planck15.luminosity_distance(z)
nu_obs = 230.58/(1+z)
inc = 41*u.deg.to('rad')
norm = 1/CO_pix*1.36
alpha_CO = 0.8*u.Unit('Msun')

L_CO_ring = alpha_CO*3.25e7*f_ring*DL.value**2/((1+z)**3*nu_obs**2)*norm
L_CO_ringrms = alpha_CO*3.25e7*(f_ring+f_ringrms)*DL.value**2/((1+z)**3*nu_obs**2)*norm - L_CO_ring

M_CO_rad = alpha_CO*3.25e7*f_rad*DL.value**2/((1+z)**3*nu_obs**2)*norm
M_CO_radrms = alpha_CO*3.25e7*(f_rad+f_radrms)*DL.value**2/((1+z)**3*nu_obs**2)*norm - M_CO_rad
G = 4.302e-6
# %%
from Barolo_analyse.parameters import load_parameters
from Dynamics.models import *

path = '/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/'
folder = "Barolo_fit/output/PG0050+124_best/"
file = "ringlog2.txt"
r_fit,rad_fit,vrot_fit,disp_fit,inc,pa,z0,xpos,ypos,vsys,vrad, evrot1_fit, evrot2_fit, edisp1_fit, edisp2_fit = np.genfromtxt(path+folder+file,skip_header=1,usecols=(0,1,2,3,4,5,7,9,10,11,12,13,14,15,16),unpack=True) 
output_dir = "/home/qyfei/Desktop/Codes/CODES/Dynamics/results/PG0050/prior/"
parameters = load_parameters(path, folder, file)
r_fit, rad_fit, vrot_fit, evrot1_fit, evrot2_fit, vdisp_fit, edisp1_fit, edisp2_fit, inc_fit, pa_fit, vrad_fit, vsys_fit, vrad, xpos,ypos, x, y, y_perp, vcirc_fit, einc1_fit, einc2_fit, epa1_fit, epa2_fit = parameters
r_test = rad

asymdrift = np.loadtxt(path+folder+"asymdrift.txt")
vasy = asymdrift[:,1]
# %%
Mb = 10.96
re = 1.62
n = 1.69

Md = 10.64#np.log10(2*np.pi*np.power(10, rd)**2*denp)
rd = 10.97

Mdh = 12.
con = 5.
a = 3.1

def bulge_mass(r_, M_, re_, n_): 
    # The bulge mass at each radius with PS97 model, where M, re, n are total mass, effective radius and Sersic index
    M, re, n, r = 10**M_, re_, n_, r_
    x = r/re
    p = 1-0.6097/n+0.05563/n**2
    b = gammaincinv(2*n,0.5)
    M_r = 2*M*gammainc((3-p)*n, b*x**(1/n))/np.power(b, (1-p)*n)/gamma(2*n)
    return M_r
def disk_mass(r_, Md_, rd_):
    r, Md, rd = r_, 10**Md_, rd_/1.68
    den = Md/(2*np.pi*rd**2) #calculate the surface density of disk at effective radius
    y = r/rd/2
    M = 2*np.pi*den*rd**2*(1-np.exp(-r/rd)*(1+r/rd))
    return M

M_bulge = bulge_mass(r_test, Mb, re, n)
M_disk = disk_mass(r_test, Md, rd)

plt.figure(figsize=(8,8))
ax = plt.subplot(111)
ax.errorbar(r_test, M_bulge, fmt='ro', mfc='none', label="$M_b$")
ax.errorbar(r_test, M_disk, fmt='bs', mfc='none', label="$M_d$")
ax.errorbar(r_test, M_CO_rad.value, fmt='kd', mfc='none', label="$M_g$")
ax.set_xlabel('radius [kpc]')
ax.set_ylabel(r'Mass [$M_\odot$]')
ax.semilogy()
plt.legend()
plt.show()

# %%
V_bulge = np.sqrt(G*M_bulge/r_test)
V_disk = np.sqrt(G*M_disk/r_test)
V_disk_t = V_d(r_test, Md, rd)

V_gas = np.sqrt(G*M_CO_rad.value/r_test)
V_tot = np.sqrt(V_bulge**2 + V_disk**2 + V_gas**2)
plt.figure(figsize=(8,8))
ax = plt.subplot(111)
ax.errorbar(r_test, V_bulge, fmt='ro', mfc='none', label="$V_b$")
ax.errorbar(r_test, V_disk, fmt='bs', mfc='none', label="$V_d$")
ax.errorbar(r_test, V_disk_t, fmt='rs', mfc='none')
ax.errorbar(r_test, V_gas, fmt='kd', mfc='none', label="$V_g$")
ax.plot(r_test, V_tot, 'k-')
ax.errorbar(r_fit[1:], vcirc_fit[1:], yerr=[-evrot1_fit[1:], evrot2_fit[1:]], fmt='bo', mfc='none', ms=10, mew=0.5, elinewidth=0.5, capsize=0)

ax.set_xlabel('radius [kpc]')
ax.set_ylabel(r'V [$\mathrm{km\,s^{-1}}$]')
#ax.semilogy()
plt.legend()
plt.show()

# %%

v_b = V_b(r_test, Mb, re, n)
v_d = V_d(r_test, Md, rd)
v_dh = V_dh(r_test, Mdh, con)
v_test = np.sqrt(v_b**2 + v_d**2 + v_dh**2)
vcirc_fit = np.sqrt(vrot_fit[1:]**2 + vasy[1:]**2)

def log_likelihood(theta, x, y, yerr):
    Mb, re, n, Md, rd, Mdh, con, a = theta #Mb, re, n, Md, rd, 
    v_b = V_b(x, Mb, re, n)
    v_d = V_d(x, Md, rd)
    #Mdh = np.log10((10**Mb+10**Md)*(1/fb-1))
    v_dh = V_dh(x, Mdh, con)
    v_g = np.sqrt(a)*vg.value
    #model = v_b**2 + v_d**2 + v_dh**2
    model = np.sqrt(v_b**2 + v_d**2 + v_dh**2 + v_g**2)
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2*np.pi*sigma2))

def log_prior(theta):
    Mb, re, n, Md, rd, Mdh, con, a = theta #Mb, re, n, Md, rd, 
    if not (9.<Mb<14. and 0.1<re<10.0 and 0.6<n<10.0 and 8.<Md<14.0 and 5.<rd<15. and 11.<Mdh<15.0 and 1.<con<15. and 0.<a<5.): #
        return -np.inf
    Mbmu = 10.96
    Mbsigma = 0.50
    lpMb = np.log(1./(np.sqrt(2*np.pi)*Mbsigma)) - 0.5*(Mb - Mbmu)**2/Mbsigma**2-np.log(Mbmu)
    remu = 1.62
    resigma = 0.05
    lpre = np.log(1./(np.sqrt(2*np.pi)*resigma)) - 0.5*(re - remu)**2/resigma**2-np.log(remu)
    nmu = 1.69
    nsigma = 0.05
    lpn = np.log(1./(np.sqrt(2*np.pi)*nsigma)) - 0.5*(n - nmu)**2/nsigma**2-np.log(nmu)
    Mdmu = 10.64
    Mdsigma = 0.50
    lp0 = np.log(1./(np.sqrt(2*np.pi)*Mdsigma)) - 0.5*(Md - Mdmu)**2/Mdsigma**2-np.log(Mdmu)
    rdmu = 6.54
    rdsigma = 0.5
    lp1 = np.log(1./(np.sqrt(2*np.pi)*rdsigma)) - 0.5*(rd - rdmu)**2/rdsigma**2-np.log(rdmu)
    lp = lpMb + lpre + lpn + lp0 + lp1
    #return 0
    return lp

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

np.random.seed(42)
from scipy.optimize import minimize
nll = lambda *args: -log_likelihood(*args)
initial = np.array([Mb, re, n, Md, rd, Mdh, con, a]) + 0.1 * np.random.randn(8)#Mb, re, n, Md, rd, 
soln = minimize(nll, initial, args=(r_fit[1:], vcirc_fit[:], evrot2_fit[1:]), method="Nelder-Mead")

print("Fitting result:")
Mb_ml, re_ml, n_ml, Md_ml, rd_ml, Mdh_ml, con_ml, a_ml = soln.x #
print("Mb = {0:.3f}".format(Mb_ml))
print("re = {0:.3f}".format(re_ml))
print("n = {0:.3f}".format(n_ml))
print("Md = {0:.3f}".format(Md_ml))
print("rd = {0:.3f}".format(rd_ml))
print("Mdh = {0:.3f}".format(Mdh_ml))
print("con = {0:.3f}".format(con_ml))
print("a = {0:.3f}".format(a_ml))

# %%
from multiprocessing import Pool
import emcee

#pos = [Mb_ml, re_ml, n_ml, Md, rd, Mdh, con] + 1e-4 * np.random.randn(50, 7)
pos = [Mb, re, n, Md, rd, Mdh, con, a] + 1e-4 * np.random.randn(200, 8)
# Mb, re, n, Md, rd, 
#[Mb_ml, re_ml, n_ml, Md_ml, rd_ml, Mdh_ml] + 1e-4 * np.random.randn(32, 6)
nwalkers, ndim = pos.shape
output_dir = "/home/qyfei/Desktop/Codes/CODES/Dynamics/results/PG0050/gas_including/"

backname = "tutorial.h5"
backend = emcee.backends.HDFBackend(output_dir+backname)
backend.reset(nwalkers, ndim)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(r_fit[1:], vcirc_fit, evrot2_fit[1:]), backend=backend)
    sampler.run_mcmc(pos, 1000, progress=True)

# %%
#### Plot the fitting parameters space
fig, axes = plt.subplots(8, figsize=(10, 40), sharex=True)
samples = sampler.get_chain()
labels = ["Mb", "re", "n", "Md", "rd", "Mdh","con", "a"] #Mb, re, n, Md, rd, 
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number")
plt.savefig(output_dir+'step.png', bbox_inches='tight',dpi=300)

# %%
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)
inds = np.random.randint(len(flat_samples), size=100)
para = np.zeros(ndim)
for i in range(ndim):
    para[i] = np.percentile(flat_samples[:, i], [50])

import corner
fig = corner.corner(
    flat_samples, labels=labels, truths=[para[0], para[1], para[2], para[3], para[4], para[5], para[6], para[7]] #, para[2], para[3], para[4], para[5], para[6]
)
plt.savefig(output_dir+'corner.png',bbox_inches='tight', dpi=300)
# %%
from IPython.display import display, Math

for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))

# %%

plt.figure(figsize=(12,8))
ax = plt.subplot(111)
for ind in inds:
    sample = flat_samples[ind]
    v_b = V_b(r_test, sample[0], sample[1], sample[2])#sample[0], sample[1], sample[2]
    v_d = V_d(r_test, sample[3], sample[4]) #sample[3], sample[4]
    #Mdh = np.log10((10**sample[0]+10**sample[3])*(1/sample[5]-1))
    v_dh = V_dh(r_test, sample[5], sample[6])
    v_g = np.sqrt(sample[7])*vg.value
    #ax.plot(r_test, v_b, "r", alpha=0.05)
    #ax.plot(r_test, v_d, "b", alpha=0.05)

    #ax.plot(r_test, v_dh, "y", alpha=0.05)
    ax.plot(r_test, np.sqrt(v_b**2+v_d**2+v_dh**2+v_g**2), "C1", alpha=0.05)

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
v_g = np.sqrt(para[7])*vg.value

ax.plot(r_test, v_b, "red", lw=2, label="Bulge")
ax.plot(r_test, v_d, "blue", lw=2, label="Disk")
ax.plot(r_test, v_dh, "yellow", lw=2, label="DM")
ax.plot(r_test, v_g, "green", lw=2, label="Gas")

ax.plot(r_test, np.sqrt(v_b**2+v_d**2+v_dh**2+v_g**2), "k", label='Total')
ax.errorbar(r_fit[1:], vcirc_fit, yerr=[-evrot1_fit[1:], evrot2_fit[1:]], fmt='bo', mfc='none', ms=10, mew=0.5, elinewidth=0.5, capsize=0)
ax.errorbar(r_fit, vrot_fit, yerr=[-evrot1_fit, evrot2_fit], fmt='co', mfc='none', ms=10, mew=0.5, elinewidth=0.5, capsize=0)

ax.set_xlim(0, r_fit[-1]+0.5)
ax.set_xlabel('radius [kpc]')
ax.set_ylabel(r'$V_\mathrm{circ}$ [$\mathrm{km\,s^{-1}}$]')
plt.legend()
#plt.show()
plt.savefig(output_dir+'mcmc_fit.pdf', bbox_inches='tight', dpi=300)

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import corner

with h5py.File("/home/qyfei/Desktop/Codes/CODES/Dynamics/results/PG0050/prior/tutorial.h5", "r") as f:
    print(list(f.keys()))

f = h5py.File("/home/qyfei/Desktop/Codes/CODES/Dynamics/results/ngc2403/tutorial.h5", "r")
accepted = f['mcmc']['accepted']
chain = f['mcmc']['chain']
log_prob = f['mcmc']['log_prob']

get_chain = np.reshape(chain[100:], (400*500, 7))

# %%
for i in range(len(get_chain[1])):
    print(np.percentile(get_chain[:,i], [50]), np.diff(np.percentile(get_chain[:,i], [16, 50, 84])))

# %%
