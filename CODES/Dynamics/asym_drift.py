# %%
## Try to calculate the luminosity distribution along radius
from astropy.cosmology import Planck15
import astropy.units as u
from scipy.special import gammaincinv
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

z = 0.061
DL = Planck15.luminosity_distance(z)
nu_obs = 230.58/(1+z)
rt = np.linspace(0, 100, 10000)
## The outer disk part
I0o = 0.420
L0o = 3.25e7*I0o*DL.value**2/(1+z)**3/nu_obs**2/188990.35/0.62 #convert to CO(1-0)
reo = 1.329/Planck15.arcsec_per_kpc_proper(z).value
no = 0.456
bno = gammaincinv(2*no, 0.5)
Lro = 3.1*L0o*np.exp(-bno*((rt/reo)**(1/no)-1))

## The inner disk part
I0i = 4.510
L0i = 3.25e7*I0i*DL.value**2/(1+z)**3/nu_obs**2/188990.35/0.9 #convert to CO(1-0)
rei = 0.322/Planck15.arcsec_per_kpc_proper(z).value#*(1-0.650)
ni = 1.829
bni = gammaincinv(2*ni, 0.5)
Lri = 3.1*L0i*np.exp(-bni*((rt/rei)**(1/ni)-1))


alpha_CO = 1.
## Inner disk component
L0i = 1096.342/0.9     #/0.18899035 ## convert to M_sun/kpc^2
rei = 0.322/Planck15.arcsec_per_kpc_proper(z).value     #*(1-0.650)
ni = 1.828
bni = gammaincinv(2*ni, 0.5)
Lri = alpha_CO*L0i*np.exp(-bni*((rt/rei)**(1/ni)-1))

## Outer disk component
L0o = 102.096/0.62    #/0.18899035
reo = 1.329/Planck15.arcsec_per_kpc_proper(z).value
no = 0.456
bno = gammaincinv(2*no, 0.5)
Lro = alpha_CO*L0o*np.exp(-bno*((rt/reo)**(1/no)-1))

## The stellar disk part
rse = 9.02/Planck15.arcsec_per_kpc_proper(0.061).value/1.68 #Convert to r_d
sigmas0 = np.power(10, 10.64)/(2*np.pi*(rse)**2)/1e6
sigmad = sigmas0*np.exp(-(rt/rse))

## The stellar bulge part
reb = 1.33/Planck15.arcsec_per_kpc_proper(0.061).value
nb = 1.69
bnb = gammaincinv(2*nb, 0.5)
sigma0b = np.power(10, 10.96)/39.4674/1e6
sigmab = sigma0b*np.exp(-bnb*((rt/reb)**(1/nb)-1))

plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
ax.plot(rt, Lro, "k:", lw=2, label="Outer")
ax.plot(rt, Lri, "k", lw=2, label="Inner")
ax.plot(rt, sigmad, "k--", lw=2, label="Stellar disk")
ax.plot(rt, sigmab, "k-.", lw=2, label="Stellar bulge")

ax.set_xlim(1e-2, 1e2)
ax.set_ylim(1, 5e5)
ax.loglog()
ax.set_xlabel("radius [kpc]")
ax.set_ylabel("$\Sigma$ [$M_\odot\,\mathrm{pc}^{-2}$]")
plt.legend()

#plt.savefig("/home/qyfei/Desktop/Codes/CODES/map_visualization/1D_fitting/PG0050/profile.pdf", bbox_inches="tight", dpi=300)

# %%
def f(x): #bulge mass
    return 2*np.pi*x*sigma0b*np.exp(-bnb*((x/reb)**(1/nb)-1))

def f_in(x): #innter gas mass
    return 2*np.pi*x*L0i*np.exp(-bni*((x/rei)**(1/ni)-1))

def f_out(x): # outer gas mass
    return 2*np.pi*x*L0o*np.exp(-bno*((x/reo)**(1/no)-1))

def f_d(x): #stellar disk mass
    return 2*np.pi*x*sigmas0*np.exp(-x/rse)

Mb = integrate.quad(f, 0, np.inf)[0]*1e6            ## Test of bulge mass
Md = integrate.quad(f_d, 0, np.inf)[0]*1e6          ## Test of stellar mass
Mgin = integrate.quad(f_in, 0, np.inf)[0]*1e6       ## Test of inner gas disk mass
Mgout = integrate.quad(f_out, 0, np.inf)[0]*1e6     ## Test of outer gas disk mass

# %%
## Define the mass distribution of different components
def sigmag(x):
    L_in = L0i*np.exp(-bni*((x/rei)**(1/ni)-1))
    L_out = L0o*np.exp(-bno*((x/reo)**(1/no)-1))
    return (L_out + L_in)*alpha_CO

def sigmad(x):
    return sigmas0*np.exp(-x/rse)

def sigmat(x):
    return sigmag(x) + sigmad(x)
# %%
from scipy.misc import derivative
from Barolo_analyse.parameters import load_parameters

path = '/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/'
folder = "Barolo_fit/output/PG0050+124_best/"
file = "ringlog2.txt"
r_fit,rad_fit,vrot_fit,disp_fit,inc,pa,z0,xpos,ypos,vsys,vrad, evrot1_fit, evrot2_fit, edisp1_fit, edisp2_fit = np.genfromtxt(path+folder+file,skip_header=1,usecols=(0,1,2,3,4,5,7,9,10,11,12,13,14,15,16),unpack=True) 
output_dir = "/home/qyfei/Desktop/Codes/CODES/Dynamics/results/PG0050/prior/"
parameters = load_parameters(path, folder, file)
r_fit, rad_fit, vrot_fit, evrot1_fit, evrot2_fit, vdisp_fit, edisp1_fit, edisp2_fit, inc_fit, pa_fit, vrad_fit, vsys_fit, vrad, xpos,ypos, x, y, y_perp, vcirc_fit, einc1_fit, einc2_fit, epa1_fit, epa2_fit = parameters
'''
vcirc = []
Mds = np.linspace(9, 12, 16)
sigmas0s = np.power(10, Mds)/(2*np.pi*(rse)**2)/1e6
for sigmas0 in sigmas0s:
    asy0 = r_fit/sigmag(r_fit)*derivative(sigmag, r_fit, dx=1e-8)
    asy1 = r_fit/sigmat(r_fit)*derivative(sigmat, r_fit, dx=1e-8)
    vcirc2 = vrot_fit**2 - vdisp_fit**2*(asy0+asy1)
    vcirc.append(np.sqrt(vcirc2))
'''
alpha_CO = 1.0
rse = 9.02/Planck15.arcsec_per_kpc_proper(0.061).value/1.68
sigmas0 = np.power(10, 10.64)/(2*np.pi*rse**2)/1e6
asy0 = r_fit/sigmag(r_fit)*derivative(sigmag, r_fit, dx=1e-8)
asy1 = r_fit/sigmat(r_fit)*derivative(sigmat, r_fit, dx=1e-8)
vcirc2 = vrot_fit**2 - vdisp_fit**2*(asy0+asy1)
vcirc = np.sqrt(vcirc2)

print(vcirc)

# %%
plt.figure(figsize=(8, 8))
ax = plt.subplot(111)

ax.errorbar(r_fit, vrot_fit, yerr=[-evrot1_fit, evrot2_fit], fmt='bo', mfc='none', ms=10, mew=1, elinewidth=1, capsize=0, label="$v_\mathrm{rot}$")

ax.errorbar(r_fit, vcirc, yerr=[-evrot1_fit, evrot2_fit], fmt='ro', mfc='none', ms=10, mew=1, elinewidth=1, capsize=0, label="$v_\mathrm{circ}$")

ax.set_xlim(0, r_fit[-1]+0.15)
ax.set_ylim(0, 380)
ax.set_xlabel('radius [kpc]')
ax.set_ylabel(r'$V_\mathrm{circ}$ [$\mathrm{km\,s^{-1}}$]')
plt.legend()
#plt.savefig("/home/qyfei/Desktop/Codes/CODES/Dynamics/results/PG0050/prior_gas/RC_ad.pdf", bbox_inches="tight", dpi=300)
print(vcirc)
# %%
# %%
