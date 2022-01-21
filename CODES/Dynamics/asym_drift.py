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


alpha_CO = 3.1

## Three components model
## Bar component
L0i = 577.861/0.90                                      # convert into M_sun/kpc^2
rei = 0.494/Planck15.arcsec_per_kpc_proper(z).value
ni = 0.304
bni = gammaincinv(2*ni, 0.5)
## Disk component
I0o = 106.093
L0o = 106.093/0.60    
reo = 1.297/Planck15.arcsec_per_kpc_proper(z).value
no = 0.476
bno = gammaincinv(2*no, 0.5)
Lri = alpha_CO*L0i*np.exp(-bni*((rt/rei)**(1/ni)-1))    # Bar correction
Lro = alpha_CO*L0o*np.exp(-bno*((rt/reo)**(1/no)-1))    # Disk correction
## Gaussian component
A = 7302.437
xstd = 0.093/Planck15.arcsec_per_kpc_proper(z).value
LrG = alpha_CO*A*np.exp(-rt**2/(2*xstd**2))
'''

## Two components model
## Bar component
L0i = 1096.342/0.9                                  # convert into M_sun/kpc^2
rei = 0.322/Planck15.arcsec_per_kpc_proper(z).value
ni = 1.828
bni = gammaincinv(2*ni, 0.5)
## Disk component
I0o = 102.096
L0o = 102.096/0.62    
reo = 1.329/Planck15.arcsec_per_kpc_proper(z).value
no = 0.456
bno = gammaincinv(2*no, 0.5)
'''
Lri = alpha_CO*L0i*np.exp(-bni*((rt/rei)**(1/ni)-1))    # Bar correction
Lro = alpha_CO*L0o*np.exp(-bno*((rt/reo)**(1/no)-1))    # Disk correction

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
alpha_CO = 0.8
def sigmag(x):
    L_bar = L0i*np.exp(-bni*((x/rei)**(1/ni)-1))
    L_disk = L0o*np.exp(-bno*((x/reo)**(1/no)-1))
    L_Gauss = A*np.exp(-x**2/(2*xstd**2))
    return (L_disk + L_bar + L_Gauss)*alpha_CO

def sigma_sd(x):
    return sigmas0*np.exp(-x/rse)

def sigmat(x):
    return sigmag(x) + sigma_sd(x)
# %%
plt.figure(figsize=(8, 16))
ax0, ax1 = plt.subplot(211), plt.subplot(212)
ax0.plot(rt, Lro, "k:", lw=2, label="Disk")
ax0.plot(rt, Lri, "k", lw=2, label="Bar")
ax0.plot(rt, LrG, "k.", lw=2, label="Point")
ax0.plot(rt, sigmad, "k--", lw=2, label="Stellar disk")
ax0.plot(rt, sigmab, "k-.", lw=2, label="Stellar bulge")

ax0.set_xlim(1e-2, 1e2)
ax0.set_ylim(1, 5e5)
ax0.loglog()
ax0.set_xlabel("radius [kpc]")
ax0.set_ylabel("$\Sigma$ [$M_\odot\,\mathrm{pc}^{-2}$]")

asy0 = rt/sigmag(rt)*derivative(sigmag, rt, dx=1e-8)   # Gas asymmetric drift correction
asy1 = rt/sigmat(rt)*derivative(sigmat, rt, dx=1e-8)
ax1.plot(rt, -asy0, label="Gas")
ax1.plot(rt, -asy1, label="Total")
ax1.set_xlim(1e-2, 1e2)
ax1.set_ylim(1e-1, 1e2)
ax1.loglog()
plt.legend()

# %%
from scipy.misc import derivative
from Barolo_analyse.parameters import load_parameters

path = '/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/'
folder = "Barolo_fit/output/PG0050+124_best/"
#folder = "Barolo_fit/output/PG0050+124/"
file = "ringlog2.txt"
r_fit,rad_fit,vrot_fit,disp_fit,inc,pa,z0,xpos,ypos,vsys,vrad, evrot1_fit, evrot2_fit, edisp1_fit, edisp2_fit = np.genfromtxt(path+folder+file,skip_header=1,usecols=(0,1,2,3,4,5,7,9,10,11,12,13,14,15,16),unpack=True)

folder = "Barolo_fit/output/PG0050+124_trash/"
file = "ringlog2.txt"
r_fit = np.loadtxt(path+folder+file)[:,0]

#output_dir = "/home/qyfei/Desktop/Codes/CODES/Dynamics/results/PG0050/prior/"
#parameters = load_parameters(path, folder, file)
#r_fit, rad_fit, vrot_fit, evrot1_fit, evrot2_fit, vdisp_fit, edisp1_fit, edisp2_fit, inc_fit, pa_fit, vrad_fit, vsys_fit, vrad, xpos,ypos, x, y, y_perp, vcirc_fit, einc1_fit, einc2_fit, epa1_fit, epa2_fit = parameters
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
alpha_CO = 4.3
def sigmag(x):
    L_bar = L0i*np.exp(-bni*((x/rei)**(1/ni)-1))
    L_disk = L0o*np.exp(-bno*((x/reo)**(1/no)-1))
    L_Gauss = A*np.exp(-x**2/(2*xstd**2))
    return (L_disk + L_bar + L_Gauss)*alpha_CO

def sigma_sd(x):
    return sigmas0*np.exp(-x/rse)

def sigmat(x):
    return sigmag(x) + sigma_sd(x)#*2/(1+60**2/disp_fit**2)

#alpha_CO = 4.3
rse = 9.02/Planck15.arcsec_per_kpc_proper(0.061).value/1.68
sigmas0 = np.power(10, 10.64)/(2*np.pi*rse**2)/1e6
asy0 = r_fit/sigmag(r_fit)*derivative(sigmag, r_fit, dx=1e-8)   # Gas asymmetric drift correction
asy1 = r_fit/sigmat(r_fit)*derivative(sigmat, r_fit, dx=1e-8)
print(asy0+asy1)
vcirc2 = vrot_fit**2 - disp_fit**2*(asy0+asy1)
vcirc = np.sqrt(vcirc2)
evcirc1_fit = np.sqrt(evrot1_fit**2 + edisp1_fit**2*(asy0+asy1)**2)
evcirc2_fit = np.sqrt(evrot2_fit**2 + edisp2_fit**2*(asy0+asy1)**2)

#print(vcirc)
# %%
G = 4.302e-6
plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
## Rotation velocity, before asymmetric drift correction
ax.errorbar(r_fit, vrot_fit, yerr=[-evrot1_fit, evrot2_fit], fmt='bo', mfc='none', ms=10, mew=2, elinewidth=2, capsize=5, label="$v_\mathrm{rot}$")
## Circular velocity, after asymmetric drift correction
ax.errorbar(r_fit, vcirc, yerr=[evcirc1_fit, evcirc2_fit], fmt='ks', mfc='k',ms=10, mew=2, elinewidth=2, capsize=5, label="ADC")
#ax.plot(r_fit, v_g*np.sqrt(alpha_CO), "k", label="Total")
#ax.plot(r_fit, vs_in*np.sqrt(alpha_CO), "k:", label="Bar")
#ax.plot(r_fit, vs_out*np.sqrt(alpha_CO), "k--", label="Disk")
#ax.plot(r_fit, vs_p*np.sqrt(alpha_CO), "k-.", label="Point")
#ax.plot(r_fit, np.sqrt(G*alpha_CO*10**9.75/r_fit), "k", lw=3, alpha=0.7, label="Total Point")
plt.legend()
ax.set_xlim(0, r_fit[-1]+0.15)
ax.set_ylim(0, 380)
ax.set_xlabel('radius [kpc]')
ax.set_ylabel(r'$V_\mathrm{circ}$ [$\mathrm{km\,s^{-1}}$]')

#plt.savefig("/home/qyfei/Desktop/Results/Dynamics/results/PG0050/asym_drift/RC_adc.pdf", bbox_inches="tight", dpi=300)

# %%
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
