# %%
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colorbar as colorbar
from matplotlib.transforms import Affine2D
from astropy.visualization.wcsaxes import WCSAxes
import astropy.units as u
from spectral_cube import SpectralCube
from astropy.modeling import models, fitting
import astropy.constants as c
import astropy.units as u
from scipy.special import gamma
from scipy.special import gammaincinv
from scipy.special import gammainc
from astropy.cosmology import Planck15

def PS_density(M_, re_, n_, r_):
    M, re, n, r = 10**M_, re_, n_, r_
    x = r/re
    p = 1-0.6097/n+0.05563/n**2
    b = gammaincinv(2*n,0.5)
    rho_0 = M*np.power(b, n*(3-p))/(4*np.pi*re**3*n*gamma(n*(3-p)))
    rho_r = rho_0*np.power(x,-p)*np.exp(-b*np.power(x,1/n))
    return rho_r

def bulge_mass(M_, re_, n_, r_):
    M, re, n, r = 10**M_, re_, n_, r_
    x = r/re
    p = 1-0.6097/n+0.05563/n**2
    b = gammaincinv(2*n,0.5)
    norm = gamma(n*(3-p))/gammainc(n*(3-p),b*np.power(1e3,1/n))
    M_tot = M*gammainc(n*(3-p),b*x**(1/n))/gamma(n*(3-p))*norm
    return M_tot
'''
def bulge_mass(M_, re_, n_, r_): 
    # The bulge mass at each radius with PS97 model, where M, re, n are total mass, effective radius and Sersic index
    M, re, n, r = 10**M_, re_, n_, r_
    x = r/re
    p = 1-0.6097/n+0.05563/n**2
    b = gammaincinv(2*n,0.5)
    M_r = 2*M*gammainc((3-p)*n, b*x**(1/n))/np.power(b, (1-p)*n)/gamma(2*n)
    return M_r
'''
def V_b(r_, M_, re_, n_):
    # Calculate the rotation velocity for Sersic bulge
    M, re, n = M_, re_, n_
    r = r_
    M_total = bulge_mass(M, re, n, r)
    M_bulge = M_total
    V = np.sqrt(c.G*M_bulge*u.Unit('M_sun')/r/u.Unit('kpc')).to('km/s').value
    return V

## Estimate the rotation velocity for exponential disk
from scipy.special import i0,i1,k0,k1
def Phi_d(r_, Md_, rd_):
    r, Md, rd = r_*u.kpc, 10**Md_*u.Unit('M_sun'), rd_*u.kpc/1.68
    den = Md/(2*np.pi*rd**2)
    y = r/rd/2
    Phi0 = -np.pi*c.G*den*2*y*rd*(i0(y)*k1(y)-i1(y)*k0(y))
    return Phi0.to('km^2/s^2')

def disk_mass(r_, Md_, rd_):
    r, Md, rd = r_*u.kpc, 10**Md_*u.Unit('M_sun'), rd_*u.kpc/1.68
    den = Md/(2*np.pi*rd**2) #calculate the surface density of disk at effective radius
    M = 2*np.pi*den*rd**2*(1-np.exp(-r/rd)*(1+r/rd))
    return M

def V_d(r_, Md_, rd_): #Md is the total mass of disk
    r, Md, rd = r_*u.kpc, 10**Md_*u.Unit('M_sun'), rd_*u.kpc/1.68 
    #The 1.68 factor is used to convert effective radius to scale length
    den = Md/(2*np.pi*rd**2) #calculate the surface density of disk at effective radius
    y = r/rd/2
    vc2 = 4*np.pi*c.G*den*rd*y**2*(i0(y)*k0(y)-i1(y)*k1(y))
    return np.sqrt(vc2).to('km/s').value


## Estimate the rotation velocity for dark matter halo
from astropy.modeling.physical_models import NFW

def Phi_dh(r_, M_, con_, z_):
    r, M, con = r_, 10**M_, con_
    redshift = z_
    concentration = con#np.power(10, a+b*np.log10(M/1e12))
    cosmo = Planck15
    massfactor = ("critical", 200)
    mass = u.Quantity(M, u.M_sun)
    # Create NFW Object
    n = NFW(mass=mass, concentration=concentration, redshift=redshift, cosmo=cosmo,
            massfactor=massfactor)
    radii = r*u.kpc
    V_c = n.circular_velocity(radii).to('km/s')
    Phi0 = -V_c**2
    return Phi0

def V_dh(r_, M_, con_):
    r, M, con = r_, 10**M_, con_
    redshift = 0.06115
    b = -0.101+0.026*redshift
    a = 0.520+(0.905 - 0.520)*np.exp(-0.617*redshift**1.21)
    concentration = con#np.power(10, a+b*np.log10(M/1e12))#
    cosmo = Planck15
    massfactor = ("critical", 200)
    mass = u.Quantity(M, u.M_sun)
    # Create NFW Object
    n = NFW(mass=mass, concentration=concentration, redshift=redshift, cosmo=cosmo,
            massfactor=massfactor)
    radii = r*u.kpc
    V_c = n.circular_velocity(radii).to('km/s')
    return V_c.value
    
# %%
## Calculate the rotation curve for FLATTENED bulges
from scipy import integrate
## Estimate the rotation curve in a FLATTENED Sersic bulge
G = 4.32e-06    ## The gravitational constant
Sigma0, r_e, n = 5e9/21.7923, 1., 4.  ## The parameters that are used to describe the surface density
## For differen Sersic index, the normalization of total mass are:
## n=0.5, 9.0647
## n=1.0, 11.9485
## n=2.0, 16.2605
## n=3.0, 19.3932
## n=4.0, 21.7923
## n=5.0, 23.7364

bn = gammaincinv(2*n, 0.5)
# %%
inc = 1.        ## The inclination angle between line-of-sight and main-axis of rotation
#q = 0.2        ## The intrinsic axis ratio between major and minor axis radius
#e2 = 1-q**2    ## The intrinsic ellipcity

def f(r, a):
    f1 = Sigma0/(n*r_e)*bn*(r/r_e)**(1/n-1)*np.exp(-bn*((r/r_e)**(1/n)-1))/np.sqrt(r**2-a**2)
    f2 = a**2/np.sqrt(R**2 - a**2*e2)
    return f1*f2
def bounds_r(a):
    return [a, np.inf]
def bounds_a():
    return [0, R]

Rs = np.linspace(0, 10., 100)
vs2 = np.zeros(len(Rs))
vs2e = np.zeros(len(Rs))

qs = [0., 0.2, 0.4, 0.6, 0.8, 1.0]
#ns = [0.5, 1., 2., 3., 4., 5.]
#q = 1.
vc = []
for j in range(len(qs)):
    q = qs[j]
    e2 = 1-q**2
    for i in range(len(Rs)):
        R = Rs[i]
        vs2[i] = integrate.nquad(f, [bounds_r, bounds_a])[0]*4*G#*q*np.sqrt(np.sin(inc)**2 + np.cos(inc)**2/q**2)
        vs2e[i] =integrate.nquad(f, [bounds_r, bounds_a])[1]
    vs = np.sqrt(vs2)
    vc.append(vs)

# %%
## Estimate the velocity through mass
def M(s): 
    ## Estimate the mass within given radius
    f = Sigma0*np.exp(-bn*((s/r_e)**(1/n)-1))*2*np.pi*s
    return f*G
def bounds_s():
    return [0, R]

vs2s = np.zeros(len(Rs))
for i in range(len(Rs)):
    R = Rs[i]
    vs2s[i] = integrate.nquad(M, [bounds_s])[0]/R
vss = np.sqrt(vs2s)
vs2s*Rs/G
# %%
Md, rd = np.log10(5e9), 1.
vdisk = V_d(Rs, Md, rd)
vbulge = V_b(Rs, np.log10(5e9), 1, 4)

plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

lws = np.array([1.2, 1.0, 0.8, 0.6, 0.4, 0.2])*1.5
plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
for i in range(len(vc)):
    ax.plot(Rs, vc[i], 'k', lw=lws[i])
ax.plot(Rs, vss, 'k--', label='Mass')
ax.plot(Rs, np.sqrt(G*5e9/Rs), 'k:', label='Point')
ax.plot(Rs, vbulge, 'b:', label='Bulge')
#ax.plot(Rs, vdisk, 'r:', label='Disk', zorder=2, lw=2.4)
ax.set_xlim(0., 10.)
ax.set_ylim(0, 150)
ax.set_xlabel("$R/R_e$")
ax.set_ylabel("$V_c$ [km/s]")
plt.legend()
plt.savefig("/home/qyfei/Desktop/Codes/CODES/Dynamics/RCs_spherical.pdf", bbox_inches="tight", dpi=300)

# %%
## Load parameters
path = '/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/'
folder = "Barolo_fit/output/PG0050+124_best/"
file = "ringlog2.txt"
r_fit,rad_fit,vrot_fit,disp_fit,inc,pa,z0,xpos,ypos,vsys,vrad, evrot1_fit, evrot2_fit, edisp1_fit, edisp2_fit = np.genfromtxt(path+folder+file,skip_header=1,usecols=(0,1,2,3,4,5,7,9,10,11,12,13,14,15,16),unpack=True) 

# %%
## Apply the rotation curve on real observation
## Inner component
z = 0.061
DL = Planck15.luminosity_distance(z)
nu_obs = 230.58/(1+z)#230.58 # Convert to CO(1-0)
I0i = 4.510 # convert to CO(1-0)
L0i = 3.25e7*I0i*DL.value**2/(1+z)**3/nu_obs**2/0.9     #/0.18899035 ## comvert to K km/s pc^2/beam
rei = 0.322/Planck15.arcsec_per_kpc_proper(z).value     #*(1-0.650)
ni = 1.829
bni = gammaincinv(2*ni, 0.5)

L0i = 1096.342/0.9*1e6                                  # convert into M_sun/kpc^2
rei = 0.322/Planck15.arcsec_per_kpc_proper(z).value*(1-0.650)
ni = 1.828
bni = gammaincinv(2*ni, 0.5)

Sigma0, r_e, n, bn = L0i, rei, ni, bni
q = 1
e2 = 1-q**2

def f(r, a):
    f1 = Sigma0/(n*r_e)*bn*(r/r_e)**(1/n-1)*np.exp(-bn*((r/r_e)**(1/n)-1))/np.sqrt(r**2-a**2)
    f2 = a**2/np.sqrt(R**2 - a**2*e2)
    return f1*f2
def bounds_r(a):
    return [a, np.inf]
def bounds_a():
    return [0, R]

vs2 = np.zeros(len(r_fit))
vs2e = np.zeros(len(r_fit))
for i in range(len(r_fit)):
    R = r_fit[i]
    vs2[i] = integrate.nquad(f, [bounds_r, bounds_a])[0]*4*G
    vs2e[i] =integrate.nquad(f, [bounds_r, bounds_a])[1]
vs_in = np.sqrt(vs2)

# %%
## Outer component
z = 0.061
DL = Planck15.luminosity_distance(z)
nu_obs = 230.58/(1+z)#230.58
I0o = 0.420
L0o = 3.25e7*I0o*DL.value**2/(1+z)**3/nu_obs**2/0.62
reo = 1.329/Planck15.arcsec_per_kpc_proper(z).value
no = 0.456
bno = gammaincinv(2*no, 0.5)

I0o = 102.096
L0o = 102.096/0.62*1e6    
reo = 1.329/Planck15.arcsec_per_kpc_proper(z).value
no = 0.456
bno = gammaincinv(2*no, 0.5)

Sigma0, r_e, n, bn = L0o, reo, no, bno
q = 0.1
e2 = 1-q**2

def f(r, a):
    f1 = Sigma0/(n*r_e)*bn*(r/r_e)**(1/n-1)*np.exp(-bn*((r/r_e)**(1/n)-1))/np.sqrt(r**2-a**2)
    f2 = a**2/np.sqrt(R**2 - a**2*e2)
    return f1*f2
def bounds_r(a):
    return [a, np.inf]
def bounds_a():
    return [0, R]

vs2 = np.zeros(len(r_fit))
vs2e = np.zeros(len(r_fit))
for i in range(len(r_fit)):
    R = r_fit[i]
    vs2[i] = integrate.nquad(f, [bounds_r, bounds_a])[0]*4*G#*q*np.sqrt(np.sin(inc)**2 + np.cos(inc)**2/q**2)
    vs2e[i] =integrate.nquad(f, [bounds_r, bounds_a])[1]
vs_out = np.sqrt(vs2)

# %%

Mb, re, n, Md, rd, Mdh, con = 10.96, 1.617, 1.69, 10.64, 10.968, 15, 8

a = 0.8
#a1, a2 = 3.1/beam_area, 3.1/beam_area
#beam_area = 0.18899035
plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
v_g = np.sqrt(vs_in**2*a + vs_out**2*a)
ax.errorbar(r_fit, v_g, fmt='ko', mfc='none')

ax.errorbar(r_fit, vs_in*np.sqrt(a), fmt='ko', mfc='none')
ax.errorbar(r_fit, vs_out*np.sqrt(a), fmt='ks', mfc='none')
ax.set_xlim(0, 3.1)
ax.set_ylim(0, 380)

v_tot2 = V_b(r_fit, Md, re, n)**2 + V_d(r_fit, Md, rd)**2 + V_dh(r_fit, Mdh, con)**2 + v_g**2
v_ad = v_ac2([Md, re, n, Md, rd, Mdh, con, a], r_fit[:], disp_fit[:])
v_tot2 + v_ad

# %%
