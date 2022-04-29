# %%
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colorbar as colorbar
from matplotlib.transforms import Affine2D
from astropy.visualization.wcsaxes import WCSAxes
import astropy.units as u
from scipy.misc.common import derivative
from spectral_cube import SpectralCube
from astropy.modeling import models, fitting
import astropy.constants as c
import astropy.units as u
from scipy.special import gamma
from scipy.special import gammaincinv
from scipy.special import gammainc
from astropy.cosmology import Planck15
from scipy import integrate

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
## Estimate the rotation curve in a FLATTENED Sersic bulge
# G = 4.32e-06    ## The gravitational constant
# Sigma0, r_e, n = 5e9/21.7923, 1., 4.  ## The parameters that are used to describe the surface density
# ## For differen Sersic index, the normalization of total mass are:
# ## n=0.5, 9.0647
# ## n=1.0, 11.9485
# ## n=2.0, 16.2605
# ## n=3.0, 19.3932
# ## n=4.0, 21.7923
# ## n=5.0, 23.7364

# bn = gammaincinv(2*n, 0.5)

# # %%
# inc = 1.        ## The inclination angle between line-of-sight and main-axis of rotation
# #q = 0.2        ## The intrinsic axis ratio between major and minor axis radius
# #e2 = 1-q**2    ## The intrinsic ellipcity

# def f(r, a):
#     f1 = Sigma0/(n*r_e)*bn*(r/r_e)**(1/n-1)*np.exp(-bn*((r/r_e)**(1/n)-1))/np.sqrt(r**2-a**2)
#     f2 = a**2/np.sqrt(R**2 - a**2*e2)
#     return f1*f2
# def bounds_r(a):
#     return [a, np.inf]
# def bounds_a():
#     return [0, R]

# Rs = np.linspace(0, 10., 2)
# vs2 = np.zeros(len(Rs))
# vs2e = np.zeros(len(Rs))
# #qs = [0., 0.2, 0.4, 0.6, 0.8, 1.0]
# #ns = [0.5, 1., 2., 3., 4., 5.]
# #q = 1.
# qs = [0.]
# vc = []
# for j in range(len(qs)):
#     q = qs[j]
#     e2 = 1-q**2
#     for i in range(len(Rs)):
#         R = Rs[i]
#         vs2[i] = integrate.nquad(f, [bounds_r, bounds_a])[0]*4*G#*q*np.sqrt(np.sin(inc)**2 + np.cos(inc)**2/q**2)
#         vs2e[i] =integrate.nquad(f, [bounds_r, bounds_a])[1]
#     vs = np.sqrt(vs2)
#     vc.append(vs)

# # %%
# ## Estimate the velocity through mass
# def M(s): 
#     ## Estimate the mass within given radius
#     f = Sigma0*np.exp(-bn*((s/r_e)**(1/n)-1))*2*np.pi*s
#     return f*G
# def bounds_s():
#     return [0, R]

# vs2s = np.zeros(len(Rs))
# for i in range(len(Rs)):
#     R = Rs[i]
#     vs2s[i] = integrate.nquad(M, [bounds_s])[0]/R
# vss = np.sqrt(vs2s)
# vs2s*Rs/G
# # %%
# Md, rd = np.log10(5e9), 1.
# vdisk = V_d(Rs, Md, rd)
# vbulge = V_b(Rs, np.log10(5e9), 1, 4)

# plt.rc('text', usetex=True)
# plt.rc('font', family='dejavuserif', size=25)
# plt.rc('xtick', direction='in', top=True)
# plt.rc('ytick', direction='in', right=True)

# lws = np.array([1.2, 1.0, 0.8, 0.6, 0.4, 0.2])*1.5
# plt.figure(figsize=(8, 8))
# ax = plt.subplot(111)
# for i in range(len(vc)):
#     ax.plot(Rs, vc[i], 'k', lw=lws[i])
# ax.plot(Rs, vss, 'k--', label='Mass')
# ax.plot(Rs, np.sqrt(G*5e9/Rs), 'k:', label='Point')
# ax.plot(Rs, vbulge, 'b:', label='Bulge')
# #ax.plot(Rs, vdisk, 'r:', label='Disk', zorder=2, lw=2.4)
# ax.set_xlim(0., 10.)
# ax.set_ylim(0, 150)
# ax.set_xlabel("$R/R_e$")
# ax.set_ylabel("$V_c$ [km/s]")
# plt.legend()
# #plt.savefig("/home/qyfei/Desktop/Codes/CODES/Dynamics/RCs_spherical.pdf", bbox_inches="tight", dpi=300)

# # %%
# ## Load parameters
# path = '/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/'
# folder = "Barolo_fit/output/PG0050+124_best/"
# file = "ringlog2.txt"
# r_fit,rad_fit,vrot_fit,disp_fit,inc,pa,z0,xpos,ypos,vsys,vrad, evrot1_fit, evrot2_fit, edisp1_fit, edisp2_fit = np.genfromtxt(path+folder+file,skip_header=1,usecols=(0,1,2,3,4,5,7,9,10,11,12,13,14,15,16),unpack=True) 

# folder = "Barolo_fit/output/PG0050+124_trash/"
# file = "ringlog2.txt"
# r_fit = np.loadtxt(path+folder+file)[:,0]


# # %%
# ## Bar contribution
# #r_fit = np.linspace(0.1, 10, 20)
# ## Apply the rotation curve on real observation
# ## Inner component
# z = 0.061
# DL = Planck15.luminosity_distance(z)
# nu_obs = 230.58/(1+z)#230.58 # Convert to CO(1-0)
# I0i = 4.510 # convert to CO(1-0)
# L0i = 3.25e7*I0i*DL.value**2/(1+z)**3/nu_obs**2/0.9     #/0.18899035 ## comvert to K km/s pc^2/beam
# rei = 0.322/Planck15.arcsec_per_kpc_proper(z).value     #*(1-0.650)
# ni = 1.829
# bni = gammaincinv(2*ni, 0.5)

# L0i = 1096.342/0.9*1e6                                  # convert into M_sun/kpc^2
# rei = 0.322/Planck15.arcsec_per_kpc_proper(z).value
# ni = 1.828
# bni = gammaincinv(2*ni, 0.5)

# ## Bar component
# L0i = 577.861/0.9*1e6                                  # convert into M_sun/kpc^2
# rei = 0.494/Planck15.arcsec_per_kpc_proper(z).value    #0.494
# ni = 0.304
# bni = gammaincinv(2*ni, 0.5)

# Sigma0, r_e, n, bn = L0i, rei, ni, bni
# q = 0.1
# e2 = 1-q**2

# def f(r, a):
#     f1 = Sigma0/(n*r_e)*bn*(r/r_e)**(1/n-1)*np.exp(-bn*((r/r_e)**(1/n)-1))/np.sqrt(r**2-a**2)
#     f2 = a**2/np.sqrt(R**2 - a**2*e2)
#     return f1*f2
# def bounds_r(a):
#     return [a, np.inf]
# def bounds_a():
#     return [0, R]

# r_fit_tot = r_fit#[4:17]

# vs2 = np.zeros(len(r_fit_tot))
# vs2e = np.zeros(len(r_fit_tot))
# for i in range(len(r_fit_tot)):
#     R = r_fit_tot[i]
#     vs2[i] = integrate.nquad(f, [bounds_r, bounds_a])[0]*4*G
#     vs2e[i] =integrate.nquad(f, [bounds_r, bounds_a])[1]
# vs_in = np.sqrt(vs2)
# vs_in

# '''
# x_test = np.linspace(0, 13, 100)
# vs2_test = np.zeros(len(x_test))
# vs2e_test = np.zeros(len(x_test))
# for i in range(len(x_test)):
#     R = x_test[i]
#     vs2_test[i] = integrate.nquad(f, [bounds_r, bounds_a])[0]*4*G
#     vs2e_test[i] =integrate.nquad(f, [bounds_r, bounds_a])[1]
# vs_in_test = np.sqrt(vs2_test)'''
# #vs_in

# # %%
# ## Disk contribution
# ## Outer component
# z = 0.061
# DL = Planck15.luminosity_distance(z)
# nu_obs = 230.58/(1+z)#230.58
# I0o = 0.420
# L0o = 3.25e7*I0o*DL.value**2/(1+z)**3/nu_obs**2/0.62
# reo = 1.329/Planck15.arcsec_per_kpc_proper(z).value
# no = 0.456
# bno = gammaincinv(2*no, 0.5)

# I0o = 102.096
# L0o = 102.096/0.62*1e6    
# reo = 1.329/Planck15.arcsec_per_kpc_proper(z).value
# no = 0.456
# bno = gammaincinv(2*no, 0.5)

# ## Disk component
# I0o = 106.093
# L0o = 106.093/0.62*1e6    
# reo = 1.297/Planck15.arcsec_per_kpc_proper(z).value
# no = 0.476
# bno = gammaincinv(2*no, 0.5)

# Sigma0, r_e, n, bn = L0o, reo, no, bno
# e2 = 1-q**2

# def f(r, a):
#     f1 = Sigma0/(n*r_e)*bn*(r/r_e)**(1/n-1)*np.exp(-bn*((r/r_e)**(1/n)-1))/np.sqrt(r**2-a**2)
#     f2 = a**2/np.sqrt(R**2 - a**2*e2)
#     return f1*f2
# def bounds_r(a):
#     return [a, np.inf]
# def bounds_a():
#     return [0, R]

# vs2 = np.zeros(len(r_fit_tot))
# vs2e = np.zeros(len(r_fit_tot))
# for i in range(len(r_fit_tot)):
#     R = r_fit_tot[i]
#     vs2[i] = integrate.nquad(f, [bounds_r, bounds_a])[0]*4*G#*q*np.sqrt(np.sin(inc)**2 + np.cos(inc)**2/q**2)
#     vs2e[i] =integrate.nquad(f, [bounds_r, bounds_a])[1]
# vs_out = np.sqrt(vs2)
# vs_out
# '''
# x_test = np.linspace(0, 13, 100)
# vs2_test = np.zeros(len(x_test))
# vs2e_test = np.zeros(len(x_test))
# for i in range(len(x_test)):
#     R = x_test[i]
#     vs2_test[i] = integrate.nquad(f, [bounds_r, bounds_a])[0]*4*G
#     vs2e_test[i] =integrate.nquad(f, [bounds_r, bounds_a])[1]
# vs_out_test = np.sqrt(vs2_test)'''

# # %%
# ## Point source component
# A = 7302.437/0.9*1e6
# xstd = 0.093/Planck15.arcsec_per_kpc_proper(z).value
# ystd = 0.052/Planck15.arcsec_per_kpc_proper(z).value
# area = np.pi*xstd*ystd*2
# M_p = 3.26e8
# vs_p = np.sqrt(G*M_p/r_fit_tot)
# #vs_p_test = np.sqrt(G*M_p/x_test)

# # %%

# Mb, re, n, Md, rd, Mdh, con = 10.96, 1.617, 1.69, 10.64, 10.968, 15, 8

# a = 1#3.1
# #a1, a2 = 3.1/beam_area, 3.1/beam_area
# #beam_area = 0.18899035
# plt.figure(figsize=(8, 8))
# ax = plt.subplot(111)
# v_g = np.sqrt(vs_in**2 + vs_out**2 + vs_p**2)# + vs_p**2
# #v_g_test = np.sqrt(vs_in_test**2 + vs_out_test**2 + vs_p_test**2)

# ax.plot(r_fit_tot, v_g*np.sqrt(a), "k", label="Total")
# #ax.plot(x_test, v_g_test*np.sqrt(a), "k:")

# #ax.plot(r_fit, vs_bar*np.sqrt(a), "k:", label="Bar")
# ax.plot(r_fit_tot, vs_in*np.sqrt(a), "b:", label="Inner")
# ax.plot(r_fit_tot, vs_out*np.sqrt(a), "k--", label="Disk")
# ax.plot(r_fit_tot, vs_p*np.sqrt(a), "k-.", label="Point")
# #ax.plot(r_fit_tot, np.sqrt(G*a*10**9.75/r_fit_tot), "k", lw=3, alpha=0.5, label="Total Point")
# #ax.plot(x_test, np.sqrt(G*a*10**9.75/x_test), "k", lw=3, alpha=0.5, label="Total Point")

# ax.errorbar(r_fit, vrot_fit, yerr=[-evrot1_fit, evrot2_fit], fmt='bo', mfc='blue', ms=10, mew=1, elinewidth=1, capsize=5, label="$v_\mathrm{rot}$")
# ax.errorbar(r_fit[4:17], vrot_fit[4:17], yerr=[-evrot1_fit[4:17], evrot2_fit[4:17]], fmt='rs', mfc='red', ms=10, mew=1, elinewidth=1, capsize=5, label="fit points")

# ax.vlines(0.43, 0, 400, color='k', ls=':')
# ax.vlines(0.86, 0, 400, color='k', ls=':')
# ax.vlines(2.10, 0, 400, color='k', ls=':')

# ax.set_xlim(0, 3)
# ax.set_ylim(0, 380)
# ax.set_xlabel("Radius [kpc]")
# ax.set_ylabel("$V$ [$\mathrm{km\,s^{-1}}$]")
# v_tot2 = V_b(r_fit_tot, Md, re, n)**2 + V_d(r_fit_tot, Md, rd)**2 + V_dh(r_fit_tot, Mdh, con)**2 + v_g**2
# #v_ad = v_ac2([Md, re, n, Md, rd, Mdh, con, a], r_fit[:], disp_fit[:])
# #v_tot2 + v_ad
# plt.legend(fontsize=25)

# #plt.savefig("/home/qyfei/Desktop/Results/Barolo/PG0050/RC.pdf", bbox_inches="tight", dpi=300)
# # %%

# v_b_test = V_b(x_test, para_out[0], para_out[1], para_out[2]) #para_out[0], para_out[1], para_out[2]
# v_d_test = V_d(x_test, para_out[3], para_out[4]) #para_out[3], para_out[4]
# #log_Mdh = np.log10((10**para_out[0]+10**para_out[3]+para_out[7]*1.36*L_CO)*(1/10**para_out[5]-1))
# v_dh_test = V_dh(x_test, para_out[5], para_out[6])
# #v_g = V_gas*np.sqrt(para_out[7])
# plt.figure(figsize=(8, 8))
# ax = plt.subplot(111)
# ax.plot(x_test, v_b_test, "red", lw=2, label="Bulge")
# ax.plot(x_test, v_d_test, "blue", lw=2, label="Disk")
# ax.plot(x_test, v_dh_test, "yellow", lw=2, label="DM")
# ax.plot(x_test, v_g_test*np.sqrt(alpha_CO), "green", lw=2, label="Gas")

# v_tot_test = np.sqrt(v_b_test**2 + v_d_test**2 + v_dh_test**2 + v_g_test**2*alpha_CO)
# ax.plot(x_test, v_tot_test, "Grey", lw=5, zorder=3, label="Total")
# #ax.plot(r_fit_tot, np.sqrt(v_b**2+v_d**2+v_dh**2+v_g**2*alpha_CO + disp_tot**2*(asy0+asy1)), "Grey", lw=5, zorder=3, label='Total')

# #ax.errorbar(rfit2D[5:], Vrot2D[5:], yerr=evrot2D[5:], fmt='ro', mfc='r', ms=8, mew=1, elinewidth=1, capsize=4, label=r'$\mathrm{H}\alpha$')
# ax.errorbar(r_fit, vrot_fit, yerr=[-evrot1_fit, evrot2_fit], fmt='bo', mfc='b', ms=10, mew=1, elinewidth=1, capsize=5, label='CO')
# ax.errorbar(r_fit_tot, vrot_tot, yerr=[-evrot1_fit[7:17], evrot2_fit[7:17]], fmt='ks', mfc='none', ms=10, mew=1, elinewidth=1, capsize=5)

# ax.vlines(0.86, 0, 400, color='k', ls=':')
# ax.vlines(2.1, 0, 400, color='k', ls=':')
# ax.vlines(3, 0, 400, color='k', ls=':')
# ax.set_xlim(0, 13)
# ax.set_ylim(0, 360)
# ax.set_xlabel('Radius [kpc]')
# ax.set_ylabel(r'$V_\mathrm{rot}$ [$\mathrm{km\,s^{-1}}$]')
# plt.legend(fontsize=18, loc="lower right")


# %%

# Mb, re, n, Md, rd = 10.96, 1.62, 1.69, 10.64, 10.97
# v_b = V_b(r_fit, Mb, re, n)
# v_d = V_d(r_fit, Md, rd)

# plt.figure(figsize=(8, 8))
# ax = plt.subplot(111)

# ax.errorbar(r_fit, vrot_fit, yerr=[-evrot1_fit, evrot2_fit], fmt='bs', mfc='none', ms=8, mew=1, elinewidth=1, capsize=4, label="$v_\mathrm{rot}$")

# ax.errorbar(r_fit, vcirc, yerr=[evcirc1_fit, evcirc2_fit], fmt='ks', mfc='none', ms=8, mew=1, elinewidth=1, capsize=4, label="$v_\mathrm{circ}$")

# #ax.plot(r_fit, v_b, 'm')
# #ax.plot(r_fit, v_d, 'g')

# ax.set_xlim(0, r_fit[-1]+0.15)
# ax.set_ylim(0, 380)
# ax.set_xlabel('radius [kpc]')
# ax.set_ylabel(r'$V_\mathrm{circ}$ [$\mathrm{km\,s^{-1}}$]')
# plt.legend()

#plt.savefig("/home/qyfei/Desktop/Results/Dynamics/results/PG0050/asym_drift/RC_ad_03.pdf", bbox_inches="tight", dpi=300)

# %%
