# %%
from matplotlib.colors import LogNorm
from sklearn.preprocessing import scale
from Physical_values.surface_density import surface_density, iso_rad, surface_density_mom0
from map_visualization.maps import *
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15
from matplotlib import colors

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

# %%
## Here we provide how to evaluate the surface density of this galaxy

def iso_rad(sizex_, sizey_, pos_cen_, pix_size_, PA_, inc_):
    sizex, sizey, pos_cen, pix_size, PA, inc = sizex_, sizey_, pos_cen_, pix_size_, PA_, inc_
    # This function calculate the radius between each pixel and the kinematic center
    # size, pos_cen, pix_size are adopted from observation, which are map size, coordinates of galaxy center and size of each pixel
    # PA, inc are position angle and inclination angle
    z = 0.029
    yy,xx = np.indices([sizey, sizex],dtype='float')
    coordinates_xx = (xx-pos_cen[1])*np.cos(PA*u.deg).value + (yy-pos_cen[0])*np.sin(PA*u.deg).value
    coordinates_yy = -(xx-pos_cen[1])*np.sin(PA*u.deg).value + (yy-pos_cen[0])*np.cos(PA*u.deg).value
    Radius_pixel = np.sqrt(coordinates_xx**2 + coordinates_yy**2/(np.cos(inc*u.deg).value)**2)
    Radius = (Radius_pixel * pix_size*u.arcsec/Planck15.arcsec_per_kpc_proper(z)).to('kpc')
    return Radius.value

def radius(x_, y_, pos_cen_, pix_size_, PA_, inc_):
    x, y, pos_cen, pix_size, PA, inc = x_, y_, pos_cen_, pix_size_, PA_, inc_
    z = 0.029
    #yy,xx = np.indices([size, size],dtype='float')
    coordinates_xx = (x-pos_cen[1])*np.cos(PA*u.deg).value + (y-pos_cen[0])*np.sin(PA*u.deg).value
    coordinates_yy = -(x-pos_cen[1])*np.sin(PA*u.deg).value + (y-pos_cen[0])*np.cos(PA*u.deg).value
    Radius_pixel = np.sqrt(coordinates_xx**2 + coordinates_yy**2/(np.cos(inc*u.deg).value)**2)
    Radius = (Radius_pixel * pix_size*u.arcsec/Planck15.arcsec_per_kpc_proper(z)).to('kpc')
    return Radius.value

def surface_density_mom0(mom0_, hdu_):
    mom0, hdu = mom0_, hdu_
    alpha_CO = 4.36      #The conversion factor between flux and mass
    z = 0.029          #The redshift of the galaxy
    DL = Planck15.luminosity_distance(z) # The luminosity distance
    nu_obs = 230.58/(1+z) #The observation frequency
    inc = 40*u.deg      #The inclination angle of the galaxy
    pix_size = 0.18
    bmaj = hdu.header['BMAJ']
    bmin = hdu.header['BMIN']
    delt = hdu.header['CDELT1']
    CO_pix = np.pi*bmaj*bmin/(4*np.log(2)*delt**2) #Derive how many pixels are contained by the beam
    # Estimate the luminosity and mass of molecular gas
    # 0.62 is the ratio between different rotational transition line and CO(1-0)
    L_CO21 = 3.25e7*mom0*DL.value**2/((1+z)**3*nu_obs**2)
    L_CO10 = L_CO21/0.62
    M_H2 = alpha_CO*L_CO10*u.Unit('M_sun')
    # The 1.36 denote the Helium contribution
    # 1e6 suggests the final unit is M_sun/pc^2
    Sigma_H2 = M_H2/CO_pix/(pix_size*u.arcsec/Planck15.arcsec_per_kpc_proper(z))**2*np.cos(inc)/1e6
    return Sigma_H2.value

# %%
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG0923/data"

file_mom0 = "/PG0923+129_CO21_final_image_mom0.fits"
hdu = fits.open(path+file_mom0)[0]
mom0 = fits.open(path+file_mom0)[0].data[0][0]
file_mom2 = "/PG0923+129_CO21_final_image_mom2.fits"
mom2 = fits.open(path+file_mom2)[0].data[0][0]

# %%
rms = 0.055 ## The rms noise of intensity map
radius = iso_rad(185, 185, np.array([89.66, 88.40]), 0.18, 260-90, 40)

Sigma_H2 = surface_density_mom0(mom0, hdu)
Sigma_rms = surface_density_mom0(2.*rms, hdu)

G = 4.302*10**(-3) #The gravitational constan, in the units of 

# %%
pos_cen = np.array([89.66, 88.40])
fov = 50
pix_size = 0.18
levels = np.array([-1,1,2,4,8,16,32])*2*rms
plt.figure(figsize=(8, 10))
ax = plt.subplot(111)
im = ax.imshow(mom0, cmap='magma', origin='lower', norm=colors.PowerNorm(0.5))
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label('CO(2-1) [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]')
## show the CO intensity, with colorbar

ax.contour(mom0, levels, linewidths=0.8, colors=['k'])
ax.contour(mom2, levels=np.arange(10, 120, 10), linewidths=0.8, colors=['cyan'])
## overplot the intensity map contour and velocity dispersion map

ax.contour(radius, levels=np.array([0.6, 1.4]), linewidths=1.0, colors=['w'])
## represent the radius defined by the inclination angle and galaxy center
ax.set_xticks([])
ax.set_yticks([])

ax.scatter(len(mom0)/2, len(mom0)/2, s=200, marker='+', color='k', label='PC')
ax.scatter(88.40, 89.66, s=200, marker='x', color='b', label='KC')
## represent the center with different method

bmaj, bmin, bpa, delt = hdu.header['BMAJ'], hdu.header['BMIN'], hdu.header['BPA'], hdu.header['CDELT1']
rec_size = abs(bmaj/delt)*1.5

rec = matplotlib.patches.Rectangle((pos_cen[0]-fov, pos_cen[1]-fov), rec_size, rec_size, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax.add_artist(rec)
Beam = beam(hdu, pos_cen[0]-fov+rec_size/2,pos_cen[1]-fov+rec_size/2, 'k', pix_size)
ax.add_artist(Beam[0])
## represent the beam at the lower left corner of the plot

scale_bar_length = 2*u.kpc*Planck15.arcsec_per_kpc_proper(0.029)
bar_size = abs(scale_bar_length.to('deg').value/delt)*1.5
rec = matplotlib.patches.Rectangle((pos_cen[0]+fov-bar_size, pos_cen[1]-fov), bar_size, rec_size, angle=0.0, fill=True, edgecolor='k', facecolor='w', zorder=2)
ax.add_artist(rec)
bar_length = abs(scale_bar_length.to('deg').value/delt)
scale_bar = matplotlib.patches.Rectangle((pos_cen[0]+fov-(bar_length+bar_size)/2, pos_cen[1]-fov+rec_size*3/5), bar_length, 2, angle=0.0, fill=True, edgecolor='k', facecolor='k', zorder=3)
ax.add_artist(scale_bar)
ax.text(pos_cen[0]+fov-bar_size/2, pos_cen[1]-fov+rec_size/6, '2kpc', ha='center')
## represent the scale bar at the lower right corner of the plot

ax.set_xlim(pos_cen[0]-fov,pos_cen[0]+fov)
ax.set_ylim(pos_cen[1]-fov,pos_cen[1]+fov)

# plt.savefig("/home/qyfei/Desktop/Results/Physical_values/Equilibrium/PG0923_show.pdf", bbox_inches="tight", dpi=300)

# %%
N = np.where((1-np.isnan(Sigma_H2))*(1-np.isnan(mom2)))
SN_threshold = np.where(mom0[N])# >= 2*rms)

conta = np.where((mom2[N][SN_threshold] >= 30) & (radius[N][SN_threshold] >= 2))

core = np.where((radius[N][SN_threshold]<=0.6))
ring = np.where((radius[N][SN_threshold]>0.6) & (radius[N][SN_threshold]<=1.4))
disk = np.where((radius[N][SN_threshold]>1.4))# & (radius[N][SN_threshold]<10))


plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
# This plot represents the relation between molecular gas surfaced density and velocity dispersion, traced by CO(2-1) emission line.
# Here different label represents different radius of pixel. 'core' means pixels radii<0.5 kpc, which locates within 1 beam. 'ring' represents the gas properties in ring region (0.5-1.5kpc). 'disk' represents the gas properties in main disk region (>1.5kpc).
ax.scatter(Sigma_H2[N][SN_threshold][core], mom2[N][SN_threshold][core], s=25, c='royalblue', edgecolors='royalblue', linewidth=2, label='core')
ax.scatter(Sigma_H2[N][SN_threshold][ring], mom2[N][SN_threshold][ring], s=25, c='silver', edgecolors='silver', linewidth=2, label='ring')
ax.scatter(Sigma_H2[N][SN_threshold][disk], mom2[N][SN_threshold][disk], s=25, c='navajowhite', edgecolors='navajowhite', linewidth=2, label='disk')
ax.scatter(Sigma_H2[N][SN_threshold][conta], mom2[N][SN_threshold][conta], s=25, c='k', edgecolor='k', linewidth=2, label='noise')

x = np.logspace(0, 5, 1000)
yp = 10**0.85*(x/1e2)**0.47 

y_sigma1 = (G*np.pi/5)**0.5*400**0.5*x**0.5
y_sigma2 = (2*G*np.pi/5)**0.5*400**0.5*x**0.5
y_sigma_vertical1 = (x*np.pi*G*150*1.05/1.3)**0.5
y_sigma_vertical2 = (x*np.pi*G*190*1.05/1.3)**0.5

##The scaling relation between dispersion and surface density in Wilson et al. 2019
# ax.plot(x, yp, 'k', lw=3, alpha=0.7)
ax.plot(x, y_sigma_vertical1, 'k--')
ax.plot(x, y_sigma_vertical2, 'k--')


ax.set_xlabel(r"$\Sigma_\mathrm{H_2}$ [$\mathrm{M_\odot\,pc^{-2}}$]")
ax.set_ylabel(r"$\sigma_\mathrm{gas}$ [$\mathrm{km\,s^{-1}}$]")
ax.loglog()
ax.set_xlim(5, 4e2)
ax.set_ylim(7, 200)
# ax.hlines(11*2/np.sqrt(8*np.log(2)), 1e-3, 1e5, color='C1', lw=2)
CW = 10.1 #Channel width
ax.fill_between([0, 1e5], [CW*2/np.sqrt(8*np.log(2)), CW*2/np.sqrt(8*np.log(2))], color='C1', alpha=0.3)
ax.fill_between([0, Sigma_rms], [1e5, 1e5], color='C1', alpha=0.3)
plt.legend()
# plt.savefig("/home/qyfei/Desktop/Results/Physical_values/Equilibrium/PG0923_equilibrium.pdf", bbox_inches="tight", dpi=300)

# %%
