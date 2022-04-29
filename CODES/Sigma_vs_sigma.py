# %%
from matplotlib.colors import LogNorm
from Physical_values.surface_density import surface_density, iso_rad, surface_density_mom0
from map_visualization.maps import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

# %%
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
file = "PG0050_CO21-combine-line-10km-mosaic-mom0.fits"
file_pbc = "PG0050_CO21-combine-line-10km-mosaic-mom0-pbcor.fits"
file_mom2 = "PG0050_CO21-combine-line-10km-mosaic-mom2.fits"
radius = iso_rad(800, 800, np.array([395, 399]), 0.05, 35, 41)

G = 4.302*10**(-3)

from Physical_values.stellar_properties import Sigma_disk

Sigma, re = np.power(10, 10.64)/40.965, 10.97
Sigma_s = Sigma_disk(Sigma, radius, re)
disp_s = 30*2.

alpha_CO = 1.27

Sigma_H2, Sigma_H2r = surface_density(path, file_pbc, alpha_CO)
mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file)
print(r)
mom0_level = np.array([-1,2/3,1,2,4,8,16,32,64])*3*r
print(mom0_level)
mom0_rms, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file_pbc)
mom2, wcs, size, pix_size, hdu, pos_cen = load_mom2(path, file_mom2)
disp_CO = np.sqrt(mom2**2)# - 9.156**2

regions_name = ["BEAM", "CENTER", "CND", "SPIRAL"]
beamregion = radius<=0.4
center = (radius<=0.8) & (radius>0.4)
CND = (radius<=2.1) & (radius>0.8)
spiral = radius>2.1
regions = [beamregion, center, CND, spiral]
#fmts = ["yo", "co", "go"]
colors = ["lightcoral", "silver", "gold", "slategrey"]
mfcs = ["yellow", "cyan", "g"]
ecolors = ["lightcoral", "silver", "gold", "slategrey"]
zorders = [0.2, 0.1, 0.3, 0.1]

center_region = np.where(center>=1)
Sigma_H2[center_region] = Sigma_H2[center_region]*0.62/0.9
beam_region = np.where(beamregion>=1)
Sigma_H2[beam_region] = Sigma_H2[beam_region]*0.62/0.9

#disp_CO[center_region] = np.sqrt(disp_CO[center_region]**2 - 10**2)
Sigma_tot = Sigma_H2# + Sigma_s*2/(1+disp_s**2/disp_CO**2) ##constrain the total disk surface density

Sigma_tot_lowlim = Sigma_H2/1.27*(1.27-0.71) + Sigma_s*2/(1+disp_s**2/disp_CO**2)
Sigma_tot_uplim = Sigma_H2/1.27*(1.27+0.83) + Sigma_s*2/(1+disp_s**2/disp_CO**2)

Sigma_totr = Sigma_H2r# + Sigma_s*2/(1+disp_s**2/disp_CO**2)

# %%
colors = ["none", "royalblue", "silver", "navajowhite"]
ecolors = ["lightcoral", "blue", "silver", "navajowhite"]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
for i in range(len(regions)):
    Sigma_H2_region = Sigma_tot*regions[i]
    disp_CO_region = disp_CO*regions[i]
    mom0_region = mom0*regions[i]*alpha_CO
    Sigma_H2_region[np.isnan(Sigma_H2_region)] = 0
    disp_CO_region[np.isnan(disp_CO_region)] = 0
    mom0_region[np.isnan(mom0_region)] = 0
    region = np.where((mom0_region>0) & (Sigma_H2_region>0) & (disp_CO_region>0))
    ax.scatter(Sigma_H2_region[region], disp_CO_region[region], s=2, c=colors[i], edgecolors=ecolors[i], linewidth=2, zorder=zorders[i], label=regions_name[i])

bins = np.logspace(1., 3.6, 14)
Sigma_H2_mean = []
disp_H2_mean = []
disp_H2_disp = []
for i in range(len(bins)-1):
    N = np.where((Sigma_tot<bins[i+1])&(Sigma_tot>=bins[i]))
    Sigma_H2_mean.append(np.nanmean(Sigma_tot[N]))
    disp_H2_mean.append(np.nanmean(disp_CO[N]))
    disp_H2_disp.append(np.nanstd(disp_CO[N]))

from literature.load import *
dir = '/home/qyfei/Desktop/Codes/CODES/literature/'
Sigma_H2_Wilson, Disp_CO_Wilson = load_Wilson(dir)
Sigma_H2_Bolatto, Sigma_s_Levy, Disp_CO_Levy = load_Bolatto(dir)
Sigma_tot_Bolatto = Sigma_H2_Bolatto# + 2/(1+disp_s**2/Disp_CO_Levy**2)*Sigma_s_Levy

ax.errorbar(Sigma_H2_Wilson, Disp_CO_Wilson, fmt='ms', mfc='none', ms=6, mew=1, zorder=3, label='z$\sim$0 ULIRGs')
ax.errorbar(Sigma_tot_Bolatto, Disp_CO_Levy, fmt='bs', mfc='none', ms=6, mew=1, zorder=3, label='z$\sim$0 SFGs')
color = 'navy'
# ax.errorbar(Sigma_H2_mean, disp_H2_mean, yerr=disp_H2_disp, fmt="ko", mfc='none', ms=10, mew=1.5, capsize=5, zorder=4)

x = np.logspace(0, 5, 1000)
yp = 10**0.85*(x/1e2)**0.47 

y_sigma1 = (G*np.pi/5)**0.5*400**0.5*x**0.5
y_sigma2 = (2*G*np.pi/5)**0.5*400**0.5*x**0.5
y_sigma_vertical1 = (x*np.pi*G*150*1.05/1.3)**0.5
y_sigma_vertical2 = (x*np.pi*G*190*1.05/1.3)**0.5

##The scaling relation between dispersion and surface density in PHANGS-ALMA, Sun et al 2018
ax.plot(x, yp, 'k', lw=3, alpha=0.7)
#ax.plot(x, y_sigma1, 'k-.')
#ax.plot(x, y_sigma2, 'k--')
ax.plot(x, y_sigma_vertical1, 'k--')
ax.plot(x, y_sigma_vertical2, 'k--')

ax.hlines(9.156, 0, 1e8, color='k', linestyle=':')
ax.fill_between([0, Sigma_H2r], [3e2, 3e2], color='C1', alpha=0.3)


ax.errorbar(np.array([1.27*5e3]), np.array([16]), xerr=[np.array([(0.71)*5e3]), np.array([(0.83)*5e3])], yerr=[np.array([4.00]), np.array([4.2])], fmt='ks', mfc='k', ms=1, mew=1, zorder=3, capsize=10)
ax.arrow((1.27-0.71)*5e3, 11.5, (3.1-1.27)*5e3, 0, width=0.01, shape="full", overhang=1, head_width=1.5, head_length=3e3, facecolor='none')
ax.text((1.27-0.91)*5e3, 10, r"$\alpha_\mathrm{CO}$")
ax.text((3.1 -1.50)*5e3, 10, r"$\alpha_\mathrm{CO, trad}$")

ax.loglog()


ax.legend(loc='upper left', fontsize=20, frameon=True, framealpha=0.95)
ax.set_xlabel("$\Sigma_\mathrm{mol}$ [$\mathrm{M_\odot\,pc^{-2}}$]")
ax.set_ylabel("$\sigma$ [$\mathrm{km\,s^{-1}}$]")
ax.set_xlim(1e1, 3e4)
ax.set_ylim(7.5, 210)

# densprof = np.loadtxt('/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/Barolo_fit/output/PG0050+124_best/densprof.txt')
# radius = densprof[:,0]
# alpha_CO = 3.1
# from astropy.cosmology import Planck15
# DL = Planck15.luminosity_distance(0.061)
# z =  0.061
# nu_obs = 230.538/(1+z)
# MH2_sd = alpha_CO*3.25e7*densprof[:,7]*DL.value**2/((1+z)**3*nu_obs**2)/(1*u.arcsec/Planck15.arcsec_per_kpc_proper(0.061))**2*np.cos(41*u.deg)/1e6/0.62
# MH2_sd[:7] = MH2_sd[:7]*0.62/0.90

# err_MH2sd = alpha_CO*3.25e7*(densprof[:,7] + densprof[:,8])*DL.value**2/((1+z)**3*nu_obs**2)/(1*u.arcsec/Planck15.arcsec_per_kpc_proper(0.061))**2*np.cos(41*u.deg)/1e6 - MH2_sd/1e6
# ringlog = np.loadtxt('/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/Barolo_fit/output/PG0050+124_best/ringlog2.txt')
# rad = ringlog[:,0]
# vrot = ringlog[:,2]
# disp = ringlog[:,3]
# evrot1 = ringlog[:,13]
# evrot2 = ringlog[:,14]
# edisp1 = ringlog[:,15]
# edisp2 = ringlog[:,16]
#ax.errorbar(MH2_sd.value, disp, fmt='k*', mfc='none', ms=10, mew=1, zorder=5, alpha=1) #, yerr=[-edisp1, edisp2]

#plt.show()

# plt.savefig('/home/qyfei/Desktop/Results/Physical_values/clouds/Line_width_propriate_molecular_gas_surface_density_relation_with_relative_conversion_factor_new_withbeam.pdf', bbox_inches='tight', dpi=300)

#N = np.where(disp_CO>2.8*Sigma_H2**0.5)
# %%
'''
maskl = (disp_CO>2.8*Sigma_H2**0.5) & (disp_CO>=30)
masku = disp_CO<9.156
plt.figure(figsize=(8,8))
ax = plt.subplot(111)
im = ax.imshow(mom0_rms*maskl, cmap='jet', origin="lower", norm=LogNorm())
cp,kw = colorbar.make_axes(ax, anchor=(0.2, -6), shrink=0.1, aspect=4, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label(r'FLUX')
ax.set_xlim(pos_cen[0]-size, pos_cen[0]+size)
ax.set_ylim(pos_cen[1]-size, pos_cen[1]+size)
ax.contour(mom0_rms, mom0_level, colors=['k'], linewidths=[1])
ax.set_xlim(300, 500)
ax.set_ylim(300, 500)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("/home/qyfei/Desktop/Codes/CODES/Physical_values/clouds/clouds_over.pdf", bbox_inches='tight', dpi=300)

plt.figure(figsize=(8,8))
ax = plt.subplot(111)
im = ax.imshow(mom0_rms*masku, cmap='jet', origin="lower", norm=LogNorm())
cp,kw = colorbar.make_axes(ax, anchor=(0.2, -6), shrink=0.1, aspect=4, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label(r'FLUX')
ax.set_xlim(200, 600)
ax.set_ylim(200, 600)
ax.contour(mom0_rms, mom0_level, colors=['k'], linewidths=[1])
ax.set_xticks([])
ax.set_yticks([])

plt.savefig("/home/qyfei/Desktop/Codes/CODES/Physical_values/clouds/clouds_lowdisp.pdf", bbox_inches='tight', dpi=300)
'''
# %%
## Load clouds properties
'''outputdir = "/home/qyfei/Desktop/Codes/CODES/Physical_values/clouds/spectral/"
para_f = np.loadtxt(outputdir+"para_f.txt")
para_s = np.loadtxt(outputdir+"para_s.txt")
f, ef1, ef2 = para_f[:,0]/1e3, para_f[:,1]/1e3, para_f[:,2]/1e3
disp_CO_clump, edisp_CO_clump1, edisp_CO_clump2 = para_s[:,0], para_s[:,1], para_s[:,2]
Sigma_H2_clump = surface_density_mom0(f, hdu)
eSigma_H2_clump1 = Sigma_H2_clump - surface_density_mom0(f-ef1, hdu)
eSigma_H2_clump2 = surface_density_mom0(f+ef2, hdu) - Sigma_H2_clump
from Physical_values.clouds_properties import clouds
path = '/home/qyfei/Desktop/Codes/CODES/Physical_values/clouds/'
file = 'PG0050_props.fits'
ith = np.delete(np.arange(77), np.array([0, 3, 14, 20, 28, 45, 66]))
xctr, yctr, vctr, PA, inc, FWHMmaj, FWHMmin, momvpix = clouds(path, file, ith)
from Physical_values.surface_density import radius
radius_s = radius(xctr, yctr, np.array([395, 399]), 0.05, 35, 41)
Sigma_s_clump = Sigma_disk(Sigma, radius_s, re)
Sigma_tot_clump = Sigma_H2_clump + 2/(1+90**2/disp_CO_clump**2)*Sigma_s_clump

bins = np.logspace(1., 3.6, 10)
Sigma_tot_mean = []
disp_H2_mean = []
disp_H2_disp = []
for i in range(len(bins)-1):
    N = np.where((Sigma_tot<bins[i+1])&(Sigma_tot>=bins[i]))
    Sigma_tot_mean.append(np.nanmean(Sigma_tot[N]))
    disp_H2_mean.append(np.nanmean(disp_CO[N]))
    disp_H2_disp.append(np.nanstd(disp_CO[N]))'''
#ax.errorbar(Sigma_tot_clump[:28], disp_CO_clump[:28], xerr=[eSigma_H2_clump1[:28], eSigma_H2_clump2[:28]], yerr=[edisp_CO_clump1[:28], edisp_CO_clump2[:28]], fmt='ko', mfc='k', mec='k', mew=1, ms=1, capsize=5)
#ax.errorbar(Sigma_tot_clump[59:], disp_CO_clump[59:], xerr=[eSigma_H2_clump1[59:], eSigma_H2_clump2[59:]], yerr=[edisp_CO_clump1[59:], edisp_CO_clump2[59:]], fmt='ko', mfc='k', mec='k', mew=1, ms=1, capsize=5)

# %%
object = "PG1011"
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/"+object+"/data"
mom0file = "/"+object+".dilmsk.mom0.fits.gz"
# mom0 = fits.open(path+mom0file)[0].data
mom0 = fits.open(path+"/PG1011-040"+"_CO21_final_image_mom0.fits")[0].data[0][0]

mom2file = "/"+object+".dilmsk.mom2.fits.gz"
mom2 = fits.open(path+mom2file)[0].data

N = np.where(mom0)

plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
ax.scatter(mom0[N], mom2[N])
ax.loglog()
# %%
