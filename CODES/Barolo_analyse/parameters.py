# %%
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

def load_parameters(path, folder, file):
    ## Parameters of fixing
    fit_para = np.loadtxt(path+folder+file, dtype='float')
    ## Load the fitting result
    r_fit = fit_para[:,0]           ## radius in kpc
    rad_fit = fit_para[:,1]         ## radius in arcsec
    vrot_fit = fit_para[:,2]        ## rotation velocity 
    evrot1_fit = fit_para[:,13]     ## error of rotation velocity
    evrot2_fit = fit_para[:,14]
    vdisp_fit = fit_para[:,3]       ## velocity dispersion
    edisp1_fit = fit_para[:,15]     ## error of velocity dispersion
    edisp2_fit = fit_para[:,16]
    inc_fit = fit_para[:,4]         ## inclination 
    pa_fit = fit_para[:,5]          ## position angle
    vrad_fit = fit_para[:,12]       ## radial velocity
    vsys_fit = fit_para[:,11]       ## systematic velocity
    xpos = fit_para[0,9]            ## x-axis position coordinates
    ypos = fit_para[0,10]           ## y-axis position coordinates
    vcirc_fit = np.sqrt(vrot_fit**2+vdisp_fit**2) ## circular velocity after asymmetric drift
    
    x = np.linspace(50,1000) 
    k = np.tan((90+np.mean(pa_fit))*u.deg)
    y = k*(x-xpos)+ypos
    y_perp =  np.tan(np.mean(pa_fit)*u.deg)*(x-xpos)+ypos
    
    fit_para_1st = np.loadtxt(path+folder+'ringlog1.txt')
    einc1_fit = fit_para_1st[:,17]  ## error of inclination
    einc2_fit = fit_para_1st[:,18]
    epa1_fit = fit_para_1st[:,19]   ## error of position angle
    epa2_fit = fit_para_1st[:,20]
    ## Output the parameters
    return np.array([r_fit, rad_fit, vrot_fit, evrot1_fit, evrot2_fit, vdisp_fit, edisp1_fit, edisp2_fit, inc_fit, pa_fit, vrad_fit, vsys_fit, vrad_fit, xpos,ypos, x, y, y_perp, vcirc_fit, einc1_fit, einc2_fit, epa1_fit, epa2_fit])

# %%
# path = '/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/'
# folder = "Barolo_fit/output/PG0050+124_best/"
object = "PG1244"
name = "PG1244+026"
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/"+object
folder = "/Barolo_fit/CO32/output/"+name
file = "/ringlog1.txt"

RAD_kpc, RAD_arcs, VROT, DISP, INC, PA, Z0, Z0_arcs, SIG, XPOS, YPOS, VSYS, VRAD, E_VROT1, E_VROT2, E_DISP1, E_DISP2, E_INC1, E_INC2, E_PA1, E_PA2, E_XPOS1, E_XPOS2, E_YPOS1, E_YPOS2, E_VSYS1, E_VSYS2  = np.genfromtxt(path+folder+file,skip_header=1,unpack=True)
#
# %%
# parameters = load_parameters(path, folder, file)
# r_fit, rad_fit, vrot_fit, evrot1_fit, evrot2_fit, vdisp_fit, edisp1_fit, edisp2_fit, inc_fit, pa_fit, vrad_fit, vsys_fit, vrad, xpos,ypos, x, y, y_perp, vcirc_fit, einc1_fit, einc2_fit, epa1_fit, epa2_fit = parameters
# folder = "Barolo_fit/output/PG0050+124_trash/"
# file = "ringlog2.txt"
# r_fit = np.loadtxt(path+folder+file)[:,0]
fig, axes = plt.subplots(figsize=(18,6), nrows=1, ncols=3)
ax0, ax1, ax2 = axes
VSYS_weight = np.sqrt(1/E_VSYS1**2 + 1/E_VSYS2**2)
VSYS_mean = np.nansum(VSYS_weight*VSYS)/np.nansum(VSYS_weight)
ax0.errorbar(RAD_kpc, VSYS, yerr=[-E_VSYS1, E_VSYS2], fmt="ko", mfc='k')
ax0.hlines(VSYS_mean, 0, 5, color='k', ls='--')
ax0.set_xlim(0, 5)

XPOS_weight = np.sqrt(1/E_XPOS1**2 + 1/E_XPOS2**2)
XPOS_mean = np.nansum(XPOS_weight*XPOS)/np.nansum(XPOS_weight)
ax1.errorbar(RAD_kpc, XPOS, yerr=[-E_XPOS1, E_XPOS2], fmt="ko", mfc='k')
ax1.hlines(XPOS_mean, 0, 5, color='k', ls='--')
ax1.set_xlim(0, 5)

YPOS_weight = np.sqrt(1/E_YPOS1**2 + 1/E_YPOS2**2)
YPOS_mean = np.nansum(YPOS_weight*YPOS)/np.nansum(YPOS_weight)
ax2.errorbar(RAD_kpc, YPOS, yerr=[-E_YPOS1, E_YPOS2], fmt="ko", mfc='k')
ax2.hlines(YPOS_mean, 0, 5, color='k', ls='--')
ax2.set_xlim(0, 5)

print(VSYS_mean, XPOS_mean, YPOS_mean)
# %%
## load the best fit parameters
# object = "PG0923"
# name = "PG0923+129"

# path = '/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/'
# folder = "Barolo_fit/output/PG0050+124_best/"

path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/"+object
folder = "/Barolo_fit/CO32/output/"+name

RAD_kpc, RAD_arcs, VROT, DISP, INC, PA, Z0, Z0_arcs, SIG, XPOS, YPOS, VSYS, VRAD, E_VROT1, E_VROT2, E_DISP1, E_DISP2, E_INC1, E_INC2, E_PA1, E_PA2  = np.genfromtxt(path+folder+"/ringlog1.txt",skip_header=1,unpack=True)

RAD_kpc, RAD_arcs, VROT, DISP, INC, PA, Z0, Z0_arcs, SIG, XPOS, YPOS, VSYS, VRAD, E_VROT1, E_VROT2, E_DISP1, E_DISP2 = np.genfromtxt(path+folder+"/ringlog2.txt",skip_header=1,unpack=True)

# path = '/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/'
# folder = "Barolo_fit/output/PG0050+124_trash/"
# RAD_kpc = np.loadtxt(path+folder+"/ringlog2.txt")[:,0]

## Plot the inclination and position angle
fig, axes = plt.subplots(figsize=(8,16), nrows=2, ncols=1)
plt.subplots_adjust(hspace=0)

msize = 4

ax2, ax0 = axes
ax0.errorbar(RAD_kpc, PA, yerr=[-E_PA1, E_PA2], fmt='mo', mfc='m', ms=msize, mew=1, elinewidth=1, capsize=5.)
ax0.fill_between(RAD_kpc, PA+E_PA1, PA+E_PA2, color = 'm', alpha=0.3)
ax0.set_ylabel('$\phi$ [deg]')
ax0.set_xlim(1e-4, RAD_kpc[-1]+(RAD_kpc[2]-RAD_kpc[1])/2)
ax0.set_ylim(np.min(PA)-15, np.max(PA)+25)
ax0.set_xlabel('Radius [kpc]')
ax1 = ax0.twinx()
ax1.errorbar(RAD_kpc, INC, yerr=[-E_INC1, E_INC2], fmt='gs', mfc='g', ms=msize, mew=1, elinewidth=1, capsize=5)
ax1.fill_between(RAD_kpc, INC+E_INC1, INC+E_INC2, color='g', alpha=0.3)
ax1.set_ylabel('$i$ [deg]')
ax1.set_ylim(np.min(INC)-25, np.max(INC)+15)

ax1.errorbar(-500,0,0, fmt='mo', mfc='m', ms=msize, mew=1, elinewidth=1, capsize=5, label='$\phi$')
ax1.errorbar(-500,0,0, fmt='gs', mfc='g',ms=msize, mew=1, elinewidth=1, capsize=5, label='$i$')
ax1.legend(loc='upper left')

## plot velocity profile
#ax2.errorbar(r_fit, vcirc_fit, yerr=[-evrot1_fit, evrot2_fit], fmt='bo', mfc='none', ms=10, mew=1, elinewidth=1, capsize=0)
ax2.errorbar(RAD_kpc, VROT, yerr=[-E_VROT1, E_VROT2], fmt='bo', mfc='b', ms=msize, mew=1, elinewidth=1, capsize=5)
ax2.fill_between(RAD_kpc, VROT+E_VROT1, VROT+E_VROT2, color='b', alpha=0.3)
ax2.set_xlabel('radius [kpc]')
ax2.set_ylabel('$V_\mathrm{rot}$ [$\mathrm{km\,s^{-1}}$]')
ax2.set_xlim(1e-4, RAD_kpc[-1]+(RAD_kpc[2]-RAD_kpc[1])/2)
ax2.set_ylim(-10, 365)

ax3 = ax2.twinx()
ax3.errorbar(RAD_kpc, DISP, yerr=[-E_DISP1, E_DISP2], fmt='ro', mfc='r', ms=msize, mew=1, elinewidth=1, capsize=5)
ax3.fill_between(RAD_kpc, DISP+E_DISP1, DISP+E_DISP2, color='r', alpha=0.3)
ax3.set_ylabel('$\sigma$ [$\mathrm{km\,s^{-1}}$]')
ax3.set_ylim(-10, 365)

ax3.errorbar(-500,0,0, fmt='bo', mfc='b', ms=msize, mew=1, capsize=5, elinewidth=1, label='$v_\mathrm{rot}$')
#ax3.errorbar(0,0,0, fmt='bo', mfc='none', ms=10, mew=1, capsize=0, elinewidth=1, label='$v_\mathrm{circ}$')
ax3.errorbar(-500,0,0, fmt='ro', mfc='r',ms=msize, mew=1, capsize=5, elinewidth=1, label='$\sigma$')
ax3.legend(loc='upper left')
for ax in axes:
    # ax.fill_between([0,0.43], [0], [500], color='k', hatch='/', alpha=0.2)
    # ax.vlines(0.43, 0, 500, color='k', lw=1, ls='--')
    ax.vlines(0.86, -500, 500, color='k', lw=1, ls='--')
    ax.vlines(2.10, -500, 500, color='k', lw=1, ls='--')
    ax.fill_between([0, 0.43], [-500], [500], color='k', hatch='/', alpha=0.2)

#plt.show()
# plt.savefig('/home/qyfei/Desktop/Results/Barolo/PG_quasars/PG0050/profiles.pdf', bbox_inches='tight', dpi=300)

# %%
## Velocity profiles
fig, axes = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
ax0 = axes

ax0.errorbar(RAD_kpc, VROT, yerr=[-E_VROT1, E_VROT2], fmt='bo', mfc='b', ms=msize, mew=1, elinewidth=1, capsize=5, label="$V_\mathrm{rot}$", zorder=3)
ax0.fill_between(RAD_kpc, VROT+E_VROT1, VROT+E_VROT2, color='b', alpha=0.3, zorder=3)

ax0.errorbar(RAD_kpc1, VROT1, yerr=[-E_VROT11, E_VROT21], fmt='bo', mfc='none', ms=msize, mew=1, elinewidth=1, capsize=5, label="$V_\mathrm{rot}$", zorder=3)
ax0.fill_between(RAD_kpc1, VROT1+E_VROT11, VROT1+E_VROT21, color='b', alpha=0.3, zorder=3)


ax0.set_xlabel('Radius [kpc]')
ax0.set_ylabel('$V_\mathrm{rot}$ [$\mathrm{km\,s^{-1}}$]')

ax1 = ax0.twinx()
ax1.errorbar(RAD_kpc, DISP, yerr=[-E_DISP1, E_DISP2], fmt='ro', mfc='r', ms=msize, mew=1, elinewidth=1, capsize=5, zorder=3)
ax1.fill_between(RAD_kpc, DISP+E_DISP1, DISP+E_DISP2, color='r', alpha=0.3, zorder=3)

ax1.errorbar(RAD_kpc1, DISP1, yerr=[-E_DISP11, E_DISP21], fmt='ro', mfc='r', ms=msize, mew=1, elinewidth=1, capsize=5, zorder=3)
ax1.fill_between(RAD_kpc1, DISP1+E_DISP11, DISP1+E_DISP21, color='r', alpha=0.3, zorder=3)


ax1.set_ylabel("$\sigma$ [$\mathrm{km\,s^{-1}}$]")

ax1.errorbar(-100,0,0, fmt='bo', mfc='b', ms=msize, mew=1, capsize=5, elinewidth=1, label='$V_\mathrm{rot}$')
ax1.errorbar(-100,0,0, fmt='ro', mfc='r',ms=msize, mew=1, capsize=5, elinewidth=1, label='$\sigma$')
ax1.legend(loc='upper left')

for ax in [ax0, ax1]:
    # ax.set_xlim(1e-4, RAD_kpc[-1]+(RAD_kpc[2]-RAD_kpc[1])/2)
    # ax.set_ylim(-5, 335)
    ax.set_xlim(1e-2, 3)
    ax.set_ylim(0, 150)
    ax.semilogx()
    ax.grid()
    ax.fill_between([0, 2*(RAD_kpc[1]-RAD_kpc[0])], [-500], [500], color='k', hatch='/', alpha=0.1)

x = np.logspace(-3,3,1000)
y = np.sqrt(G*10**6.62/x)

ax.plot(x, y, 'k-')


# plt.savefig("/home/qyfei/Desktop/Results/Barolo/PG_quasars/"+object+"/velocity_profile_tot.pdf", bbox_inches="tight", dpi=300)

# %%
## Angle profiles
fig, axes = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
ax0 = axes

ax0.errorbar(RAD_kpc, PA, yerr=[-E_PA1, E_PA2], fmt='mo', mfc='m', ms=msize, mew=1, elinewidth=1, capsize=5.)
ax0.fill_between(RAD_kpc, PA+E_PA1, PA+E_PA2, color = 'm', alpha=0.3)

ax0.errorbar(RAD_kpc1, PA1, yerr=[-E_PA11, E_PA21], fmt='mo', mfc='m', ms=msize, mew=1, elinewidth=1, capsize=5.)
ax0.fill_between(RAD_kpc1, PA1+E_PA11, PA1+E_PA21, color = 'm', alpha=0.3)


ax0.set_ylabel('$\phi$ [deg]')
ax0.set_xlabel('Radius [kpc]')
ax0.set_ylim(np.min(PA)-15, np.max(PA)+25)

ax1 = ax0.twinx()
ax1.errorbar(RAD_kpc, INC, yerr=[-E_INC1, E_INC2], fmt='gs', mfc='g', ms=msize, mew=1, elinewidth=1, capsize=5)
ax1.fill_between(RAD_kpc, INC+E_INC1, INC+E_INC2, color='g', alpha=0.3)

ax1.errorbar(RAD_kpc1, INC1, yerr=[-E_INC11, E_INC21], fmt='gs', mfc='g', ms=msize, mew=1, elinewidth=1, capsize=5)
ax1.fill_between(RAD_kpc1, INC1+E_INC11, INC1+E_INC21, color='g', alpha=0.3)


ax1.set_ylabel('$i$ [deg]')
ax1.set_ylim(np.min(INC)-25, np.max(INC)+15)

ax1.errorbar(-100,0,0, fmt='mo', mfc='m', ms=msize, mew=1, elinewidth=1, capsize=5, label='$\phi$')
ax1.errorbar(-100,0,0, fmt='gs', mfc='g',ms=msize, mew=1, elinewidth=1, capsize=5, label='$i$')
ax1.legend(loc='upper left')

for ax in [ax0, ax1]:
    # ax.set_xlim(1e-4, RAD_kpc[-1]+(RAD_kpc[2]-RAD_kpc[1])/2)
    ax.set_xlim(1e-2, 3)
    
    # ax.grid()
    ax.semilogx()
    ax.fill_between([0,2*(RAD_kpc[1]-RAD_kpc[0])], [-500], [500], color='k', hatch='/', alpha=0.1)
ax0.set_ylim(0, 90)
# plt.savefig("/home/qyfei/Desktop/Results/Barolo/PG_quasars/"+object+"/angle_profile_tot.pdf", bbox_inches="tight", dpi=300)

# %%
## estimate the error of angles
from astropy.io import fits
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/"+object
folder = "/Barolo_fit/output/"+name+"_angles/"
f = fits.open(path+folder+"spacepar_"+name+".fits")[0]
d, h = f.data, f.header
p1, p2 = h['CTYPE1'], h['CTYPE2']
p1u, p2u, ru = h['CUNIT1'], h['CUNIT2'], h['CUNIT3']
p1ran = (np.arange(0,d.shape[2])+1-h['CRPIX1'])*h['CDELT1']+h['CRVAL1']
p2ran = (np.arange(0,d.shape[1])+1-h['CRPIX2'])*h['CDELT2']+h['CRVAL2']
rings = (np.arange(0,d.shape[0])+1-h['CRPIX3'])*h['CDELT3']+h['CRVAL3']

# %%
plt.figure(figsize=(8, 10))
# ax = plt.subplot(111)
plt.imshow(d[0], origin='lower', cmap='jet')
plt.contour(d[0], levels=np.percentile(d[1], [16, 50, 84]), colors="k")

# %%
N = np.where(d[1] <= np.percentile(d[1], [16]))
plt.hist(d[1][N])
chi2 = np.reshape(d, [1, len(d)*len(d[0])*len(d[0][0])])

plt.hist(np.log10(chi2[0]), bins=30)
plt.vlines(np.nanpercentile(np.log10(chi2[0]), [16, 50, 84]), ymin=0, ymax=1e4, color='k', ls='--')
plt.ylim(0, 1200)

chi2_lo = np.nanpercentile(chi2[0], [16])

# %%
PA_mean, INC_mean, PA_std, INC_std = np.zeros(len(d)), np.zeros(len(d)), np.zeros(len(d)), np.zeros(len(d))
for i in range(len(d)):
    ring_data = d[i]
    N = np.where(ring_data <= np.nanpercentile(ring_data, [16]))
    PA_mean[i] = np.mean(p1ran[N[1]])
    INC_mean[i] = np.mean(p2ran[N[0]])

    PA_std[i] = np.nanstd(p1ran[N[1]])
    INC_std[i] = np.nanstd(p2ran[N[0]])

# %%
## Angle profiles
fig, axes = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
ax0 = axes

ax0.errorbar(RAD_kpc, PA, yerr=PA_std, fmt='mo', mfc='m', ms=msize, mew=1, elinewidth=1, capsize=5.)
ax0.fill_between(RAD_kpc, PA-PA_std, PA+PA_std, color = 'm', alpha=0.3)
ax0.set_ylabel('$\phi$ [deg]')
ax0.set_xlabel('Radius [kpc]')
ax0.set_ylim(np.min(PA)-15, np.max(PA)+35)

ax1 = ax0.twinx()
ax1.errorbar(RAD_kpc, INC, yerr=INC_std, fmt='gs', mfc='g', ms=msize, mew=1, elinewidth=1, capsize=5)
ax1.fill_between(RAD_kpc, INC-INC_std, INC+INC_std, color='g', alpha=0.3)
ax1.set_ylabel('$i$ [deg]')
ax1.set_ylim(np.min(INC)-35, np.max(INC)+15)

ax1.errorbar(-100,0,0, fmt='mo', mfc='m', ms=msize, mew=1, elinewidth=1, capsize=5, label='$\phi$')
ax1.errorbar(-100,0,0, fmt='gs', mfc='g',ms=msize, mew=1, elinewidth=1, capsize=5, label='$i$')
ax1.legend(loc='upper left')

for ax in [ax0, ax1]:
    ax.set_xlim(1e-4, RAD_kpc[-1]+(RAD_kpc[2]-RAD_kpc[1])/2)
    # ax.grid()
    ax.fill_between([0, 2*(RAD_kpc[1]-RAD_kpc[0])], [-500], [500], color='k', hatch='/', alpha=0.1)

# plt.savefig("/home/qyfei/Desktop/Results/Barolo/PG_quasars/"+object+"/angle_profile.pdf", bbox_inches="tight", dpi=300)

# %%
x_test = np.random.randn(1, 100)
plt.hist(x_test, bins=100)
# %%
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/"+object
folder = "/Barolo_fit/output/"+name

RAD_kpc1, RAD_arcs1, VROT1, DISP1, INC1, PA1, Z01, Z0_arcs1, SIG1, XPOS1, YPOS1, VSYS1, VRAD1, E_VROT11, E_VROT21, E_DISP11, E_DISP21, E_INC11, E_INC21, E_PA11, E_PA21  = np.genfromtxt(path+folder+"/ringlog1.txt",skip_header=1,unpack=True)

RAD_kpc1, RAD_arcs1, VROT1, DISP1, INC1, PA1, Z01, Z0_arcs1, SIG1, XPOS1, YPOS1, VSYS1, VRAD1, E_VROT11, E_VROT21, E_DISP11, E_DISP21 = np.genfromtxt(path+folder+"/ringlog2.txt",skip_header=1,unpack=True)
