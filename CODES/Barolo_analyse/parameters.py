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
path = '/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/'
folder = "Barolo_fit/output/PG0050+124_best/"
file = "ringlog2.txt"
parameters = load_parameters(path, folder, file)
r_fit, rad_fit, vrot_fit, evrot1_fit, evrot2_fit, vdisp_fit, edisp1_fit, edisp2_fit, inc_fit, pa_fit, vrad_fit, vsys_fit, vrad, xpos,ypos, x, y, y_perp, vcirc_fit, einc1_fit, einc2_fit, epa1_fit, epa2_fit = parameters


## Plot the inclination and position angle
fig, axes = plt.subplots(figsize=(8,16), nrows=2, ncols=1)
plt.subplots_adjust(hspace=0)

ax2, ax0 = axes
ax0.errorbar(r_fit, pa_fit, yerr=[-epa1_fit, epa2_fit], fmt='mo', mfc='none', ms=10, mew=1, elinewidth=1, capsize=0.)
ax0.set_ylabel('$\phi$ [deg]')
ax0.set_xlim(-0.1, 3.2)
ax0.set_ylim(80, 150)
ax0.set_xlabel('Radius [kpc]')
ax1 = ax0.twinx()
ax1.errorbar(r_fit, inc_fit, yerr=[-einc1_fit, einc2_fit], fmt='gs', mfc='none', ms=10, mew=1, elinewidth=1, capsize=0)
ax1.set_ylabel('$i$ [deg]')
ax1.set_ylim(30, 80)

ax1.errorbar(0,0,0, fmt='mo', mfc='none', ms=10, mew=1, elinewidth=1, capsize=0, label='$\phi$')
ax1.errorbar(0,0,0, fmt='gs', mfc='none',ms=10, mew=1, elinewidth=1, capsize=0, label='$i$')
ax1.legend(loc='upper right')

## plot velocity profile
#ax2.errorbar(r_fit, vcirc_fit, yerr=[-evrot1_fit, evrot2_fit], fmt='bo', mfc='none', ms=10, mew=1, elinewidth=1, capsize=0)
ax2.errorbar(r_fit, vrot_fit, yerr=[-evrot1_fit, evrot2_fit], fmt='bo', mfc='none', ms=10, mew=1, elinewidth=1, capsize=0)
ax2.set_xlabel('radius [kpc]')
ax2.set_ylabel('$v_\mathrm{rot}$ [$\mathrm{km\,s^{-1}}$]')
ax2.set_xlim(-0.1, 3.2)
ax2.set_ylim(5, 420)

ax3 = ax2.twinx()
ax3.errorbar(r_fit, vdisp_fit, yerr=[-edisp1_fit, edisp2_fit], fmt='ro', mfc='none', ms=10, mew=1, elinewidth=1, capsize=0)
ax3.set_ylabel('$\sigma$ [$\mathrm{km\,s^{-1}}$]')
ax3.set_ylim(5, 110)

ax3.errorbar(0,0,0, fmt='bo', mfc='none', ms=10, mew=1, capsize=0, elinewidth=1, label='$v_\mathrm{rot}$')
#ax3.errorbar(0,0,0, fmt='bo', mfc='none', ms=10, mew=1, capsize=0, elinewidth=1, label='$v_\mathrm{circ}$')
ax3.errorbar(0,0,0, fmt='ro', mfc='none',ms=10, mew=1, capsize=0, elinewidth=1, label='$\sigma$')
ax3.legend(loc='upper right')
for ax in axes:
    ax.fill_between([0,0.44], [0], [500], color='b', alpha=0.2)
    ax.vlines(0.8, 0, 500, color='b', lw=1, ls='--')
    ax.vlines(2.1, 0, 500, color='b', lw=1, ls='--')
#plt.show()
#plt.savefig('/home/qyfei/Desktop/Results/Barolo/PG0050/profiles.pdf', bbox_inches='tight', dpi=300)

# %%
