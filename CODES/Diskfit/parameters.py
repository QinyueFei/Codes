# %%
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15
import astropy.units as u
import astropy.constants as c
# %%
path = "/home/qyfei/Desktop/Results/Diskfit/PG_quasars/PG0050/OUT"
file = "/bi.txt"
r, npts, Vt, eVt, Vr, eVr, Vm_t, eVm_t, Vm_r, eVm_r = np.genfromtxt(path+file,skip_header=1,unpack=True)

r_kpc = (r*0.05*u.arcsec/Planck15.arcsec_per_kpc_proper(0.061)).value

# %%
plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)


fig, axes = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
ax0 = axes
msize = 4

ax0.errorbar(r_kpc, -Vt, yerr=eVt, fmt='ro', mfc='r', ms=msize, mew=1, elinewidth=1, capsize=5, label="$V_\mathrm{rot}$", zorder=3)
ax0.fill_between(r_kpc, -Vt-eVt, -Vt+eVt, color='r', alpha=0.3, zorder=3)
ax0.set_xlabel('Radius [kpc]')
ax0.set_ylabel('$V_\mathrm{rot}$ [$\mathrm{km\,s^{-1}}$]')

ax0.errorbar(RAD_kpc, VROT, yerr=[-E_VROT1, E_VROT2], fmt='ko', mfc='k', ms=msize, mew=1, elinewidth=1, capsize=5)


# ax1 = ax0.twinx()
ax0.errorbar(r_kpc, Vm_t, yerr=eVm_t, fmt='go', mfc='g', ms=msize, mew=1, elinewidth=1, capsize=5, zorder=3, label="$V_{2,t}$")
ax0.fill_between(r_kpc, Vm_t-eVm_t, Vm_t+eVm_t, color='g', alpha=0.3, zorder=3)

ax0.errorbar(r_kpc, Vm_r, yerr=eVm_r, fmt='bo', mfc='b', ms=msize, mew=1, elinewidth=1, capsize=5, zorder=3, label="$V_{2,r}$")
ax0.fill_between(r_kpc, Vm_r-eVm_r, Vm_r+eVm_r, color='b', alpha=0.3, zorder=3)



for ax in [ax0]:
    ax.set_xlim(1e-4, r_kpc[-1]+(r_kpc[2]-r_kpc[1])/2)
    ax.set_ylim(-100, 435)
    ax.grid()
    ax.fill_between([0, 0.43], [-500], [500], color='k', hatch='/', alpha=0.1)
plt.legend()
# plt.savefig("/home/qyfei/Desktop/Results/Diskfit/PG_quasars/PG0050/vels.pdf", bbox_inches="tight", dpi=300)

# %%
