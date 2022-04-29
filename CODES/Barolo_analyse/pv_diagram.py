# %%
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization.wcsaxes.core import WCSAxesSubplot
from astropy.wcs import WCS
import matplotlib.colorbar as colorbar
from matplotlib.transforms import Affine2D
from astropy.visualization.wcsaxes import WCSAxes
import astropy.units as u
from matplotlib.colors import LogNorm

plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

# path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/"
# folder = "Barolo_fit/output/PG0050+124_best/pvs/"
# file = "PG0050+124"
# %%
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG1244"
folder = "/Barolo_fit/CO32/output/PG1244+026/pvs/"
file = "PG1244+026"


def load_PV(path, folder, file):
    pv_obs, pv_mod = [], []
    for i in ["a", "b"]:
        pv_obs.append(fits.open(path+folder+file+"_pv_"+i+".fits")[0].data)
        pv_mod.append(fits.open(path+folder+file+"mod_pv_"+i+"_local.fits")[0].data)
    return pv_obs, pv_mod

pv_maj_obs, pv_min_obs = load_PV(path, folder, file)[0]
pv_maj_mod, pv_min_mod = load_PV(path, folder, file)[1]

# %%
from astropy.stats import sigma_clipped_stats
sigma_PV = sigma_clipped_stats(pv_maj_obs, sigma = 3)[-1]
# PV_level = np.array([-2,2,np.sqrt(8),4, np.sqrt(32),8,np.sqrt(128),16,np.sqrt(512),32,np.sqrt(2048),64])*sigma_PV
PV_level = np.array([-2,2,4,8,16,32,64])*sigma_PV
sigma_PV

pix_size = 0.2

fig = plt.figure(figsize=(16, 16))

transform = Affine2D()
transform.scale(pix_size, 10.2)
# transform.translate(-88*pix_size, -116*10.78065)
transform.translate(-86*pix_size, -52*10.2)

transform.rotate(0.)  # radians
coord_meta = {}
coord_meta['name'] = 'Offset [arcsec]', '$\Delta$V [km/s]'
coord_meta['type'] = 'longitude', 'scalar'
coord_meta['wrap'] = 180, None
coord_meta['unit'] = u.arcsec, u.Unit("")
coord_meta['format_unit'] = None, None

ax1 = WCSAxes(fig, [0.1, 0.0, 0.9, 0.5], aspect='auto', transform=transform, coord_meta=coord_meta)
ax1.imshow(pv_min_obs, cmap='Greys', origin='lower', aspect='auto')
ax1.contour(pv_min_obs, PV_level, linewidths=1, colors='#00008B')
ax1.contour(pv_min_mod, PV_level, linewidths=1, colors='#B22222')
ax1.hlines(52, 0, 1000, 'k', lw=1)
ax1.vlines(86, 0, 1000, 'k', lw=1)
ax1.set_xlim(86-50, 86+50)
ax1.set_ylim(52-41, 52+41)
ax1.text(43, 15, "$\phi=334\deg$", fontsize=40)

transform = Affine2D()
transform.scale(pix_size, 10.2)
transform.translate(-86*pix_size, -52*10.2)
transform.rotate(0.)  # radians

coord_meta = {}
coord_meta['name'] = 'Offset [arcsec]', '$\Delta$V [km/s]'
coord_meta['type'] = 'longitude', 'scalar'
coord_meta['wrap'] = 180, None
coord_meta['unit'] = u.arcsec, u.Unit("")
coord_meta['format_unit'] = None, None

ax0 = WCSAxes(fig, [0.1, 0.5, 0.9, 0.5], aspect='auto', transform=transform, coord_meta=coord_meta)
plt.subplots_adjust(wspace=0)

ax0.imshow(pv_maj_obs, cmap='Greys', origin='lower', aspect='auto')
ax0.contour(pv_maj_obs, PV_level, linewidths=1, colors='#00008B', label='Data')
ax0.contour(pv_maj_mod, PV_level, linewidths=1, colors='#B22222', label='Model')

ax0.hlines(52, 0, 1000, 'k', lw=1)
ax0.vlines(86, 0, 1000, 'k', lw=1)
ax0.set_xlim(86-50, 86+50)
ax0.set_ylim(52-41, 52+41)
ax0.text(43, 15, '$\phi=244\deg$', fontsize=40)
ax0.plot([0,0],[0,0], c='#00008B', label='DATA')
ax0.plot([0,0],[0,0], c='#B22222', label='MODEL')

ax0.legend(loc='upper right', fontsize=40)#bbox_to_anchor=(0.15, -0.95)
fig.add_axes(ax0)
fig.add_axes(ax1)

#plt.show()

# plt.savefig('/home/qyfei/Desktop/Results/Barolo/PG_quasars/PG2130/fit_PV.pdf', bbox_inches='tight', dpi=300)
# 
# %%
