# %%

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colorbar as colorbar
from matplotlib.transforms import Affine2D
from astropy.visualization.wcsaxes import WCSAxes
from spectral_cube import SpectralCube
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from matplotlib.colors import LogNorm

plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

def beam(HDU, XPOS, YPOS, col, cellsize):
    hdu = HDU
    xpos, ypos = XPOS, YPOS
    c = col
    cell = cellsize
    bmaj, bmin, bpa = hdu.header['BMAJ'], hdu.header['BMIN'], hdu.header['BPA']
    Bmaj = bmaj*u.Unit('deg').to('arcsec')/cell
    Bmin = bmin*u.Unit('deg').to('arcsec')/cell
    Beam = matplotlib.patches.Ellipse((xpos,ypos), Bmaj, Bmin, 90+bpa, 
                                      edgecolor='k', facecolor='gray', fill=True, zorder=3)    
    return Beam, Bmaj, Bmin
# %%

###############
## Load data ##
###############
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/"
barolo = "Barolo_fit/output/PG0050+124_fixed_novrad/maps/"
i = 0
obs_mom0 = fits.open(path+barolo+'PG0050+124_0mom.fits')[0].data
mod_mom0 = fits.open(path+barolo+'PG0050+124_local_0mom.fits')[0].data
#model_mom0 = fits.open('moment/PG0050_CO21.combine.all.line.10km.sub.model.mom0.fits')[0].data[0][0]
res_mom0 = obs_mom0 - mod_mom0

i = 1

obs_mom1 = fits.open(path+barolo+'PG0050+124_1mom.fits')[0].data
mod_mom1 = fits.open(path+barolo+'PG0050+124_local_1mom.fits')[0].data
#model_mom1 = fits.open('moment/PG0050_CO21.combine.all.line.10km.sub.model.mom1.fits')[0].data[0][0]
res_mom1 = obs_mom1 - mod_mom1

i = 2
obs_mom2 = fits.open(path+barolo+'PG0050+124_2mom.fits')[0].data
mod_mom2 = fits.open(path+barolo+'PG0050+124_local_2mom.fits')[0].data
#model_mom2 = fits.open('moment/PG0050_CO21.combine.all.line.10km.sub.model.mom2.fits')[0].data[0][0]
res_mom2 = obs_mom2 - mod_mom2

# %%
## Load fitted parameters

folder = "Barolo_fit/output/PG0050+124_fixed_novrad/"
fit_para = np.loadtxt(path+folder+'ringlog1.txt')
dens_para = np.loadtxt(path+folder+'densprof.txt')

r_fit = fit_para[:,0]
rad_fit = fit_para[:,1]
vrot_fit = fit_para[:,2]
vdisp_fit = fit_para[:,3]
inc_fit = fit_para[0,4]
pa_fit = fit_para[0,5]
vrad_fit = fit_para[:,12]
vsys_fit = fit_para[:,11]
evrot1 = fit_para[:,13]
evrot2 = fit_para[:,14]
evdisp1 = fit_para[:,15]
evdisp2 = fit_para[:,16]
#evrad1 = fit_para[:,17]
#evrad2 = fit_para[:,18]

vrad = fit_para[:,12]
xpos = fit_para[0,9]
ypos = fit_para[0,10]

I_fit = dens_para[:,9]#*u.Unit('Jy km/s/arcsec^2')
eI_fit = dens_para[:,8]/I_fit[0]

vcirc_fit = np.sqrt(vrot_fit**2+vdisp_fit**2)

# %%
obs = 'Barolo_fit/'
## Plot the fitting result

x = np.linspace(50,450)
k = np.tan((90+np.mean(pa_fit))*u.deg)
y = k*(x-xpos)+ypos
y_perp =  np.tan((np.mean(pa_fit))*u.deg)*(x-xpos)+ypos

size=70
pix_size=0.05
import matplotlib
import matplotlib.patches as patches
hdu = fits.open(path+obs+'PG0050_CO21.combine.all.line.10km.sub.4sigma.mom0.fits')[0]
mom0_rms = hdu.data[0][0]
hdu = fits.open(path+"working/PG0050_CO21.combine.all.line.10km.mom0.fits")[0]
mom0_tot = hdu.data[0][0]
sigma = sigma_clipped_stats(mom0_tot)[-1]
mom0_level = np.array([-1,1,2,4,8,16,32,64])*3*sigma


level = np.linspace(-50,50,11)

transform = Affine2D()
transform.scale(0.05, 0.05)
transform.translate(-xpos*0.05, -ypos*0.05)
transform.rotate(0.)  # radians

# Set up metadata dictionary
coord_meta = {}
coord_meta['name'] = '', ''
coord_meta['type'] = 'longitude', 'latitude'
coord_meta['wrap'] = 180, None
coord_meta['unit'] = u.arcsec, u.arcsec
coord_meta['format_unit'] = None, None

fig = plt.figure(figsize=(6,8))

ax = WCSAxes(fig, [0.1,0.1,0.8,0.8], aspect='equal',
             transform=transform, coord_meta=coord_meta)
fig.add_axes(ax)
#ax = plt.subplot(111)
im = ax.imshow(res_mom1, vmin=-70, vmax=70, cmap='jet', origin='lower')
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label(r'$V_\mathrm{los}$ Residual [km/s]')
ax.contour(mom0_rms, mom0_level/3, colors=['k'])

ax.scatter(xpos, ypos, c='w', edgecolor='k', zorder=3, linewidth=0.9, s=100, marker='*')
ax.plot(x, y,'--',linewidth=1.5, c='k', alpha=0.8)
ax.plot(x, y_perp,':',linewidth=1.5, c='k', alpha=0.8)

rec = matplotlib.patches.Rectangle((xpos-size, ypos-size), 10, 11,
                                   angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax.add_artist(rec)
Beam = beam(hdu, xpos-size+5,ypos-size+5, 'w', pix_size)
ax.add_artist(Beam[0])

'''for i in range(len(rad_fit)):
    ell = patches.Ellipse((pos_cen[0], pos_cen[1]), (2*rad_fit[i]+0.15/2)/0.05, (2*rad_fit[i]+0.15/2)*np.cos(inc_fit[i]*u.deg)/0.05
                          , 90+pa_fit[i], edgecolor='k',facecolor='k', fill=False, zorder=3)
    ax.add_artist(ell)'''


ax.set_xlim(xpos-size,xpos+size)
ax.set_ylim(ypos-size,ypos+size)

print(level)

plt.savefig('/home/qyfei/Desktop/Codes/Result/rms_mom0_overplot_res_mom1_novrad.pdf', bbox_inches='tight', dpi=300)

# %%
