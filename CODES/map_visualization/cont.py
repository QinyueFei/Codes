# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy.ma import count
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colorbar as colorbar
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from map_visualization.maps import beam

#%matplotlib inline
plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=20)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

def load_cont(path, file):
    # Using procedure
    # cont, wcs, size, pix_size, hdu, pos_cen = load_cont(path, file)
    hdu = fits.open(path+file)[0]
    cont = hdu.data[0][0]*1e3
    pos_cen = np.where(cont == np.nanmax(cont))
    size = 200
    pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    #hdu = fits.open(path+file_pbc)[0]
    #cont_pbc = hdu.data[0][0]*1e3
    wcs = WCS(hdu.header)
    return cont, wcs, size, pix_size, hdu, pos_cen

#############################
## plot the momentum 0 map ##
#############################
def plot_cont(path, file):
    cont, wcs, size, pix_size, hdu, pos_cen = load_cont(path, file)
    size = 50
    rms = 1.3e-02#sigma_clipped_stats(cont, sigma=3)[-1]
    cont_level = np.array([-1,1,2,4,8,16,32])*3*rms
    fig = plt.figure(figsize=(8,10))
    ax = plt.subplot(projection=wcs[0,0])
    im = ax.imshow(cont, cmap='jet', vmin=-rms, vmax=np.nanmax(cont), origin='lower', norm=colors.PowerNorm(0.5))
    cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
    cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
    cb.set_label(r'cont [mJy$\cdot$beam$^{-1}$]')
    ax.contour(cont, cont_level, colors=['k'], linewidths=1.)

    rec = matplotlib.patches.Rectangle((pos_cen[0]-size, pos_cen[1]-size), 10, 10, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2.5)
    ax.add_artist(rec)
    Beam = beam(hdu, pos_cen[0]-size+5,pos_cen[1]-size+5, 'gray', pix_size)
    ax.add_artist(Beam[0])

    ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
    ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)
    ax.set_xlabel("R.A. (J2000)", labelpad=0.5, fontsize=20)
    ax.set_ylabel("Dec (J2000)", labelpad=-1.0, fontsize=20)

    # plt.show()
    # plt.savefig('/home/qyfei/Desktop/Results/map_visualization/Image/PG0050/PG0050p0124_cont_show.pdf', bbox_inches='tight', dpi=300)

path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/cont/"
file = "PG0050-cont-mosaic-pbc.fits"
cont, wcs, size, pix_size, hdu, pos_cen = load_cont(path, file)
plot_cont(path, file)

# %%
## Estimate the continuum flux, directly from continuum map

yy, xx = np.mgrid[:800,:800]
radius = np.sqrt((xx-pos_cen[1])**2 + (yy-pos_cen[0])**2)

size = 50
rms = 1.3e-02#sigma_clipped_stats(cont, sigma=3)[-1]
cont_level = np.array([-1,1,2,4,8,16,32])*3*rms

fig = plt.figure(figsize=(8,10))
ax = plt.subplot(projection=wcs[0,0])
im = ax.imshow(cont, cmap='jet', origin='lower')
cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
cb.set_label(r'cont [mJy$\cdot$beam$^{-1}$]')
ax.contour(cont, cont_level, colors=['k'], linewidths=1.)
ax.contour(radius, levels=[10, 30, 50], colors=['k'])

rec = matplotlib.patches.Rectangle((pos_cen[0]-size, pos_cen[1]-size), 10, 10, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2.5)
ax.add_artist(rec)
Beam = beam(hdu, pos_cen[1]-size+5,pos_cen[0]-size+5, 'k', pix_size)
ax.add_artist(Beam[0])

ax.set_xlim(pos_cen[1]-size,pos_cen[1]+size)
ax.set_ylim(pos_cen[0]-size,pos_cen[0]+size)
ax.set_xlabel("R.A. (J2000)", labelpad=0.5, fontsize=20)
ax.set_ylabel("Dec (J2000)", labelpad=-1.0, fontsize=20)

# %%
## Original result
mask = (radius<=30) & (abs(cont)>=3*rms)
beam2pixels = np.pi*hdu.header["BMAJ"]*hdu.header["BMIN"]/(4*np.log(2))/hdu.header["CDELT1"]**2
beam2pixels

f_cont = np.nansum(cont/beam2pixels*mask)
ef_cont = 4.2e-05*1e3
print("The continuum flux is:", f_cont, "mJy")

from astropy.cosmology import Planck15
A = len(np.where(mask)[0])*(0.05*u.arcsec/Planck15.arcsec_per_kpc_proper(0.061))**2/np.cos(np.rad2deg(41))

SFR = f_cont/2.2*26.3*u.Unit("M_sun/yr")
eSFR = ef_cont/2.2*26.3*u.Unit("M_sun/yr")
Sigma_SFR = SFR/A
eSigma_SFR = eSFR/A

# %%
## same region in CO(2-1) map
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
file = "PG0050_CO21-combine-line-10km-mosaic-mom0-pbcor.fits"

CO_data = fits.open(path+file)[0].data[0][0]
CO_hdr = fits.open(path+file)[0].header
CO_beam2pixels = np.pi*CO_hdr["BMAJ"]*CO_hdr["BMIN"]/(4*np.log(2))/CO_hdr["CDELT1"]**2
CO_beam2pixels

f_CO_cen = np.nansum(CO_data/CO_beam2pixels*mask)
print("The CO(2-1) flux within 1.3mm emitting region:", f_CO_cen, "Jy km/s")

DL = Planck15.luminosity_distance(0.061)
L_CO_cen = 3.25e7*f_CO_cen*DL.value**2/(1+0.061)/(230.58)**2/0.9
M_H2 = 1.27*L_CO_cen
M_H2_low = (1.27-0.71)*L_CO_cen
M_H2_up = (1.27+0.83)*L_CO_cen
M_H2_tra = 3.1*1.4*L_CO_cen

Sigma_H2 = M_H2*u.Unit("M_sun")/A.to("pc^2")
Sigma_H2_tra = M_H2_tra*u.Unit("M_sun")/A.to("pc^2")
Sigma_H2_low = M_H2_low*u.Unit("M_sun")/A.to("pc^2")
Sigma_H2_up = M_H2_up*u.Unit("M_sun")/A.to("pc^2")
e1Sigma_H2 = Sigma_H2_low - Sigma_H2
e2Sigma_H2 = Sigma_H2_up - Sigma_H2
e1Sigma_H2_tra = M_H2_tra*u.Unit("M_sun")/A.to("pc^2")/2
e2Sigma_H2_tra = M_H2_tra*u.Unit("M_sun")/A.to("pc^2")

# %%
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/cont/"
file = "PG0050-cont-mosaic-pbc.fits"
cont_origin = fits.open(path+file)[0].data[0][0]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
plt.imshow(cont)
test = np.zeros(10)
for i in np.arange(0, len(test)):
    r = np.random.uniform(50, 350)
    theta = 2*np.pi*np.random.randn()
    ex, ey = pos_cen[0]+r*np.cos(theta), pos_cen[1]+r*np.sin(theta)
    eradius = np.sqrt((xx-ex)**2+(yy-ey)**2)
    emask = eradius<9.44
    test[i] = np.nansum(cont_origin/beam2pixels*emask)
    ax.scatter(ex, ey, c="m")
    
ax.imshow(cont_origin, origin='lower', cmap='Greys')
print(np.nanstd(test))

# %%
## Peak value

npeak = np.where(cont == np.nanmax(cont))
print("The peak continuum flux is:", cont[npeak], "mJy/beam")
print("The peak CO(2-1) flux is:", CO_data[npeak], "Jy/beam km/s")

## surface density of SFR and CO flux
Acont_peak = beam2pixels*(0.05*u.arcsec/Planck15.arcsec_per_kpc_proper(0.061))**2/np.cos(np.rad2deg(41))

SFR_peak = cont[npeak][0]/2.2*26.3*u.Unit("M_sun/yr")
eSFR_peak = 0.012/2.2*26.3*u.Unit("M_sun/yr")
Sigma_SFR_peak = SFR_peak/Acont_peak
eSigma_SFR_peak = eSFR_peak/Acont_peak

## surface density of CO flux
ACO_peak = CO_beam2pixels*(0.05*u.arcsec/Planck15.arcsec_per_kpc_proper(0.061))**2/np.cos(np.rad2deg(41))
L_CO_peak = 3.25e7*CO_data[npeak][0]*DL.value**2/(1+0.061)/(230.58)**2/0.9
M_H2_peak = 1.27*L_CO_peak
M_H2_low_peak = (1.27-0.71)*L_CO_peak
M_H2_up_peak = (1.27+0.83)*L_CO_peak
M_H2_tra_peak = 3.1*1.4*L_CO_peak

Sigma_H2_peak = M_H2_peak*u.Unit("M_sun")/ACO_peak.to("pc^2")
Sigma_H2_tra_peak = M_H2_tra_peak*u.Unit("M_sun")/ACO_peak.to("pc^2")
Sigma_H2_low_peak = M_H2_low_peak*u.Unit("M_sun")/ACO_peak.to("pc^2")
Sigma_H2_up_peak = M_H2_up_peak*u.Unit("M_sun")/ACO_peak.to("pc^2")
e1Sigma_H2_peak = Sigma_H2_low_peak - Sigma_H2_peak
e2Sigma_H2_peak = Sigma_H2_up_peak - Sigma_H2_peak
e1Sigma_H2_tra_peak = M_H2_tra_peak*u.Unit("M_sun")/ACO_peak.to("pc^2")/2
e2Sigma_H2_tra_peak = M_H2_tra_peak*u.Unit("M_sun")/ACO_peak.to("pc^2")

# %%
