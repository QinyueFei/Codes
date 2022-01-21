# %%

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import Planck15
# %%
path = "/home/qyfei/Desktop/Results/Barolo/PG0050/Ha/"
file = "PG0050+124_Ha_SN3lim_fullmaps.fits"
hdu_Ha = fits.open(path + file)
file = "PG0050+124_Wimg_Rmap_fullmaps.fits"
hdu_Wimg = fits.open(path + file)

# %%
hdu_vlosHa = hdu_Ha[1]
vlosHa = hdu_Ha[1].data
vlosHa_hdr = hdu_Ha[1].header
evlosHa = hdu_Ha[5].data
vdispHa = hdu_Ha[2].data
evdispHa = hdu_Ha[6].data

Ha_pos_cen = [166, 167]

Ha_xslit = np.linspace(Ha_pos_cen[1]-80, Ha_pos_cen[1]+80, 30)
Ha_yslit = np.tan(np.radians(127.59-90))*(Ha_xslit-Ha_pos_cen[1])+Ha_pos_cen[0]

Ha_yy, Ha_xx = np.mgrid[:324, :331]
radius = np.sqrt((Ha_xx-Ha_xslit[5])**2 + (Ha_yy-Ha_yslit[5])**2)
vlosHa_level = np.linspace(-250, 250, 11)
plt.figure(figsize=(10, 8))
#ax = plt.subplot(111)
shift = 116
plt.imshow(vlosHa-shift, vmin=-250, vmax=250, cmap='jet', origin="lower")
plt.colorbar()
plt.contour(vlosHa-shift, vlosHa_level, colors=['k'])
plt.plot(Ha_xslit, Ha_yslit, 'k--', lw=3)
#plt.savefig(output_dir+"Ha_vlos.pdf", bbox_inches="tight", dpi=300)
#ax.colorbar()

# %%
hdu = fits.open(path+"vlosHa.fits")[0]
#hdu.header.append('CDELT1')
#hdu.header.append('CDELT2')
hdu.header['CDELT1'] = -5.5e-05
hdu.header['CDELT2'] = 5.5e-05
#hdu.header.append('CRPIX1')
#hdu.header.append('CRPIX2')
hdu.header['CRPIX1'] = 166
hdu.header['CRPIX2'] = 167
#hdu.header.append('CRVAL1')
#hdu.header.append('CRVAL2')
hdu.header['CRVAL1'] = 1.339554874996E+01
hdu.header['CRVAL2'] = 1.269338611111E+01

#hdu.header
# %%
#hdu.header.append('CTYPE1')
#hdu.header.append('CTYPE2')
hdu.header['CTYPE1'] = 'RA---SIN'
hdu.header['CTYPE2'] = 'DEC---SIN'


#hdu.header.append('CUNIT1')
#hdu.header.append('CUNIT2')
hdu.header['CUNIT1'] = 'deg'
hdu.header['CUNIT2'] = 'deg'


#hdu.header.append('BMAJ')
#hdu.header.append('BMIN')
#hdu.header.append('BPA')
hdu.header['BMAJ'] = 2.77777777778e-04
hdu.header['BMIN'] = 2.77777777778e-04
hdu.header['BPA'] = 0

#hdu.header
#fits.writeto(path+"vlosHa.fits", hdu.data, header=hdu.header, overwrite=True)

# %%
Ha_vrot = np.zeros(len(Ha_xslit))
Ha_evrot = np.zeros(len(Ha_xslit))
Ha_dis = np.zeros(len(Ha_xslit))
Ha_edis = np.zeros(len(Ha_xslit))
Ha_disp = np.zeros(len(Ha_xslit))
Ha_edisp = np.zeros(len(Ha_xslit))

for i in range(len(Ha_xslit)):
    radius = np.sqrt((Ha_xx-Ha_xslit[i])**2 + (Ha_yy-Ha_yslit[i])**2)
    mask = radius<5*np.exp(abs(i-15)/7.5)
    N = np.where(mask)
    Ha_vrot[i] = np.nanmean(vlosHa[N]-shift)
    Ha_evrot[i] = np.nanstd(vlosHa[N]-shift)
    Ha_disp[i] = np.nanmean(vdispHa[N])
    Ha_edisp[i] = np.nanstd(vdispHa[N])

Ha_position = (Ha_xslit-Ha_pos_cen[1])*0.20/np.cos(np.radians(130))
Ha_distance = Ha_position/Planck15.arcsec_per_kpc_proper(0.061).value

Ha_mask = np.where(abs(Ha_position)>3)
r_Ha = -Ha_distance[17:]
#Ha_dis = Ha_dis*0.2/Planck15.arcsec_per_kpc_proper(0.061).value
# %%
plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
ax.errorbar(Ha_position, Ha_disp, yerr=Ha_edisp, fmt='rs', mfc='r',  ms=8, mew=1, elinewidth=1, capsize=4)

# %%
r_fit_tot = np.array(list(r_fit[7:])+list(-Ha_distance[16:]))
vrot_tot = np.array(list(vrot_fit)[7:]+list(-Ha_vrot[16:]/np.sin(np.radians(39))))
evrot_tot = np.array(list(evrot2_fit)[7:]+list(-Ha_evrot[16:]/np.sin(np.radians(39))))

plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
ax.errorbar(distance, vrot/np.sin(np.radians(39)), yerr=evrot, fmt='ko', mfc='k', ms=8, mew=1, elinewidth=1, capsize=4)
ax.errorbar(r_fit, vrot_fit, yerr=[-evrot1_fit, evrot2_fit], fmt='bo', mfc='none', ms=8, mew=1, elinewidth=1, capsize=4)

#ax.errorbar(-r_fit, -vrot_fit, yerr=[-evrot1_fit, evrot2_fit], fmt='bo', mfc='none', ms=8, mew=1, elinewidth=1, capsize=4)
ax.errorbar(Ha_distance[Ha_mask], Ha_vrot[Ha_mask]/np.sin(np.radians(39)), yerr=Ha_evrot[Ha_mask], fmt='ro', mfc='r', ms=8, mew=1, elinewidth=1, capsize=4)
#ax.errorbar(-Ha_distance[Ha_mask], -Ha_vrot[Ha_mask]/np.sin(np.radians(39)), yerr=Ha_evrot[Ha_mask], fmt='rs', mfc='none', ms=8, mew=1, elinewidth=1, capsize=4)

ax.errorbar(r_fit_tot, vrot_tot, yerr=evrot_tot, fmt='ks', mfc='none', ms=8, mew=1, elinewidth=1, capsize=4)


ax.set_xlabel("Radius [kpc]")
ax.set_ylabel("$V_\mathrm{rot}$ [$\mathrm{km\,s^{-1}}$]")

#plt.savefig(output_dir+"vrot_dbl.pdf", bbox_inches="tight", dpi=300)
ax.set_xlim(0, 30)
#ax.semilogx()
# %%
path = "/home/qyfei/Desktop/Results/Barolo/PG0050/Ha/"
file = "PG0050+124_OIII-nuclear_intHa.fits"
hdu = fits.open(path+file)
# %%
plt.figure(figsize=(10, 8))
plt.imshow(evlosHa, vmin=0, vmax=20, cmap='jet', origin='lower')
plt.colorbar()
# %%


# %%
# %%
