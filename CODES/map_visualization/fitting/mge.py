# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colorbar as colorbar
from matplotlib.colors import LogNorm
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from map_visualization.maps import beam
from matplotlib import colors

#%matplotlib inline
plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=20)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

# %%
import mgefit
from mgefit.find_galaxy import find_galaxy
from mgefit.mge_fit_1d import mge_fit_1d
from mgefit.sectors_photometry import sectors_photometry
from mgefit.mge_fit_sectors import mge_fit_sectors
from mgefit.mge_print_contours import mge_print_contours
from mgefit.mge_fit_sectors_twist import mge_fit_sectors_twist
from mgefit.sectors_photometry_twist import sectors_photometry_twist
from mgefit.mge_print_contours_twist import mge_print_contours_twist

# %%
# These parameters are given by find_galaxy for the mosaic image
skylevel = 0.0
sigmapsf = [0.356]  # pixels
normpsf = [1]
eps = 0.68
ang = 33.0  # major axis in the inner regions (gives a starting guess for the PA)
xc = 95
yc = 99
ngauss = 4
minlevel = 1.0
scale = 0.125

file_dir = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
# object = "PG0050_CO21-combine-line-10km-mosaic-mom0-pbcor.fits"
object = "PG0050_CO21-combine-line-10km-mosaic-psf.fits"
file = file_dir + object

hdu = fits.open(file)[0]
img = hdu.data[0][0][300:500, 300:500]

mask = img > 0   # mask before sky subtraction
img -= skylevel

    # Mask a nearby galaxy
# mask &= dist_circle(1408.09, 357.749, img.shape) > 200

plt.figure(figsize=(8, 8))
plt.clf()
f = find_galaxy(img, fraction=0.01, plot=1)
plt.pause(1)  # Allow plot to appear on the screen

# %%
plt.figure(figsize=(8, 8))
plt.clf()
s = sectors_photometry(img, f.eps, f.theta, f.xpeak, f.ypeak,
                           minlevel=0.08, plot=1)
plt.pause(1)  # Allow plot to appear on the screen

# %%
    # Do the actual MGE fit
    # *********************** IMPORTANT ***********************************
    # For the final publication-quality MGE fit one should include the line
    # "from mge_fit_sectors_regularized import mge_fit_sectors_regularized"
    # at the top of this file, rename mge_fit_sectors() into
    # mge_fit_sectors_regularized() and re-run the procedure.
    # See the documentation of mge_fit_sectors_regularized for details.
    # *********************************************************************
plt.figure(figsize=(12, 12))
plt.clf()
m = mge_fit_sectors(s.radius, s.angle, s.counts, f.eps, ngauss=ngauss, scale=scale, plot=1, bulge_disk=0, linear=0)
plt.pause(1)  # Allow plot to appear on the screen

# %%
# Show contour plots of the results

plt.clf()
plt.subplot(121)
mge_print_contours(img.clip(minlevel), f.theta, f.xpeak, f.ypeak, m.sol, scale=scale,
                       binning=1, sigmapsf=sigmapsf, normpsf=normpsf, magrange=5)

    # Extract the central part of the image to plot at high resolution.
    # The MGE is centered to fractional pixel accuracy to ease visual comparson.

n = 50
img = img[f.xpeak-n:f.xpeak+n, f.ypeak-n:f.ypeak+n]
xc, yc = n - f.xpeak + f.xmed, n - f.ypeak + f.ymed
plt.subplot(122)
mge_print_contours(img, f.theta, xc, yc, m.sol,
                       sigmapsf=sigmapsf, normpsf=normpsf, scale=scale)
plt.pause(1)  # Allow plot to appear on the screen

# %%
# Perform galaxy photometry
eps = 0.68
ang = 33.0  # major axis in the inner regions (gives a starting guess for the PA)
xc = 50
yc = 50
ngauss = 5

# file_dir = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
# object = "PG0050_CO21-combine-line-10km-mosaic-mom0-pbcor.fits"
# object = "PG0050_CO21-combine-line-10km-mosaic-psf.fits"
file_dir = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG0923/data"
object = "/PG0923+129_CO21_final_image_mom0.fits"

file = file_dir + object

hdu = fits.open(file)[0]
img = hdu.data[0][0][30:140, 30:140]

# mask = img > 0   # mask before sky subtraction
# skylevel = 0
# img -= skylevel

    # Mask a nearby galaxy
# mask &= dist_circle(1408.09, 357.749, img.shape) > 200

plt.figure(figsize=(8, 8))
plt.clf()
f = find_galaxy(img, fraction=0.20, plot=1)
plt.pause(1)  # Allow plot to appear on the screen

sigmapsf = []
normpsf = [1]

# %%
plt.figure(figsize=(8, 8))
plt.clf()
s = sectors_photometry_twist(img, f.pa, 58.97, 56.87, minlevel=0.08, plot=1)
plt.pause(1)  # Allow plot to appear on the screen

# %%
plt.figure(figsize=(12, 12))
plt.clf()
m = mge_fit_sectors_twist(s.radius, s.angle, s.counts, eps, ngauss=ngauss, scale=scale, plot=1, negative=True)
plt.pause(1)  # Allow plot to appear on the screen

# %%
fignum = 1
minrad = np.min(m.radius)*m.scale
maxrad = np.max(m.radius)*m.scale
mincnt = np.min(m.counts)
maxcnt = np.max(m.counts)
xran = minrad * (maxrad/minrad)**np.array([-0.02, +1.02])
yran = mincnt * (maxcnt/mincnt)**np.array([-0.05, +1.05])
yran = [5e-02, 2e1]

n = m.sectors.size
dn = int(round(n/6.))
nrows = (n-1)//dn + 1 # integer division

plt.figure(figsize=(12, 12))
fig, ax = plt.subplots(nrows, 2, sharex=True, sharey='col', num=fignum)
fig.subplots_adjust(hspace=0.01)

fig.text(0.04, 0.5, 'counts', va='center', rotation='vertical')
fig.text(0.96, 0.5, 'error (%)', va='center', rotation='vertical')

ax[-1, 0].set_xlabel("arcsec")
ax[-1, 1].set_xlabel("arcsec")

row = 0
for j in range(0, n, dn):
    w = np.nonzero(m.angle == m.sectors[j])[0]
    w = w[np.argsort(m.radius[w])]
    r = m.radius[w]*m.scale
    txt = "$%.f^\circ$" % m.sectors[j]

    ax[row, 0].set_xlim(xran)
    ax[row, 0].set_ylim(yran)
    ax[row, 0].loglog(r, m.counts[w], 'ko')            # counts obtained from image
    ax[row, 0].loglog(r, m.yfit[w], 'r', linewidth=2)  # total results of MGE
    ax[row, 0].text(0.98, 0.95, txt, ha='right', va='top', transform=ax[row, 0].transAxes)
    ax[row, 0].loglog(r, m.gauss[w, :]*m.weights[None, :], color='Gray')  #show each Gaussian profile

    # ax[row, 1].semilogx(r, m.err[w]*100, 'ko')
    ax[row, 1].semilogx(r, (m.counts[w] - m.yfit[w])/abs(m.counts[w])*100, 'ko')
    ax[row, 1].axhline(linestyle='--', color='k', linewidth=2)
    ax[row, 1].yaxis.tick_right()
    ax[row, 1].yaxis.set_label_position("right")
    ax[row, 1].set_ylim([-19.9, 20.0])

    row += 1

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

#----------------------------------------------------------------------------

def _gauss2d_mge(n, xc, yc, sx, sy, pos_ang):
    """
    Returns a 2D Gaussian image with size N[0]xN[1], center (XC,YC),
    sigma (SX,SY) along the principal axes and position angle POS_ANG, measured
    from the positive Y axis to the Gaussian major axis (positive counter-clockwise).

    """
    ang = np.radians(-pos_ang)
    x, y = np.ogrid[-xc:n[0] - xc, -yc:n[1] - yc]

    xcosang = np.cos(ang)/(np.sqrt(2.)*sx)*x
    ysinang = np.sin(ang)/(np.sqrt(2.)*sx)*y
    xsinang = np.sin(ang)/(np.sqrt(2.)*sy)*x
    ycosang = np.cos(ang)/(np.sqrt(2.)*sy)*y

    im = (xcosang + ysinang)**2 + (ycosang - xsinang)**2

    return np.exp(-im)

#----------------------------------------------------------------------------

def _multi_gauss(pars, img, sigmaPSF, normPSF, xpeak, ypeak, theta):

    lum, sigma, q, pa = pars

    # Analytic convolution with an MGE circular Gaussian
    # Eq.(4,5) in Cappellari (2002)
    #
    u = 0.
    for lumj, sigj, qj, paj in zip(lum, sigma, q, pa):
        for sigP, normP in zip(sigmaPSF, normPSF):
            sx = np.sqrt(sigj**2 + sigP**2)
            sy = np.sqrt((sigj*qj)**2 + sigP**2)
            g = _gauss2d_mge(img.shape, xpeak, ypeak, sx, sy, theta + paj)
            u += lumj*normP/(2.*np.pi*sx*sy) * g

    return u

#----------------------------------------------------------------------------

def mge_print_contours_twist_new(img, ang, xc, yc, sol, binning=None, normpsf=1,
                       magrange=10, mask=None, scale=None, sigmapsf=0):

    sigmapsf = np.atleast_1d(sigmapsf)
    normpsf = np.atleast_1d(normpsf)

    assert normpsf.size == sigmapsf.size, "sigmaPSF and normPSF must have the same length"
    assert round(np.sum(normpsf), 2) == 1, "PSF not normalized"

    if mask is not None:
        assert mask.dtype == bool, "MASK must be a boolean array"
        assert mask.shape == img.shape, "MASK and IMG must have the same shape"

    model = _multi_gauss(sol, img, sigmapsf, normpsf, xc, yc, ang)
    peak = img[int(round(xc)), int(round(yc))]
    levels = peak * 10**(-0.4*np.arange(0, magrange, 0.5)[::-1]) # 0.5 mag/arcsec^2 steps

    if binning is None:
        binning = 1
    else:
        model = ndimage.filters.gaussian_filter(model, binning/2.355)
        model = ndimage.zoom(model, 1./binning, order=1)
        img = ndimage.filters.gaussian_filter(img, binning/2.355)
        img = ndimage.zoom(img, 1./binning, order=1)

    ax = plt.gca()
    ax.axis('equal')
    ax.set_adjustable('box')
    s = img.shape

    if scale is None:
        extent = [0, s[1], 0, s[0]]
        plt.xlabel("pixels")
        plt.ylabel("pixels")
    else:
        extent = np.array([0, s[1], 0, s[0]])*scale*binning
        plt.xlabel("arcsec")
        plt.ylabel("arcsec")

    cnt = ax.contour(img, levels, colors = 'k', linestyles='solid', extent=extent)
    ax.contour(model, levels, colors='r', linestyles='solid', extent=extent)
    if mask is not None:
        a = np.ma.masked_array(mask, mask)
        ax.imshow(a, cmap='autumn_r', interpolation='nearest', origin='lower',
                  extent=extent, zorder=3, alpha=0.7)

    return img, model, cnt# %%

# %%
plt.figure(figsize=(8, 8))

plt.clf()
image, mge_model, mge_print = mge_print_contours_twist_new(img.clip(minlevel), f.pa, 58.97, 56.87, m.sol, scale=scale, binning=1, magrange=5)
plt.pause(1)  # Allow plot to appear on the screen
#, sigmapsf=sigmapsf, normpsf=normpsf
# %%
cmap = "jet"
pix_size = 0.18
f_res = img - mge_model
mom0_level = np.array([-1,1,2,4,8,16,32,64,128])*3*0.055#r_test
vmin, vmax = 0.25, np.percentile(img, [99.999])

fig, axes = plt.subplots(figsize=(18, 7), nrows=1, ncols=3)
plt.subplots_adjust(wspace=0)
ax0, ax1, ax2 = axes
im0 = ax0.imshow(img, vmin=-vmin, vmax=vmax, cmap=cmap, origin='lower', norm=colors.PowerNorm(gamma=0.525))
ax0.contour(img, mom0_level, colors=["k"], linewidths=1.)
ax0.text(10, 10, "DATA", color="w")
im1 = ax1.imshow(mge_model, vmin=-vmin, vmax=vmax, cmap=cmap, origin='lower', norm=colors.PowerNorm(gamma=0.525))
ax1.contour(mge_model, mom0_level, colors=["k"], linewidths=1.)
ax1.text(10, 10, "MODEL", color="w")
im2 = ax2.imshow(f_res, vmin=-0.86, vmax=0.86, cmap=cmap, origin='lower')
ax2.contour(f_res, mom0_level, colors=["k"], linewidths=1.)

for ax in axes[:]:
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.125, 0.115, 0.517, 0.05])
cb_ax = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
#cb_ax.set_label(r"$I_\mathrm{CO(2-1)}$ [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]")
cb_ax.set_label(r"$I_\mathrm{CO(2-1)}$ [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]")

cbar_res = fig.add_axes([0.643, 0.115, 0.257, 0.05])
cb_res = fig.colorbar(im2, cax=cbar_res, orientation='horizontal')
#cb_res.set_label(r"$I_\mathrm{res}$ [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]")
cb_res.set_label(r"$I_\mathrm{res}$ [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]")

rec = matplotlib.patches.Rectangle((0, 0), 10, 10,
angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
ax0.add_artist(rec)
Beam = beam(hdu, 5., 5., 'w', pix_size)
ax0.add_artist(Beam[0])

# %%
