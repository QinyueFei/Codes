import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.colorbar as colorbar
import astropy.units as u
from astropy.stats import sigma_clipped_stats

## Adopt the beam shape of observation
def beam(HDU, XPOS, YPOS, col, cellsize):
    hdu = HDU
    xpos, ypos = XPOS, YPOS
    c = col
    cell = cellsize
    bmaj, bmin, bpa = hdu.header['BMAJ'], hdu.header['BMIN'], hdu.header['BPA']
    Bmaj = bmaj*u.Unit('deg').to('arcsec')/cell
    Bmin = bmin*u.Unit('deg').to('arcsec')/cell
    Beam = matplotlib.patches.Ellipse((xpos,ypos), Bmaj, Bmin, 90+bpa, edgecolor='k', facecolor=c, fill=True, zorder=3)
    return Beam, Bmaj, Bmin

## Deal with moment 0 maps
def load_mom0(path, file):
    # mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file)
    hdu = fits.open(path+file)[0]
    mom0 = hdu.data[0][0]
    wcs = WCS(hdu.header)
    pos_cen = np.where(mom0 == np.nanmax(mom0))
    pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    r = sigma_clipped_stats(mom0)[-1]
    size = 200
    # Output map data, coordinate system, galaxy center, size of each pixel, noise level and beam shape
    return mom0, wcs, pos_cen, size, pix_size, r, hdu

def load_mom0_NOEMA(path, file):
    # mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file)
    hdu = fits.open(path+file)[0]
    mom0 = hdu.data[0]
    wcs = WCS(hdu.header)
    pos_cen = np.where(mom0 == np.nanmax(mom0))
    pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    r = sigma_clipped_stats(mom0)[-1]
    size = 200
    # Output map data, coordinate system, galaxy center, size of each pixel, noise level and beam shape
    return mom0, wcs, pos_cen, size, pix_size, r, hdu


def plot_mom0(path, file):
    mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file)
    fov = 200 #define the size of the map
    mom0_level = np.array([-1,1,2,4,8,16,32])*3*r
    fig = plt.figure(figsize=(8,10))
    ax = plt.subplot(projection=wcs[0,0])
    im = ax.imshow(mom0, cmap='jet', origin='lower')
    cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
    cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
    cb.set_label('CO(2-1) [$\mathrm{Jy\,beam^{-1}\,km\,s^{-1}}$]')
    ax.contour(mom0, mom0_level, colors=['k'])
    ## plot the figure
    rec = matplotlib.patches.Rectangle((pos_cen[0]+100, pos_cen[1]-200), 200, 30, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
    ax.add_artist(rec)
    xbar = np.linspace(pos_cen[0]+110, pos_cen[0]+110+82.05) 
    ybar = (pos_cen[1]-195)*xbar/xbar
    ax.plot(xbar, ybar, 'k', lw=2, zorder=2.5) #plot the representer bar
    ax.text(xbar[0]+20,ybar[0]+5,'5kpc',size=20, color='k', zorder=2.5)
    ## Add beam and annotated information
    rec_size = 10
    rec = matplotlib.patches.Rectangle((pos_cen[0]-fov, pos_cen[1]-fov), rec_size, rec_size, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2)
    ax.add_artist(rec)
    Beam = beam(hdu, pos_cen[0]-size+rec_size/2,pos_cen[1]-size+rec_size/2, 'k', pix_size)
    ax.add_artist(Beam[0])
    ## Set the limit of x- and y-axis
    ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
    ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)
    ax.set_xlabel("R.A. (J2000)", labelpad=0.5, fontsize=20)
    ax.set_ylabel("Dec (J2000)", labelpad=-1.0, fontsize=20)

## Deal with moment 1 maps
def load_mom1(path, file):
    # mom1, wcs, size, pix_size, hdu, pos_cen = load_mom1(path, file)
    hdu = fits.open(path+file)[0]
    wcs = WCS(hdu.header)
    pos_cen = [395, 399]
    mom1 = hdu.data[0][0] - hdu.data[0][0][395, 399]
    size = 200
    pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    return mom1, wcs, size, pix_size, hdu, pos_cen
def plot_mom1(path, file):
    mom1, wcs, size, pix_size, hdu, pos_cen = load_mom1(path, file)
    level = np.linspace(-220,220,12)
    print(level)
    fig = plt.figure(figsize=(8,10))
    ax = plt.subplot(projection=wcs[0,0])
    im = ax.imshow(mom1, vmin=-220, vmax=220, cmap=plt.cm.get_cmap('coolwarm', 11), origin='lower')
    cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
    cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
    cb.set_label(r'$V_\mathrm{los}$ [$\mathrm{km\,s^{-1}}$]')
    ax.contour(mom1, level, colors=['k'])
    rec = matplotlib.patches.Rectangle((pos_cen[0]-size, pos_cen[1]-size), 10, 10, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2.5)
    ax.add_artist(rec)
    Beam = beam(hdu, pos_cen[0]-size+5,pos_cen[1]-size+5, 'k', pix_size)
    ax.add_artist(Beam[0])
    ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
    ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)
    ax.set_xlabel("R.A. (J2000)", labelpad=0.5, fontsize=20)
    ax.set_ylabel("Dec (J2000)", labelpad=-1.0, fontsize=20)

## Deal with moment 2 maps
def load_mom2(path, file):
    # mom2, wcs, size, pix_size, hdu, pos_cen = load_mom2(path, file)
    hdu = fits.open(path+file)[0]   ## header of observation data
    wcs = WCS(hdu.header)           ## coordinates system
    pos_cen = [395, 399]            ## galaxy center
    mom2 = hdu.data[0][0]           ## moment 2 data
    size = 200                      ## field of view
    pix_size = hdu.header['CDELT1']*u.deg.to('arcsec')
    return mom2, wcs, size, pix_size, hdu, pos_cen
def plot_mom2(path, file):
    mom2, wcs, size, pix_size, hdu, pos_cen = load_mom2(path, file)
    level = np.linspace(5,105,6)
    print(level)
    fig = plt.figure(figsize=(8,10))
    ax = plt.subplot(projection=wcs[0,0])
    im = ax.imshow(mom2, vmin=5, vmax=110, cmap='jet', origin='lower')
    cp,kw = colorbar.make_axes(ax, pad=0.01, aspect=18, location='top')
    cb = plt.colorbar(im, cax=cp, orientation='horizontal', ticklocation='top')
    cb.set_label(r'$\sigma_\mathrm{gas}$ [$\mathrm{km\,s^{-1}}$]')
    ax.contour(mom2, level, colors=['k'])
    rec = matplotlib.patches.Rectangle((pos_cen[0]-size, pos_cen[1]-size), 10, 10, angle=0.0,fill=True, edgecolor='k', facecolor='w', zorder=2.5)
    ax.add_artist(rec)
    Beam = beam(hdu, pos_cen[0]-size+5,pos_cen[1]-size+5, 'k', pix_size)
    ax.add_artist(Beam[0])
    ax.set_xlim(pos_cen[0]-size,pos_cen[0]+size)
    ax.set_ylim(pos_cen[1]-size,pos_cen[1]+size)
    ax.set_xlabel("R.A. (J2000)", labelpad=0.5, fontsize=20)
    ax.set_ylabel("Dec (J2000)", labelpad=-1.0, fontsize=20)
