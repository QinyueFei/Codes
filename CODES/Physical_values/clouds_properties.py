from matplotlib.colors import LogNorm
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from map_visualization.maps import *
from astropy.cosmology import Planck15
from spectral_cube import SpectralCube

def clouds(path_, file_, ith_): 
    # This function is used to extract the cloud region that are identified from previous result
    path, file, i = path_, file_, ith_
    hdu = fits.open(path+file)
    props = hdu[1].data[i]
    xctr = props['XCTR_PIX']
    yctr = props['YCTR_PIX']
    vctr = props['VCTR_PIX']
    mom2maj= abs(props['MOMMAJPIX'])
    mom2min = abs(props['MOMMINPIX'])
    momvpix = props['MOMVPIX']
    FWHMmaj = np.sqrt(8*np.log(2))*mom2maj
    FWHMmin = np.sqrt(8*np.log(2))*mom2min
    PA = props['POSANG']*u.rad.to('deg')
    inc = np.arccos(mom2min/mom2maj)*u.rad.to('deg')
    #sigv = props['SIGV_KMS']
    return xctr, yctr, vctr, PA, inc, FWHMmaj, FWHMmin, momvpix
    
path = '/home/qyfei/Desktop/Codes/CODES/Physical_values/clouds/'
file = 'PG0050_props.fits'
from Physical_values.surface_density import iso_rad
from map_visualization.maps import *

def clouds_spectra(path_, file_, i_, cube_directory_):
    ## Construct the spectrum of each cloud
    path, file, i = path_, file_, i_
    cloud_xctr, cloud_yctr, cloud_vctr, cloud_PA, cloud_inc, cloud_FWHMmaj, cloud_FWHMmin, momvpix = clouds(path, file, i)
    Radius = iso_rad(800, [cloud_yctr, cloud_xctr], 0.05, cloud_PA, cloud_inc)
    mask = Radius<=cloud_FWHMmaj*0.05*Planck15.arcsec_per_kpc_proper(0.061).value

    file_cube = cube_directory_
    hdu = fits.open(file_cube)[0]
    bmaj = hdu.header['BMAJ']
    bmin = hdu.header['BMIN']
    delt = hdu.header['CDELT1']
    beam_area = np.pi*bmaj*bmin/delt**2/4/np.log(2)

    cube = hdu.data[0]
    spectrum = cube * mask
    spectrum[spectrum == 0] = np.nan
    flux = np.nanmean(spectrum, axis=(1, 2))*1e3

    return flux

def clouds_spectra_save(path_, file_, cube_directory_, outfile_):
    outfile = outfile_
    path, file, cube_directory = path_, file_, cube_directory_
    FLUXES = []
    for i in range(78):
        FLUXES.append(clouds_spectra(path, file, i, cube_directory))
    np.savetxt(outfile, np.array(FLUXES).T)

path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
file_cube = "PG0050_CO21-combine-line-10km-mosaic.fits"
cube_directory = path+file_cube
cube = SpectralCube.read(cube_directory)
CO21_cube = cube.with_spectral_unit(unit='km/s', rest_value=217.253*u.GHz, velocity_convention='radio') 
velo = CO21_cube.spectral_axis.value

path = '/home/qyfei/Desktop/Codes/CODES/Physical_values/clouds/'
file = 'PG0050_props.fits'
outfile = "/home/qyfei/Desktop/Codes/CODES/Physical_values/clouds/spectral.txt"

#flux = clouds_spectra(path, file, 68, cube_directory)
#plt.step(velo, flux)
#plt.show()
#clouds_spectra_save(path, file, cube_directory, outfile)

#FLUXES = np.loadtxt("/home/qyfei/Desktop/Codes/CODES/Physical_values/clouds/spectral.txt")
#plt.step(velo, FLUXES[:,10])
#plt.show()
#print(FLUXES[:,10])