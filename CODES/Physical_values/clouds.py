from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from map_visualization.maps import *
from skimage.feature import peak_local_max
from matplotlib.colors import LogNorm
from spectral_cube import SpectralCube
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
file = "PG0050_CO21-combine-line-10km-mosaic.fits"

def finds(path, file):
    hdu = fits.open(path+file)[0]
    CO21_cube = hdu.data[0]
    sigma_rms = np.nanstd(CO21_cube[:85, 150:])
    coordinates = peak_local_max(CO21_cube, min_distance=5, threshold_abs = 7*sigma_rms)
    vpos = coordinates[:,0]
    ypos = coordinates[:,1]
    xpos = coordinates[:,2]
    return vpos, ypos, xpos

from pycprops import pycprops
import astropy.units as u
datadir = "/home/qyfei/Desktop/Codes/TEST"
cubefile = "PG0050_CO21-combine-line-10km-mosaic.fits"
outputdir = "/home/qyfei/Desktop/Codes/CODES/Physical_values/clouds"
d = 264*u.Mpc

pycprops.fits2props(cubefile, 
                    datadir=datadir, 
                    output_directory=outputdir, 
                    distance=d, 
                    asgnname="PG0050_asgn.fits", 
                    propsname="PG0050_props.fits", 
                    allow_huge=True)
