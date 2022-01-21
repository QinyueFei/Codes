import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.cosmology import Planck15
import astropy.units as u
import matplotlib.pyplot as plt
from map_visualization.maps import *

def surface_density(path_, file_, alpha_CO_):
    path, file = path_, file_
    alpha_CO = alpha_CO_      #The conversion factor between flux and mass, normalize to 1.0
    z = 0.06115          #The redshift of the galaxy
    DL = Planck15.luminosity_distance(z) # The luminosity distance
    nu_obs = 230.58/(1+z) #The observation frequency
    inc = 41*u.deg      #The inclination angle of the galaxy
    mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file)
    bmaj = hdu.header['BMAJ']
    bmin = hdu.header['BMIN']
    delt = hdu.header['CDELT1']
    CO_pix = np.pi*bmaj*bmin/(4*np.log(2)*delt**2) #Derive how many pixels are contained by the beam
    # Estimate the luminosity and mass of molecular gas
    # 0.62 is the ratio between different rotational transition line and CO(1-0)
    L_CO = 3.25e7*mom0*DL.value**2/((1+z)**3*nu_obs**2)/0.62
    M_H2 = alpha_CO*L_CO*u.Unit('M_sun')
    # The 1.36 denote the Helium contribution
    # 1e6 suggests the final unit is M_sun/pc^2
    Sigma_H2 = M_H2/CO_pix/(pix_size*u.arcsec/Planck15.arcsec_per_kpc_proper(z))**2*np.cos(inc)*1.36/1e6

    L_COr = 3.25e7*2*r*DL.value**2/((1+z)**3*nu_obs**2)/0.62
    M_H2r = alpha_CO*L_COr*u.Unit('M_sun')
    Sigma_H2r = M_H2r/CO_pix/(pix_size*u.arcsec/Planck15.arcsec_per_kpc_proper(z))**2*np.cos(inc)*1.36/1e6

    return Sigma_H2.value, Sigma_H2r.value    
    

def iso_rad(sizex_, sizey_, pos_cen_, pix_size_, PA_, inc_):
    sizex, sizey, pos_cen, pix_size, PA, inc = sizex_, sizey_, pos_cen_, pix_size_, PA_, inc_
    # This function calculate the radius between each pixel and the kinematic center
    # size, pos_cen, pix_size are adopted from observation, which are map size, coordinates of galaxy center and size of each pixel
    # PA, inc are position angle and inclination angle
    z = 0.06115
    yy,xx = np.indices([sizey, sizex],dtype='float')
    coordinates_xx = (xx-pos_cen[1])*np.cos(PA*u.deg).value + (yy-pos_cen[0])*np.sin(PA*u.deg).value
    coordinates_yy = -(xx-pos_cen[1])*np.sin(PA*u.deg).value + (yy-pos_cen[0])*np.cos(PA*u.deg).value
    Radius_pixel = np.sqrt(coordinates_xx**2 + coordinates_yy**2/(np.cos(inc*u.deg).value)**2)
    Radius = (Radius_pixel * pix_size*u.arcsec/Planck15.arcsec_per_kpc_proper(z)).to('kpc')
    return Radius.value

def radius(x_, y_, pos_cen_, pix_size_, PA_, inc_):
    x, y, pos_cen, pix_size, PA, inc = x_, y_, pos_cen_, pix_size_, PA_, inc_
    z = 0.06115
    #yy,xx = np.indices([size, size],dtype='float')
    coordinates_xx = (x-pos_cen[1])*np.cos(PA*u.deg).value + (y-pos_cen[0])*np.sin(PA*u.deg).value
    coordinates_yy = -(x-pos_cen[1])*np.sin(PA*u.deg).value + (y-pos_cen[0])*np.cos(PA*u.deg).value
    Radius_pixel = np.sqrt(coordinates_xx**2 + coordinates_yy**2/(np.cos(inc*u.deg).value)**2)
    Radius = (Radius_pixel * pix_size*u.arcsec/Planck15.arcsec_per_kpc_proper(z)).to('kpc')
    return Radius.value

def surface_density_mom0(mom0_, hdu_):
    mom0, hdu = mom0_, hdu_
    alpha_CO = 3.1      #The conversion factor between flux and mass
    z = 0.06115          #The redshift of the galaxy
    DL = Planck15.luminosity_distance(z) # The luminosity distance
    nu_obs = 230.58/(1+z) #The observation frequency
    inc = 41*u.deg      #The inclination angle of the galaxy
    pix_size = 0.05
    bmaj = hdu.header['BMAJ']
    bmin = hdu.header['BMIN']
    delt = hdu.header['CDELT1']
    CO_pix = np.pi*bmaj*bmin/(4*np.log(2)*delt**2) #Derive how many pixels are contained by the beam
    # Estimate the luminosity and mass of molecular gas
    # 0.62 is the ratio between different rotational transition line and CO(1-0)
    L_CO = 3.25e7*mom0*DL.value**2/((1+z)**3*nu_obs**2)*0.62 
    M_H2 = alpha_CO*L_CO*u.Unit('M_sun')
    # The 1.36 denote the Helium contribution
    # 1e6 suggests the final unit is M_sun/pc^2
    Sigma_H2 = M_H2/CO_pix/(pix_size*u.arcsec/Planck15.arcsec_per_kpc_proper(z))**2*np.cos(inc)*1.36/1e6
    return Sigma_H2.value

