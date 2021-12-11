from matplotlib.colors import LogNorm
from Physical_values.surface_density import surface_density, iso_rad, surface_density_mom0
from map_visualization.maps import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)


path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/"
file = "PG0050_CO21-combine-line-10km-mosaic-mom0.fits"
file_rms = "PG0050_CO21-combine-line-10km-mosaic-mom0-rms.fits"
file_mom2 = "PG0050_CO21-combine-line-10km-mosaic-mom2.fits"
radius = iso_rad(800, np.array([395, 399]), 0.05, 35, 41)

G = 4.302*10**(-3)

from Physical_values.stellar_properties import Sigma_disk

Sigma, re = np.power(10, 10.64)/40.965, 10.97
Sigma_s = Sigma_disk(Sigma, radius, re)
disp_s = 90
Sigma_H2 = surface_density(path, file_rms)
mom0, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file)
mom0_level = np.array([-1,1,2,4,8,16,32,64])*2*r
print(mom0_level)
mom0_rms, wcs, pos_cen, size, pix_size, r, hdu = load_mom0(path, file_rms)
mom2, wcs, size, pix_size, hdu, pos_cen = load_mom2(path, file_mom2)
disp_CO = np.sqrt(mom2**2)# - 9.156**2
Sigma_tot = Sigma_H2 + Sigma_s*2/(1+60**2/disp_CO**2)

regions_name = ["CND", "SPIRAL", "CENTER"]
center = radius<=0.8
CND = (radius<=2.1) & (radius>0.8)
spiral = radius>2.1
regions = [CND, spiral, center]
fmts = ["yo", "co", "go"]
mfcs = ["yellow", "cyan", "g"]
ecolors = ["green", "blue", "m"]

N = np.where(center == 1)
Sigma_H2[N] = Sigma_H2[N]/0.62*0.9

N = np.where(Sigma_H2 >= 0)
bins = np.logspace(1, 9, 30)
print(bins)
P_turb = 61.3*Sigma_H2*disp_CO**2/(400/40)

print(np.log10(np.nanmean(P_turb)), np.nanstd(P_turb))

M = []
for i in range(len(bins)-1):
    N = np.where((P_turb>bins[i]) & (P_turb<bins[i+1]))
    M.append(np.nansum(Sigma_H2[N]))

f_M = np.array(M)/np.nansum(Sigma_H2)

plt.figure(figsize=(8, 8))
plt.step(bins[:-1], f_M, where='mid')
plt.semilogx()
plt.show()