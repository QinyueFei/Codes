# %%
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
# %%
# path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG0923/Barolo_fit/output/PG0923+129/"
# file = "spacepar_PG0923+129.fits"

path = '/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/Barolo_fit/output/PG0050+124'
# folder = "Barolo_fit/output/PG0050+124/"
file = "/spacepar_PG0050+124.fits"


f = fits.open(path+file)[0]
d, h = f.data, f.header
p1, p2 = h['CTYPE1'], h['CTYPE2']
p1u, p2u, ru = h['CUNIT1'], h['CUNIT2'], h['CUNIT3']
p1ran = (np.arange(0,d.shape[2])+1-h['CRPIX1'])*h['CDELT1']+h['CRVAL1']
p2ran = (np.arange(0,d.shape[1])+1-h['CRPIX2'])*h['CDELT2']+h['CRVAL2']
rings = (np.arange(0,d.shape[0])+1-h['CRPIX3'])*h['CDELT3']+h['CRVAL3']

# %%
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/Barolo_fit/output/PG0050+124_best/"
densprof = "denfprof.txt"

RADIUS, SUM, MEAN, MEDIAN, STDDEV, MAD, NPIX, SURFDENS, ERR_SD, SURFDENS_FO, MSURFDENS, MSURFDENS2 = np.genfromtxt(path+"/densprof.txt", skip_header=1,usecols=(0,1,2,3,4,5,6,7,8,9,10,11),unpack=True)


chi2 = (d.T * NPIX).T
# %%
contour_level = np.percentile(d, [16, 50, 84])

# contour_level = np.array([0.01, 0.05, 0.10])

# %%
nrad = d.shape[0]
ncols = 4
nrows = int(np.ceil(nrad/float(ncols)))
ext = [p1ran[0],p1ran[-1],p2ran[0],p2ran[-1]]
cmap = plt.get_cmap('nipy_spectral') #plt.get_cmap('gnuplot')

plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=15)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)


fig = plt.figure(figsize=(8,8))
x_axis_len, y_axis_len = 0.27, 0.27 
x_sep, y_sep = 0.07, 0.08 
count = 0
axis, bottom_corner = [], [0.1,0.7]
for i in range (nrows):
    bottom_corner[0] = 0.1
    for j in range (ncols):
        if (count>=nrad): break
        axis.append(fig.add_axes([bottom_corner[0],bottom_corner[1],x_axis_len,y_axis_len]))
        bottom_corner[0]+=x_axis_len+x_sep
        count += 1
    bottom_corner[1]-=(y_axis_len+y_sep)

for i in range (nrad):
    nr = int(i/ncols) + 1
    nc = i - (nr-1)*ncols + 1
    toplot = d[i]
    # toplot = chi2[i]
    # contour_level = np.percentile(toplot, [16, 50, 84])
    # print(contour_level)

    a = np.unravel_index(np.argmin(toplot),toplot.shape)
    p1min, p2min = p1ran[a[1]], p2ran[a[0]]
    ax = axis[i]
    ax.set_xlim(ext[0],ext[1])
    ax.set_ylim(ext[2],ext[3])
    ax.imshow(toplot,origin='lower',extent=ext,aspect='auto',cmap=cmap)
    ax.contour(toplot,contour_level,extent=ext,colors=['w','r','k'])
    ax.plot(p1min,p2min,'x',mew=2,ms=8,c='w')
    radstr = 'R = %.2f %s'%(rings[i],ru)
    minstr = 'min = (%.1f %s, %.1f %s)'%(p1min,p1u,p2min,p2u)
    ax.text(0.01,1.1,radstr,transform=ax.transAxes)
    ax.text(0.01,1.03,minstr,transform=ax.transAxes)
    if nc==1: ax.set_ylabel(p2+' ('+p2u+')')
    if (nr==nrows) or (nr==nrows-1 and nrad%ncols!=0 and nc>nrad%ncols):
    	ax.set_xlabel(p1+' ('+p1u+')')

# plt.savefig("/home/qyfei/Desktop/Results/Barolo/PG_quasars/PG0923/par_space.pdf", bbox_inches="tight", dpi=300)

# %%

f_res_tot = np.reshape(d[:], [12*61*74])

value = np.percentile(f_res_tot, [16, 50, 84])

plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
counts, chi2v, figs = ax.hist(f_res_tot, bins=50)

ax.vlines(value, 0, 10000, color='k', ls='--')

ax.set_xlabel("$\chi^2$")
ax.set_ylabel("Counts")
    
# %%

mean_VROT, mean_VDISP, sigma_VROT, sigma_VDISP = np.zeros(nrad), np.zeros(nrad), np.zeros(nrad), np.zeros(nrad)

VROT_range = np.arange(ext[0], ext[1]+6, 6)
VDISP_range = np.arange(ext[2], ext[3]+6, 6)

contour_level = np.percentile(d, [5, 50, 84])

for i in range(nrad):
    f_res = d[i]

    contour_level = np.percentile(f_res, [5, 50, 84])

    par = np.where(f_res <= contour_level[0])
    weights = f_res[par[0], par[1]]

    mean_VROT[i] = np.average(VROT_range[par[1]], weights=weights)
    mean_VDISP[i] = np.average(VDISP_range[par[0]], weights=weights)

    var_VROT = np.average((VROT_range[par[1]]-mean_VROT[i])**2, weights=weights)
    sigma_VROT[i] = np.sqrt(var_VROT)
    var_VDISP = np.average((VDISP_range[par[0]]-mean_VDISP[i])**2, weights=weights)
    sigma_VDISP[i] = np.sqrt(var_VDISP)

    # sigma_VDISP[i] = np.nanstd(VDISP_range[par[0]])

# %%

# path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/Type1AGN/PG_quasars/PG0923"
# folder = "/Barolo_fit/output/PG0923+129_well"
path = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/Barolo_fit/output/PG0050+124_best/"


RAD_kpc, RAD_arcs, VROT, DISP, INC, PA, Z0, Z0_arcs, SIG, XPOS, YPOS, VSYS, VRAD, E_VROT1, E_VROT2, E_DISP1, E_DISP2, E_INC1, E_INC2, E_PA1, E_PA2  = np.genfromtxt(path+"/ringlog1.txt",skip_header=1,unpack=True)

RAD_kpc, RAD_arcs, VROT, DISP, INC, PA, Z0, Z0_arcs, SIG, XPOS, YPOS, VSYS, VRAD, E_VROT1, E_VROT2, E_DISP1, E_DISP2 = np.genfromtxt(path+"/ringlog2.txt",skip_header=1,unpack=True)

# %%
fig, axes = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
ax0 = axes
msize = 4
ax0.errorbar(RAD_kpc, VROT, yerr=[-E_VROT1, E_VROT2], fmt='bo', mfc='b', ms=msize, mew=1, elinewidth=1, capsize=5, label="$V_\mathrm{rot}$", zorder=3)
ax0.fill_between(RAD_kpc, VROT+E_VROT1, VROT+E_VROT2, color='b', alpha=0.3, zorder=3)

ax0.errorbar(RAD_kpc, mean_VROT, yerr=sigma_VROT, fmt='ks', mfc='none', ms=msize, mew=1, elinewidth=1, capsize=5, label="$V_\mathrm{rot}$", zorder=3)

# ax0.errorbar(RAD_kpc1, VROT1, yerr=[-E_VROT11, E_VROT21], fmt='bo', mfc='none', ms=msize, mew=1, elinewidth=1, capsize=5, label="$V_\mathrm{rot}$", zorder=3)
# ax0.fill_between(RAD_kpc1, VROT1+E_VROT11, VROT1+E_VROT21, color='b', alpha=0.3, zorder=3)


ax0.set_xlabel('Radius [kpc]')
ax0.set_ylabel('$V_\mathrm{rot}$ [$\mathrm{km\,s^{-1}}$]')

ax1 = ax0.twinx()
ax1.errorbar(RAD_kpc, DISP, yerr=[-E_DISP1, E_DISP2], fmt='ro', mfc='r', ms=msize, mew=1, elinewidth=1, capsize=5, zorder=3)
ax1.fill_between(RAD_kpc, DISP+E_DISP1, DISP+E_DISP2, color='r', alpha=0.3, zorder=3)

ax1.errorbar(RAD_kpc, mean_VDISP, yerr=sigma_VDISP, fmt='cs', mfc='none', ms=msize, mew=1, elinewidth=1, capsize=5, zorder=3)


# ax1.errorbar(RAD_kpc1, DISP1, yerr=[-E_DISP11, E_DISP21], fmt='ro', mfc='r', ms=msize, mew=1, elinewidth=1, capsize=5, zorder=3)
# ax1.fill_between(RAD_kpc1, DISP1+E_DISP11, DISP1+E_DISP21, color='r', alpha=0.3, zorder=3)


ax1.set_ylabel("$\sigma$ [$\mathrm{km\,s^{-1}}$]")

ax1.errorbar(-100,0,0, fmt='bo', mfc='b', ms=msize, mew=1, capsize=5, elinewidth=1, label='$V_\mathrm{rot}$')
ax1.errorbar(-100,0,0, fmt='ro', mfc='r',ms=msize, mew=1, capsize=5, elinewidth=1, label='$\sigma$')
ax1.legend(loc='upper left')

for ax in [ax0, ax1]:
    # ax.set_xlim(1e-4, RAD_kpc[-1]+(RAD_kpc[2]-RAD_kpc[1])/2)
    # ax.set_ylim(-5, 335)
    ax.set_xlim(1e-2, 3.5)
    ax.set_ylim(-5, 355)
    # ax.semilogx()
    ax.grid()
    ax.fill_between([0, 2*(RAD_kpc[1]-RAD_kpc[0])], [-500], [500], color='k', hatch='/', alpha=0.1)

# plt.savefig("/home/qyfei/Desktop/Results/Barolo/PG_quasars/PG0923/velocity_profile_per_ring.pdf", bbox_inches="tight", dpi=300)
# x = np.logspace(-3,3,1000)
# y = np.sqrt(G*10**6.62/x)

# ax.plot(x, y, 'k-')

# %%
