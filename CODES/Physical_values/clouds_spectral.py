import numpy as np
from astropy.modeling import models
from spectral_cube import SpectralCube
from clouds_properties import clouds
import astropy.units as u
import emcee
from multiprocessing import Pool
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from scipy.optimize import minimize

def Gauss(x_, a_, m_, s_):
    x, a, m, s = x_, a_, m_, s_
    s = np.sqrt(9.156**2+s**2)
    gauss = models.Gaussian1D(a, m, s)
    return gauss(x)

def log_likelihood(theta, x, y, yerr):
    a, m, s = theta
    model = Gauss(x, a, m, s)
    sigma2 = yerr ** 2# + model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2)# + np.log(sigma2))

def log_prior(para):
    a, m, s = para
    if 0<a<80 and -500<m<500 and 0<=s<500:
        return 0.0
    return -np.inf

def log_probability(para, x, y, yerr):
    lp = log_prior(para)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(para, x, y, yerr)


def Gaussian_fit(path_, file_, i_, cube_directory_, flux_directory_, outputdir_):
    ## FIT the spectrum constructed from GMC with Gaussian profile
    from scipy.optimize import minimize
    para_A = []
    para_m = []
    para_s = []
    para_f = []
    para = [para_A, para_m, para_s, para_f]

    path, file, i = path_, file_, i_
    cloud_xctr, cloud_yctr, cloud_vctr, cloud_PA, cloud_inc, cloud_FWHMmaj, cloud_FWHMmin, momvpix = clouds(path, file, i)
    cube_directory = cube_directory_
    cube = SpectralCube.read(cube_directory)
    CO21_cube = cube.with_spectral_unit(unit='km/s', rest_value=217.253*u.GHz, velocity_convention='radio') 
    velo = CO21_cube.spectral_axis.value

    FLUXES = np.loadtxt(flux_directory_)
    f = FLUXES[:,i]
    f_err = sigma_clipped_stats(f)[-1]

    print("Fit with maximum likelyhood:")
    np.random.seed(42)
    nll = lambda *args: -log_likelihood(*args)
    a_g, m_g, s_g = 1, velo[int(cloud_vctr)], 0
    initial = np.array([a_g, m_g, s_g]) + 0.1 * np.random.randn(3)
    soln = minimize(nll, initial, args=(velo, f, f_err))
    a_ml, m_ml, s_ml = soln.x
    print(soln.x)
    pos = np.array([a_ml, m_ml, s_ml]) + 1e-4 * np.random.randn(32, 3)
    nwalkers, ndim = pos.shape
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(velo, f, f_err), pool=pool)
        sampler.run_mcmc(pos, 5000, progress=True)
    samples = sampler.get_chain()
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

    para_fit = np.zeros(3)
    #labels = ["A", "m", "s"]
    for j in range(ndim):
        mcmc = np.percentile(flat_samples[:, j], [16, 50, 84])
        para_fit[j] = mcmc[1]
        q = np.diff(mcmc)
        para[j].append([mcmc[1], q[0], q[1]])

    from scipy.integrate import trapz
    x_test = np.linspace(velo[0], velo[-1], 1000)
    flux_model = np.zeros(len(flat_samples))
    for j in range(len(flat_samples)):
        f_model = Gauss(x_test, flat_samples[j][0], flat_samples[j][1], flat_samples[j][2])
        F_model = trapz(f_model, x_test)
        flux_model[j] = F_model

    FLUX = np.percentile(flux_model, [16, 50, 84])
    q = np.diff(FLUX)
    para[3].append([FLUX[1], q[0], q[1]])

    inds = np.random.randint(len(flat_samples), size=100)

    plt.figure(figsize=(16, 8))
    grid=plt.GridSpec(6,1,wspace=0,hspace=0)
    ax1=plt.subplot(grid[0:5])
    ax2=plt.subplot(grid[5:])
    ax1.step(velo, f,'k',label='Spectrum', where='mid')
    for ind in inds:
        sample = flat_samples[ind]
        ax1.plot(x_test, Gauss(x_test, sample[0], sample[1], sample[2]), "r", alpha=0.1)
    ax1.plot(x_test, Gauss(x_test, para_fit[0], para_fit[1], para_fit[2]), 'r', label='Fit')
    ax1.fill_between(velo, -f_err, f_err, facecolor='k',hatch='/',linestyle=':',alpha=0.5, label=r'1$\sigma$ noise')
    w50 = np.sqrt(2*np.log(2))*np.sqrt(para_fit[2]**2 + 9.156**2) 
    #note the later term is the sigma!!!
    ax1.vlines(para_fit[1]-w50, -100,500, 'b', ls='--', label='$W_{50}$')
    ax1.vlines(para_fit[1]+w50, -100,500, 'b', ls='--')

    ax1.hlines(0,-1000,1000,'k',':')
    ax1.set_xlim(para_fit[1]-8*w50, para_fit[1]+8*w50)
    ax1.set_ylim(-3*f_err, para_fit[0]+5*f_err)
    ax1.set_ylabel('Flux density [mJy/beam]')
    ax1.legend(loc='upper left', frameon=False)

    res = f - Gauss(velo, para_fit[0], para_fit[1], para_fit[2])
    ax2.step(velo, res, 'k', where='mid')
    ax2.fill_between(velo, -f_err, f_err, facecolor='k',hatch='/',linestyle=':',alpha=0.5)
    ax2.hlines(0,-1000,1000,'k',':')
    ax2.set_xlim(para_fit[1]-8*w50, para_fit[1]+8*w50)
    ax2.set_ylim(-4*f_err, 4*f_err)
    ax2.set_xlabel("Velocity [km/s]")
    ax2.set_ylabel("Residual [mJy/beam]")
    outputdir = outputdir_
    plt.savefig(outputdir+"spectral_of_%02ith_cloud.pdf"%(i+1), bbox_inches="tight", dpi=300)
    #np.savetxt(outputdir+"chi2.txt", np.array(chi2_tot))
    return para

path = '/home/qyfei/Desktop/Codes/CODES/Physical_values/clouds/'
file = 'PG0050_props.fits'
cube_directory = "/media/qyfei/f6e0af82-2ae6-44a3-a033-f66b47f50cf4/ALMA/PG0050+124/CO21_combine/combine/line/PG0050_CO21-combine-line-10km-mosaic.fits"
flux_directory = "/home/qyfei/Desktop/Codes/CODES/Physical_values/clouds/spectral.txt"
outputdir = "/home/qyfei/Desktop/Codes/CODES/Physical_values/clouds/spectral/"
#para_As, para_ms, para_ss, para_fs = [], [], [], []

clouds_id = np.delete(np.arange(78), np.array([0,3,14,20,28,45,48,66]))

'''
for i in clouds_id:
    parameters = Gaussian_fit(path, file, i, cube_directory, flux_directory, outputdir)
    para_As.append(parameters[0][0])
    para_ms.append(parameters[1][0])
    para_ss.append(parameters[2][0])
    para_fs.append(parameters[3][0])
    #print(para_ss)
    np.savetxt(outputdir+"para_A.txt", np.array(para_As))
    np.savetxt(outputdir+"para_m.txt", np.array(para_ms))
    np.savetxt(outputdir+"para_s.txt", np.array(para_ss))
    np.savetxt(outputdir+"para_f.txt", np.array(para_fs))

'''

para_f = np.loadtxt(outputdir+"para_f.txt")
para_s = np.loadtxt(outputdir+"para_s.txt")
f, ef1, ef2 = para_f[:,0], para_f[:,1], para_f[:,2]
sigma, esigma1, esigma2 = para_s[:,0], para_s[:,1], para_s[:,2]

plt.errorbar(f, sigma, xerr=[-ef1, ef2], yerr = [-esigma1, esigma2], fmt='ko', mfc='none', mew=10, ms=2)
plt.loglog()
plt.show()