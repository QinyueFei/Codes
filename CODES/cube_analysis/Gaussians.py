# %%
## 
import numpy as np
from astropy.modeling import models
from astropy.stats import sigma_clipped_stats
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
import emcee
from multiprocessing import Pool

# %%
## Single Gaussian model
def Gauss(x_, a_, m_, s_):
    x, a, m, s = x_, a_, m_, s_
    gauss = models.Gaussian1D(a, m, s)
    return gauss(x)
def log_likelihood(theta, x, y, yerr):
    a, m, s = theta
    model = Gauss(x, a, m, s)
    sigma2 = yerr ** 2# + model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2*np.pi*sigma2))
def log_prior(para):
    a, m, s = para
    if 0<a<80 and 7000<m<11000 and 0<=s<500:
        return 0.0
    return -np.inf
def log_probability(para, x, y, yerr):
    lp = log_prior(para)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(para, x, y, yerr)

# def Single_Gaussian_fit(x_, y_):
#     x, y = x_, y_
#     yerr = sigma_clipped_stats(y, sigma=3)[-1]
#     nvctr = np.where(y == np.nanmax(y))
#     np.random.seed(42)
#     nll = lambda *args: -log_likelihood(*args)
#     a_g, m_g, s_g = y[nvctr], x[nvctr], 100
#     initial = np.array([a_g, m_g, s_g]) + 0.1 * np.random.randn(3)
#     soln = minimize(nll, initial, args=(x, y, yerr), method="Nelder-Mead")
#     a_ml, m_ml, s_ml = soln.x
#     BIC = 3*np.log(len(x)) - 2*log_likelihood(soln.x, x, y, yerr)
#     return a_ml, m_ml, s_ml, BIC

def Single_Gaussian_fit(x_, y_, yerr_):
    x, y, yerr = x_, y_, yerr_
    nvctr = np.argsort(y)[-1]
    para_i = np.array([y[nvctr], x[nvctr], 30], dtype='float')

    popt, pcov = curve_fit(Gauss, x, y, p0=para_i, method='lm')
    BIC = 3*np.log(len(x)) - 2*log_likelihood(popt, x, y, yerr)
    return popt, BIC

def Single_Gaussian_mcmc(x_, y_, yerr_):
    x, y, yerr = x_, y_, yerr_
    # a_ml, m_ml, s_ml = popt
    # pos = np.array([a_ml, m_ml, s_ml]) + 1e-4 * np.random.randn(32, 3)
    pos = np.array([np.nanmax(y), 8500, 60]) + 1e-4 * np.random.randn(32, 3)
    nwalkers, ndim = pos.shape
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr), pool=pool)
        sampler.run_mcmc(pos,1000, progress=True)
    samples = sampler.get_chain()
    flat_samples = samples[500:].reshape(32*500, 3)
    paras = np.zeros(3)
    for i in range(3):
        paras[i] = np.percentile(flat_samples[:,i], [50])
    BIC = 3*np.log(len(x)) - 2*log_likelihood(paras, x, y, yerr)
    # flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    return paras, BIC


# para1, BIC1 = Single_Gaussian_fit(velo, mask_spectrum, rms_spectrum)
# para, BIC1_mcmc = Single_Gaussian_mcmc(velo, mask_spectrum, rms_spectrum)
# plt.figure(figsize=(16, 6))
# plt.step(velo, mask_spectrum)
# plt.plot(velo, Gauss(velo, para1[0], para1[1], para1[2]))
# plt.plot(velo, Gauss(velo, para[0], para[1], para[2]))
# print(BIC1, BIC1_mcmc)

# %%
## Double Gaussian model
def log_likelihood_double(theta, x, y, yerr):
    a0, m0, s0, a1, m1, s1 = theta
    model = Gauss(x, a0, m0, s0) + Gauss(x, a1, m1, s1)
    sigma2 = yerr ** 2# + model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2*np.pi*sigma2))
def log_prior_double(para):
    a0, m0, s0, a1, m1, s1 = para
    if s0<500 and s1<500:
        return 0.0
    return -np.inf
def log_probability_double(para, x, y, yerr):
    lp = log_prior_double(para)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_double(para, x, y, yerr)

# def Double_Gaussian_fit(x_, y_):
#     x, y = x_, y_
#     yerr = sigma_clipped_stats(y, sigma=3)[-1]
#     nvctr = np.where(y == np.nanmax(y))
#     np.random.seed(42)
#     nll = lambda *args: -log_likelihood_double(*args)
#     a0, m0, s0, a1, m1, s1 = y[nvctr], x[nvctr], 30, y[nvctr]/2, x[nvctr]+40, 20
#     initial = np.array([a0, m0, s0, a1, m1, s1]) + 0.1 * np.random.randn(6)
#     soln = minimize(nll, initial, args=(x, y, yerr), method="Nelder-Mead")
#     a0_ml, m0_ml, s0_ml, a1_ml, m1_ml, s1_ml = soln.x
#     BIC = 6*np.log(len(x)) - 2*log_likelihood_double(soln.x, x, y, yerr)
#     return a0_ml, m0_ml, s0_ml, a1_ml, m1_ml, s1_ml, BIC

def Double_Gauss(x, a0, m0, s0, a1, m1, s1):
    model = Gauss(x, a0, m0, s0) + Gauss(x, a1, m1, s1)
    return model

def Double_Gaussian_fit(x_, y_, yerr_):
    x, y, yerr = x_, y_, yerr_
    # coordinates = peak_local_max(y, min_distance=5)
    # n0, n1 = coordinates[0][0], coordinates[1][0]
    n0, n1 = np.argsort(y)[-1], np.argsort(y)[-2]
    parai = np.array([y[n0], x[n0], 30, y[n1], x[n1], 20])
    popt, pcov = curve_fit(Double_Gauss, x, y, p0=parai, method='lm')
    BIC = 6*np.log(len(x)) - 2*log_likelihood_double(popt, x, y, yerr)
    return popt, BIC

def Double_Gaussian_mcmc(x_, y_, yerr_, popt):
    x, y, yerr = x_, y_, yerr_
    a0_ml, m0_ml, s0_ml, a1_ml, m1_ml, s1_ml = abs(popt)
    pos = np.array([a0_ml, m0_ml, s0_ml, a1_ml, m1_ml, s1_ml]) + 1e-4 * np.random.randn(64, 6)
    # pos = np.array([np.nanmax(y), (x[0]+x[-1])/2, 60, np.nanmax(y)/2, (x[0]+x[-1])/2+100, 30]) + 1e-4 * np.random.randn(64, 6)
    nwalkers, ndim = pos.shape
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_double, args=(x, y, yerr), pool=pool)
        sampler.run_mcmc(pos,1000, progress=True)
    samples = sampler.get_chain()
    flat_samples = samples[500:].reshape(64*500, 6)
    paras = np.zeros(6)
    for i in range(6):
        paras[i] = np.percentile(flat_samples[:,i], [50])
    BIC = 6*np.log(len(x)) - 2*log_likelihood_double(paras, x, y, yerr)
    # flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    return samples, paras, BIC

# para1, BIC1 = Double_Gaussian_fit(velo, mask_spectrum, rms_spectrum)
# samples, para, BIC1_mcmc = Double_Gaussian_mcmc(velo, mask_spectrum, rms_spectrum, para1)
# plt.figure(figsize=(16, 6))
# plt.step(velo, mask_spectrum)
# plt.plot(velo, Double_Gauss(velo, para1[0], para1[1], para1[2], para1[3], para1[4], para1[5]))
# plt.plot(velo, Double_Gauss(velo, para[0], para[1], para[2], para[3], para[4], para[5]))
# print(BIC1, BIC1_mcmc)

# %%
## Triple Gaussian model
def log_likelihood_triple(theta, x, y, yerr):
    a0, m0, s0, a1, m1, s1, a2, m2, s2 = theta
    model = Gauss(x, a0, m0, s0) + Gauss(x, a1, m1, s1) + Gauss(x, a2, m2, s2)
    sigma2 = yerr ** 2# + model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2*np.pi*sigma2))
def log_prior_triple(para):
    a0, m0, s0, a1, m1, s1, a2, m2, s2 = para
    if 0<a0 and s0<500 and 0<a1 and s1<500 and 0<a2 and s2<500:
        return 0.0
    return -np.inf
def log_probability_triple(para, x, y, yerr):
    lp = log_prior_triple(para)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_triple(para, x, y, yerr)

# def Triple_Gaussian_fit(x_, y_):
#     x, y = x_, y_
#     yerr = sigma_clipped_stats(y, sigma=3)[-1]
#     nvctr = np.where(y == np.nanmax(y))
#     np.random.seed(42)
#     nll = lambda *args: -log_likelihood_triple(*args)
#     a0, m0, s0, a1, m1, s1, a2, m2, s2 = y[nvctr], x[nvctr], 30, y[nvctr]/2, x[nvctr]+40, 20, y[nvctr]/3, x[nvctr]+60, 30
#     initial = np.array([a0, m0, s0, a1, m1, s1, a2, m2, s2]) + 0.1 * np.random.randn(9)
#     soln = minimize(nll, initial, args=(x, y, yerr), method="Nelder-Mead")
#     a0_ml, m0_ml, s0_ml, a1_ml, m1_ml, s1_ml, a2_ml, m2_ml, s2_ml = soln.x
#     BIC = 9*np.log(len(x)) - 2*log_likelihood_triple(soln.x, x, y, yerr)
#     return a0_ml, m0_ml, s0_ml, a1_ml, m1_ml, s1_ml, a2_ml, m2_ml, s2_ml, BIC

def Triple_Gauss(x, a0, m0, s0, a1, m1, s1, a2, m2, s2):
    model = Gauss(x, a0, m0, s0) + Gauss(x, a1, m1, s1) + Gauss(x, a2, m2, s2)
    return model

def Triple_Gaussian_fit(x_, y_, yerr_):
    x, y, yerr = x_, y_, yerr_
    # coordinates = peak_local_max(y, min_distance=3)
    # n0, n1, n2 = coordinates[0][0], coordinates[1][0], coordinates[2][0]
    sort = np.argsort(y)
    n0, n1, n2 = sort[-1], sort[-2], sort[-3]
    print(n0, n1, n2)
    parai = np.array([y[n0], x[n0], 60, y[n1], x[n1], 30, y[n2], x[n2], 15])
    popt, pcov = curve_fit(Triple_Gauss, x, y, p0=parai, method='lm')
    BIC = 9*np.log(len(x)) - 2*log_likelihood_triple(popt, x, y, yerr)
    return popt, BIC

def Triple_Gaussian_mcmc(x_, y_, yerr_, popt):
    x, y, yerr = x_, y_, yerr_
    a0_ml, m0_ml, s0_ml, a1_ml, m1_ml, s1_ml, a2_ml, m2_ml, s2_ml = abs(popt)
    pos = np.array([a0_ml, m0_ml, s0_ml, a1_ml, m1_ml, s1_ml, a2_ml, m2_ml, s2_ml]) + 1e-4 * np.random.randn(96, 9)
    nwalkers, ndim = pos.shape
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_triple, args=(x, y, yerr), pool=pool)
        sampler.run_mcmc(pos,1000, progress=True)
    samples = sampler.get_chain()
    flat_samples = samples[500:].reshape(96*500, 9)
    paras = np.zeros(9)
    for i in range(9):
        paras[i] = np.percentile(flat_samples[:,i], [50])
    BIC = 9*np.log(len(x)) - 2*log_likelihood_triple(paras, x, y, yerr)
    # flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    return samples, paras, BIC

# para1, BIC1 = Triple_Gaussian_fit(velo, mask_spectrum, rms_spectrum)
# para, BIC1_mcmc = Triple_Gaussian_mcmc(velo, mask_spectrum, rms_spectrum, para1)
# plt.figure(figsize=(16, 6))
# plt.step(velo, mask_spectrum)
# plt.plot(velo, Triple_Gauss(velo, para1[0], para1[1], para1[2], para1[3], para1[4], para1[5], para1[6], para1[7], para1[8]))
# plt.plot(velo, Triple_Gauss(velo, para[0], para[1], para[2], para[3], para[4], para[5], para[6], para[7], para[8]))
# print(BIC1, BIC1_mcmc)
