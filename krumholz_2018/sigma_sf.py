"""
Functions to compute sigma_sf and Sigma_sf
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants and unit conversions (cgs)
from scipy.constants import G
G = 1e3*G
Msun = 1.9891e33
pc = 3.0856776e18
yr = 365.25*24.*3600.
Myr = 1e6*yr
Gyr = 1e9*yr
kmps = 1e5

def sigma_sf(sigma_th=2.5, fsf=0.5, epsff=0.015, fgP=0.5, fgQ=0.5,
             eta=1.5, phimp=1.4, phiQ=2.0, pmstar=3e3, Q=1.0,
             beta=0.0, tsfmax=2e3, torb=200., phint=0.0):
    """
    Returns the velocity dispersion sigma_sf that can be supported by
    star formation alone.

    Parameters:
       sigma_th : float or arraylike
          thermal velocity dispersion, in km/s
       fsf : float or arraylike
          fraction of the ISM in the star-forming phase
       epsff : float or arraylike
          star formation efficiency per free-fall time in star-forming
          phase
       fgP : float or arraylike
          fraction of the midplane pressure due to gas self-gravity
       fgQ : float or arraylike
          fractional contribution of gas to gravitational instability
       eta : float or arraylike
          turbulent dissipation rate parameter
       phimp : float or arraylike
          ratio of midplane total pressure to turbulent plus thermal
          pressure
       phiQ : float or arraylike
          ratio 1 + Q_gas / Q_*
       pmstar : float or arraylike
          momentum per unit mass of stars formed provided by star
          formation feedback
       Q : float or arraylike
          generalized Toomre Q parameter of disk
       beta : float or arraylike
          logarithmic index of galaxy rotation curve, beta = d ln v /
          d ln r
       tsfmax : float or arraylike
          maximum timescale for star formation, in Myr
       torb : float or arraylike
          galaxy orbital time, in Myr
       phint : float or arraylike
          if specified, phi_nt will be fixed to this value; if left as
          zero, phi_nt will be calculated self-consistently

    Returns:
       sigma_sf : float or arraylike
          velocity dispersion, in km/s
    """

    # Output is broadcast over inputs
    bcast = np.broadcast(np.asarray(sigma_th), np.asarray(fsf),
                         np.asarray(epsff), np.asarray(fgP),
                         np.asarray(fgQ), np.asarray(eta),
                         np.asarray(phimp), np.asarray(phiQ),
                         np.asarray(pmstar), np.asarray(Q),
                         np.asarray(beta), np.asarray(tsfmax),
                         np.asarray(torb), np.asarray(phint))
    sigma_sf = np.zeros(bcast.shape)
    i = 0
    for (sigma_th_, fsf_, epsff_, fgP_, fgQ_, eta_, phimp_,
         phiQ_, pmstar_, Q_, beta_, tsfmax_, torb_, phint_) in bcast:
        
        # Get prefactors
        fac1 = max(np.sqrt(3.0*fgP_/(8.0*(1.0+beta_))) * 
                   Q_*phimp_ / (4*fgQ_*epsff_) * torb_/tsfmax_, 1.0)
        fac = 4.0 * fac1 * fsf_ * epsff_ * pmstar_ / \
              (np.sqrt(3.0*fgP_) * np.pi * eta_ * phimp_ * phiQ_)

        # If phi_nt has been set manually, just use it
        if phint_ > 0.0:
            sigma_sf_val = fac / phint_**1.5
        else:
            # Problem is a cubic in sigma_sf^2
            coef = np.array([1.0, -(3.0*sigma_th_**2+fac**2),
                             3.0*sigma_th_**4, -sigma_th_**6])
            rts = np.roots(coef)
            sigma_sf2 = rts[np.logical_and(np.imag(rts) == 0.0,
                                           np.real(rts) > 0.0)]
            sigma_sf_val = np.real(np.sqrt(sigma_sf2))

        # Store results
        if sigma_sf.size > 0:
            sigma_sf.flat[i] = sigma_sf_val
        else:
            sigma_sf = sigma_sf_val[0]
        i = i+1

    return sigma_sf


def Sigma_sf(sigma_th=2.5, fsf=0.5, epsff=0.015, fgP=0.5, fgQ=0.5,
             eta=1.5, phimp=1.4, phiQ=2.0, pmstar=3e3, Q=1.0,
             beta=0.0, tsfmax=2e3, torb=200., phint=0.0,
             sigma_sf_fix=0.0):
    """
    Returns the gas surface density Sigma_sf below which the galaxy
    can be supported by star formation alone.

    Parameters:
       sigma_th : float or arraylike
          thermal velocity dispersion, in km/s
       fsf : float or arraylike
       epsff : float or arraylike
          star formation efficiency per free-fall time in star-forming
          phase
       fgP : float or arraylike
          fraction of the midplane pressure due to gas self-gravity
       fgQ : float or arraylike
          fractional contribution of gas to gravitational instability
       eta : float or arraylike
          turbulent dissipation rate parameter
       phimp : float or arraylike
          ratio of midplane total pressure to turbulent plus thermal
          pressure
       phiQ : float or arraylike
          ratio 1 + Q_gas / Q_*
       pmstar : float or arraylike
          momentum per unit mass of stars formed provided by star
          formation feedback
       Q : float or arraylike
          generalized Toomre Q parameter of disk
       beta : float or arraylike
          logarithmic index of galaxy rotation curve, beta = d ln v /
          d ln r
       tsfmax : float or arraylike
          maximum timescale for star formation, in Myr
       torb : float or arraylike
          galaxy orbital time, in Myr
       phint : float or arraylike
          if specified, phi_nt will be fixed to this value; if left as
          zero, phi_nt will be calculated self-consistently
       sigma_sf_fix : float or arraylike
          if set to a non-zero value, sigma_sf is fixed to this value,
          taken to be in units of km/s; otherwise if is computed
          automatically


    Returns:
       Sigma_sf : float or arraylike
          gas surface density, in Msun/pc^2
    """

    # Output is broadcast over inputs
    bcast = np.broadcast(np.asarray(sigma_th), np.asarray(fsf),
                         np.asarray(epsff), np.asarray(fgP),
                         np.asarray(fgQ), np.asarray(eta),
                         np.asarray(phimp), np.asarray(phiQ),
                         np.asarray(pmstar), np.asarray(Q),
                         np.asarray(beta), np.asarray(tsfmax),
                         np.asarray(torb), np.asarray(phint),
                         np.asarray(sigma_sf_fix))
    Sigma_sf = np.zeros(bcast.shape)
    i = 0
    for (sigma_th_, fsf_, epsff_, fgP_, fgQ_, eta_, phimp_,
         phiQ_, pmstar_, Q_, beta_, tsfmax_, torb_, phint_,
         sigma_sf_fix_) in bcast:
        
        # Get sigma_sf if necessary
        if sigma_sf_fix_ == 0.0:
            sigma_sf_ \
                = sigma_sf(sigma_th = sigma_th_, fsf = fsf_,
                           epsff = epsff_, fgP = fgP_, fgQ = fgQ_,
                           eta = eta_, phimp = phimp_,
                           phiQ = phiQ_, pmstar = pmstar_, Q = Q_,
                           beta = beta_, tsfmax = tsfmax_,
                           torb = torb_, phint = phint_)
        else:
            sigma_sf_ = sigma_sf_fix_

        # Store results
        Sigma_sf.flat[i] \
            = np.sqrt(2.0*(1.0+beta_)) * (2.0*np.pi/(torb*Myr)) * \
            sigma_sf_*kmps / (np.pi * G * (Q_/fgQ_)) / \
            (Msun/pc**2)
        i = i+1

    return Sigma_sf

    
