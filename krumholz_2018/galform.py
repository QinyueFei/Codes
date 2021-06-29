"""
Script to produce evolutionary tracks for galaxies.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import astropy.units as u
import astropy.constants as const
from scipy.optimize import brentq
from scipy.integrate import odeint
from sigma_sf import sigma_sf

################################################
# Cosmology function definitions
################################################

# Cosmology constants
Omega_Lam_0 = 0.73
Omega_m_0 = 1.0 - Omega_Lam_0
h = 0.71
H0 = 100.0*h*(u.km/u.s)/u.Mpc
fb = 0.17

def tU(z):
    """
    Function to return age of the universe tU as a function of z

    Parameters:
       z : float or array
          redshift

    Returns:
       tU : astropy.units.quantity.Quantity
          age of the universe
    """
    Omega_m = Omega_m_0 * (1.0+z)**3 / (Omega_Lam_0 + Omega_m_0 *
                                        (1.0+z)**3)
    H = H0 * np.sqrt(Omega_Lam_0 + Omega_m_0*(1.0+z)**3)
    tU = (2./3.) * (1.0/H) * np.arcsinh(
        np.sqrt((1.0-Omega_m)/Omega_m) / np.sqrt(1.0-Omega_m))
    return tU


def dtUdz(z):
    """
    Function to return dt/dz, where t = age of the universe

    Parameters:
       z : float or array
          redshift

    Returns:
       dtUdz : astropy.units.quantity.Quantity
          dt/dz
    """
    return -1.0 / (H0*(1.0+z) *
                   np.sqrt((1.0+z)**3*Omega_m_0+Omega_Lam_0))


def omega_dot(z):
    """
    Neistein & Dekel approximation to time derivative of EPS formalism
    time variable

    Parameters:
       z : float or array
          redshift

    Returns:
       omega_dot : float or array
          self-similar time variable derivative
    """
    return -0.0476 * (1.0 + z + 0.093 * (1.0+z)**-1.22)**2.5 / \
        u.Gyr


def Mdot_h(Mh, z):
    """
    Function that returns the halo dark matter accretion rate

    Parameters:
       Mh : float, array, or astropy.units.quantity.Quantity
          halo mass; if not an astropy Quantity, should have be in
          units of Msun
       z : float or array
          redshift

    Returns:
       Mdot_g : astropy.units.quantity.Quantity
          gas accretion rate
    """
    # Add units if needed
    if type(Mh) is not u.quantity.Quantity:
        Mh_ = Mh * u.Msun
    else:
        Mh_ = Mh

    # Compute fit
    return (1e12*u.Msun * (-0.628) * (Mh_/(1e12*u.Msun))**1.14 * \
            omega_dot(z)).to(u.Msun/u.yr)
    

def eps_in(Mh, z):
    """
    Function that computes the epsilon_in fit of Faucher-Giguere,
    Keres, & Ma (2011)

    Parameters:
       Mh : float, array, or astropy.units.quantity.Quantity
          halo mass; if not an astropy Quantity, should have be in
          units of Msun
       z : float or array
          redshift

    Returns:
       eps_in : float or array
          halo accretion efficiency
    """

    # Constant fit parameters
    eps_0 = 0.31
    beta_Mh = -0.25
    beta_z = 0.38
    eps_max = 1.0
    
    # Add units if needed
    if type(Mh) is not u.quantity.Quantity:
        Mh_ = Mh * u.Msun
    else:
        Mh_ = Mh

    # Compute result
    eps_in = np.minimum(eps_0 * (Mh_/(1e12*u.Msun))**beta_Mh *
                        (1.0+z)**beta_z, eps_max)

    # Convert from astropy quantity if that's not what we were given
    if type(Mh) is not u.quantity.Quantity and \
       type(z) is not u.quantity.Quantity:
        try:
            eps_in = float(eps_in)
        except:
            eps_in = np.array(eps_in)

    # Return
    return eps_in


def Mdot_g(Mh, z):
    """
    Function that returns the halo gas accretion rate

    Parameters:
       Mh : float, array, or astropy.units.quantity.Quantity
          halo mass; if not an astropy Quantity, should have be in
          units of Msun
       z : float or array
          redshift

    Returns:
       Mdot_g : astropy.units.quantity.Quantity
          gas accretion rate
    """

    # Compute fit
    return fb * eps_in(Mh, z) * Mdot_h(Mh, z).to(u.Msun/u.yr)


def vphi_max(Mh, z, c=10.0):
    """
    Function that returns the halo maximum circular velocity

    Parameters:
       Mh : float, array, or astropy.units.quantity.Quantity
          halo mass; if not an astropy Quantity, should have be in
          units of Msun
       z : float or array
          redshift
       c : float or array
          halo concentration index

    Returns:
       vphi_max : astropy.units.quantity.Quantity
          maximum circular velocity
    """
    
    # Add units if needed
    if type(Mh) is not u.quantity.Quantity:
        Mh_ = Mh * u.Msun
    else:
        Mh_ = Mh

    # Compute fit
    v_vir = 117.*u.km/u.s * (Mh_/(1e12*u.Msun))**(1./3.) * \
            np.sqrt(1+z)
    vphi_max = 0.465 * np.sqrt(c/(np.log(1+c)-c/(1+c))) * v_vir

    # Return
    return vphi_max


def r_vir(Mh, z):
    """
    Returns the virial radius versus halo mass and redshift

    Parameters:
       Mh : float, array, or astropy.units.quantity.Quantity
          halo mass; if not an astropy Quantity, should have be in
          units of Msun
       z : float or array
          redshift

    Returns:
       r_vir : astropy.units.quantity.Quantity
          virial radius
    """

    # Add units if needed
    if type(Mh) is not u.quantity.Quantity:
        Mh_ = Mh * u.Msun
    else:
        Mh_ = Mh

    # Compute fit
    r_v = 163*u.kpc * (Mh_/(1e12*u.Msun))**(1./3.) / (1.0+z)

    # Return
    return r_v


def r_disk(Mh, z, spin_param=0.07):
    """
    Returns the disk radius versus halo mass and redshift

    Parameters:
       Mh : float, array, or astropy.units.quantity.Quantity
          halo mass; if not an astropy Quantity, should have be in
          units of Msun
       z : float or array
          redshift
       spin_param : float or array
          spin parameter

    Returns:
       r_disk : astropy.units.quantity.Quantity
          disk radius
    """
    return 0.05 * (spin_param/0.1) * r_vir(Mh, z)


def _halo_hist_dMdz(Mh12, z):
    deriv = (Mdot_h(Mh12*1e12, z) * dtUdz(z)).to(u.Msun).value / 1e12
    return deriv

def halo_hist(Mh0, z=None):
    """
    Generates the halo mass versus redshift for a halo of mass Mh0 at
    z = 0

    Parameters:
       Mh0 : float, array, or astropy.units.quantity.Quantity
          halo mass at z = 0; if not an astropy Quantity, should have
          be in units of Msun
       z : arraylike
          grid of redshifts at which to compute halo properties; if
          left as None, a grid is created automatically

    Returns:
       z : array
          array of output redshifts
       Mh : astropy.units.quantity.Quantity
          halo mass at each redshift; if given more than one Mh, this
          will be a 2D array, otherwise it will be a 1D array
    """

    # Get halo mass in units of 10^12 Msun, which we will use as our
    # integration variable
    if type(Mh0) is u.quantity.Quantity:
        Mh12 = Mh0.to(u.Msun).value / 1e12
    else:
        Mh12 = np.array(Mh0)/1e12

    # Make grid in z if not provided
    if z is None:
        z = np.linspace(0, 3)

    # Integrate to get Mh12 vs time, then return
    if np.size(Mh12) == 1:
        Mh12_z = np.array(
            odeint(_halo_hist_dMdz, np.asarray(Mh12), z)).flatten()
        return z, Mh12_z * 1e12*u.Msun
    else:
        Mh12_z = []
        for Mh12_ in Mh12:
            Mh12_z.append(np.array(
                odeint(_halo_hist_dMdz, Mh12_, z)).flatten())
        Mh12_z = np.array(Mh12_z) * 1e12*u.Msun
        return z, Mh12_z
    

################################################
# Functions describing the present theory
################################################

def _sigma_mdot_helper(sigma_g, fac, mdot, sigma_sf, sigma_th):
    phint = 1.0 - (sigma_th/sigma_g)**2
    phisf = 1.0 - sigma_sf / sigma_g
    ret = fac * phint**1.5 * phisf * sigma_g**3 - mdot
    return ret


def sigma_mdot(mdot, sigma_th=2.5, fsf=0.5, epsff=0.015, fgP=0.5,
               fgQ=0.5, eta=1.5, phimp=1.4, phiQ=2.0, pmstar=3e3,
               Qmin=1.0, beta=0.0, tsfmax=2e3, torb=200.,
               phint=0.0, sigmasf=0.0, sigmasf_return=False):
    """
    Function to compute the gas velocity dispersion for a given infall
    rate

    Parameters:
       Mdot : float, array, or astropy.units.quantity.Quantity
          mass accretion rate; if not an astropy quantity, units must
          be Msun/yr
       sigma_th : float, array, or astropy.units.quantity.Quantity
          thermal velocity dispersion; if not an astropy quantity,
          units must be km/s
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
       pmstar : float, array, or astropy.units.quantity.Quantity
          momentum per unit mass of stars formed provided by star
          formation feedback; if not an astropy quantity, units must
          be km/s
       Q : float or arraylike
          generalized Toomre Q parameter of disk
       beta : float or arraylike
          logarithmic index of galaxy rotation curve, beta = d ln v /
          d ln r
       tsfmax : float, array, or astropy.units.quantity.Quantity
          maximum timescale for star formation; if not an astropy
          quantity, units must be Myr
       torb : float, array, or astropy.units.quantity.Quantity
          galaxy orbital time; if not an astropy quantity, units must
          be Myr
       phint : float or arraylike
          if specified, phi_nt will be fixed to this value; if left as
          zero, phi_nt will be calculated self-consistently
       sigmasf : float, array, or astropy.units.quantity.Quantity
          if specified, this gives the value of sigma_sf; if
          left as zero, sigma_sf is calculated self-consistently; if
          not an astropy quantity, units must be km/s
       sigmasf_return : bool
          if True, the value of sigmasf is returned

    Returns:
       sigma_g : astropy.units.quantity.Quantity
          velocity dispersion
       sigma_sf : astropy.units.quantity.Quantity
          value of sigma_sf (only returned if sigmasf_return is True)
    """

    # Adopt a unit system where velocities are in km/s, mass
    # accretion rates in Msun/yr; necessary because the various
    # numpy routines we need to invoke here do not play nicely
    # with units
    if type(sigma_th) is u.quantity.Quantity:
        sigma_th_kmps = sigma_th.to(u.km/u.s).value
    else:
        sigma_th_kmps = sigma_th
    if type(torb) is u.quantity.Quantity:
        torb_Myr = torb.to(u.Myr).value
    else:
        torb_Myr = torb
    if type(tsfmax) is u.quantity.Quantity:
        tsfmax_Myr = tsfmax.to(u.Myr).value
    else:
        tsfmax_Myr = tsfmax
    if type(sigmasf) is u.quantity.Quantity:
        sigmasf_kmps = sigmasf.to(u.km/u.s).value
    else:
        sigmasf_kmps = sigmasf
    if type(mdot) is u.quantity.Quantity:
        mdot_msun_yr = mdot.to(u.Msun/u.yr).value
    else:
        mdot_msun_yr = mdot
    if type(pmstar) is u.quantity.Quantity:
        pmstar_kmps = pmstar.to(u.km/u.s).value
    else:
        pmstar_kmps = pmstar

    # Broadcast over inputs
    bcast = np.broadcast(np.asarray(mdot_msun_yr),
                         np.asarray(sigma_th_kmps),
                         np.asarray(fsf), np.asarray(epsff),
                         np.asarray(fgP), np.asarray(fgQ),
                         np.asarray(eta), np.asarray(phimp),
                         np.asarray(phiQ), np.asarray(pmstar_kmps),
                         np.asarray(Qmin), np.asarray(beta),
                         np.asarray(tsfmax_Myr),
                         np.asarray(torb_Myr),
                         np.asarray(phint),
                         np.asarray(sigmasf_kmps))
    sigma_g = np.zeros(bcast.shape)
    sigma_sf_ret = np.zeros(bcast.shape)
    i = 0
    for (mdot_, sigma_th_, fsf_, epsff_, fgP_, fgQ_, eta_,
         phimp_, phiQ_, pmstar_, Qmin_, beta_, tsfmax_,
         torb_, phint_, sigmasf_) in bcast:

        # Get sigma_sf if needed
        if sigmasf_ == 0.0:
            ssf_ = sigma_sf(sigma_th = sigma_th_, fsf = fsf_,
                            epsff = epsff_, fgP = fgP_, fgQ = fgQ_,
                            eta = eta_, phimp = phimp_,
                            phiQ = phiQ_, pmstar = pmstar_, Q = Qmin_,
                            beta = beta_, tsfmax = tsfmax_,
                            torb = torb_, phint = phint_)
        else:
            ssf_ = sigmasf_

        # Get prefactor, assuming phi_nt = 1
        fac = (4.0 * (1.0+beta_) * eta_ * phiQ_ * fgQ_**2 / \
               ((1.0-beta_) * Qmin_**2 * const.G)).\
               to(u.Msun/u.yr/(u.km/u.s)**3).value
        
        # Solve for fixed phi_nt; in this case the probem is a cubic
        # and can be solved analytically. If phi_nt was specified,
        # this is the final answer, and, if not, it gives us a useful
        # limit we can use to apply Brent's method
        if phint_ > 0.0:
            phint_tmp = phint_
        else:
            phint_tmp = 1.0 - (sigma_th_/ssf_)**2
        coef = np.array([
            fac * phint_tmp**1.5, -fac * phint_tmp**1.5 * ssf_,
            0.0, -mdot_])
        rts = np.roots(coef)
        sigma_g_lim = np.real(
            rts[np.logical_and(np.imag(rts) == 0.0,
                               np.real(rts) > 0.0)])[0]
        
        # If phi_nt was fixed, we're done with this loop; otherwise
        # now use Brent's method to get the exact value
        if phint_ > 0.0:
            sigma_g_val = sigma_g_lim
        else:
            sigma_g_val = brentq(_sigma_mdot_helper, ssf_,
                                 sigma_g_lim,
                                 args=(fac, mdot_, ssf_,
                                       sigma_th_))

        # Store and iterate
        if sigma_g.size > 0:
            sigma_g.flat[i] = sigma_g_val
            sigma_sf_ret.flat[i] = ssf_
        else:
            sigma_g = sigma_g_val
            sigma_sf_ret = ssf_
        i = i+1

    # Done
    if sigmasf_return:
        return sigma_g*u.km/u.s, sigma_sf_ret*u.km/u.s
    else:
        return sigma_g*u.km/u.s

    
def sfr_sigma(sigma_g, vphi, phia=2.0, fsf=0.5, beta=0.0, Q=1.0,
              fgQ=0.5, fgP=0.5, epsff=0.015, phimp=1.4, tsfmax=2e3,
              torb=200., Z=0.5, rhomin_fac=2.0, fc=5.0):
    """
    Returns star formation rate for a given velocity dispersion and
    rotation curve speed.

    Parameters:
       sigma_g : float, array, or astropy.units.quantity.Quantity
          velocity dispersion; if not an astropy quantity, units must
          be km/s
       vphi : float, array, or astropy.units.quantity.Quantity
          outer radius rotation speed; if not an astropy quantity,
          units must be km/s
       phia : float or arraylike
          offset between local and galaxy-averaged SF laws
       fsf : float or arraylike
          fraction of ISM in star-forming phase
       beta : float or arraylike
          rotation curve index
       Q : float or arraylike
          Toomre parameter for gas plus stars
       fgQ : float or arraylike
          fractional gas contribution to Q
       fgP : float or arraylike
          fractional gas contribution to midplane pressure
       epsff : float or arraylike
          dimensionless star formation rate per freefall time
       phimp : float or arraylike
          ratio of midplane total pressure to turbulent plus thermal
          pressure
       tsfmax : float, array, or astropy.units.quantity.Quantity
          maximum timescale for star formation; if not an astropy
          quantity, units must be Myr
       torb : float, array, or astropy.units.quantity.Quantity
          galaxy orbital time; if not an astropy quantity, units must
          be Myr
       Z : float or arraylike
          metallicity normalised to Solar; only used if fsf is zero
       rhomin_fac : float or arraylike
          ratio of rho_* to rho_min for KMT+ theory; only used if fsf
          is zero
       fc : float or arraylike
          clumping factor for KMT+ theory; only used if fsf is zero

    Returns:
       sfr : astropy.units.quantity.Quantity
          star formation rate, in Msun/yr
    """

    # Deal with units
    if type(sigma_g) is u.quantity.Quantity:
        sigma_g_ = sigma_g
    else:
        sigma_g_ = sigma_g * u.km/u.s
    if type(vphi) is u.quantity.Quantity:
        vphi_ = vphi
    else:
        vphi_ = vphi * u.km/u.s
    if type(torb) is u.quantity.Quantity:
        torb_ = torb
    else:
        torb_ = torb * u.Myr
    if type(tsfmax) is u.quantity.Quantity:
        tsfmax_ = tsfmax
    else:
        tsfmax_ = tsfmax * u.Myr

    # Get mdot
    mdot = np.sqrt(2.0/(1.0+beta)) * phia * fsf * fgQ * vphi_**2 * \
           sigma_g_ / (np.pi*const.G*Q) * \
           np.maximum(np.sqrt(2.0*(1+beta)/(3.0*fgP*phimp)) *
                      8.0*epsff*fgQ/Q, torb_/tsfmax_)

    # Return, converted to Msun/yr
    return mdot.to(u.Msun/u.yr)
    

################################################
# Generate example histories
################################################

# Constants
mdot_fac = 1.0
phi_v = 1.4
fsf = 1.0
sigma_th = 0.2

# Generate sample halo mass histories
nhist = 11
Mh0 = np.logspace(12, 13, nhist)
z = np.linspace(0, 3, 400)
z, Mh = halo_hist(Mh0, z=z)

# Generate gas accretion rates
Mdot_in = Mdot_g(Mh, z)

# Generate disk radii, circular velocities, and orbital times
r_d = r_disk(Mh, z)
vphi = vphi_max(Mh, z)*phi_v
torb = 2.0 * np.pi * r_d / vphi

# Generate velocity dispersions
sigma_g, sigmasf \
    = sigma_mdot(Mdot_in*mdot_fac, sigma_th=sigma_th,
                 fsf=fsf, torb=torb,
                 sigmasf_return=True)

# Compute star formation rates
sfr = sfr_sigma(sigma_g, vphi, fsf=fsf, torb=torb)
    

################################################
# Make plot
################################################

fig = plt.figure(1, figsize=(4,4))
plt.clf()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
try:
    # Depends on matplotlib version
    plt.rc('xtick', direction='in', top=True)
    plt.rc('ytick', direction='in', right=True)
except:
    pass

# Mdot_in / disc SFR
p = []
lab = []
ax = fig.add_subplot(2,1,1)
for i in range(nhist):
    p1,=plt.plot(z, np.log10(Mdot_in[i].to(u.Msun/u.yr)/
                             sfr[i].to(u.Msun/u.yr)),
                 lw=2,
                 color = cmap.Greens(0.3 + 0.7*float(i)/(nhist-1)))
    if i == 0 or i == nhist-1:
        p.append(p1)
        lab.append('$M_{{\mathrm{{h,0}}}} = 10^{{{:d}}}\,M_\odot$'.
                   format(int(round(np.log10(Mh0[i])))))
plt.fill_between(z, np.ones(z.shape)*np.log10(2)/2,
                 np.ones(z.shape)*0.5,
                 color='k', alpha=0.25, lw=0)
plt.fill_between(z, np.ones(z.shape)*np.log10(0.5)/2,
                 np.ones(z.shape)*np.log10(2)/2,
                 color='b', alpha=0.25, lw=0)
plt.fill_between(z, np.ones(z.shape)*(-0.5),
                 np.ones(z.shape)*np.log10(0.5)/2,
                 color='r', alpha=0.25, lw=0)
plt.xlim([0,3])
plt.ylim([-0.5,0.5])
ax.set_xticklabels([])
plt.legend(p, lab, loc='lower right', prop={'size':8})
plt.ylabel(
    r'$\log\, \dot{M}_{\mathrm{g,acc}} / \dot{M}_{\mathrm{*,disc}}$')
plt.text(0.4, 0.35, 'Bulge building', color='k',
         size=8, va='center')
plt.text(0.4, 0.0, 'Disc building', color='b',
         size=8, va='center')
plt.text(0.45, -0.35, 'Central quenching', color='r',
         size=8, va='center')

# sigma_g / sigma_sf
ax = fig.add_subplot(2,1,2)
for i in range(nhist):
    plt.plot(z,
             np.log10(sigma_g[i].to(u.km/u.s) /
                      sigmasf[i].to(u.km/u.s)), lw=2,
             color=cmap.Greens(0.3 + 0.7*float(i)/(nhist-1)))
plt.fill_between(z, np.ones(z.shape)*np.log10(2),
                 np.ones(z.shape)*0.7,
                 color='b', alpha=0.25, lw=0)
plt.fill_between(z, 0, np.ones(z.shape)*np.log10(2),
                 color='r', alpha=0.25, lw=0)
plt.text(0.15, 0.625, 'Transport driving',
         color='b', size=8)
plt.text(0.7, 0.15, 'Feedback driving',
         color='r', size=8)
plt.xlim([0,3])
plt.ylim([0,0.7])
plt.yticks([0, 0.2, 0.4, 0.6])
plt.xlabel(r'$z$')
plt.ylabel(r'$\log\,\sigma_{\mathrm{g}} / \sigma_{\mathrm{sf}}$')

# Adjust spacing
plt.subplots_adjust(wspace=0, hspace=0, left=0.2, bottom=0.13, top=0.95)

# Save
plt.savefig('halo_hist.pdf')
