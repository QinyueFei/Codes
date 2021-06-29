# KMT star formation model with density floor

import numpy as np
import scipy.constants as physcons
from scipy.optimize import brentq

# Physical constants
G = physcons.G*1e3
kB = physcons.k/physcons.erg
c = physcons.c*100.
Msun = 1.99e33
pc = 3.09e18
yr=365.25*24.*3600.
mu=2.1e-24

# parameters defined by OML10
zetad=0.33
fwt=0.5
cw=8e5
alpha=5.
SigmaSFR0 = 2.5e-9*Msun/pc**2/yr

# paramters defined by KDM12
SigmaGMC=85*Msun/pc**2
sigmagal=8e5
epsff=0.01

# thermal pressure from OML model
def Pth(Sigmag, rhos, fH2a):
    RH2 = fH2a / (1-fH2a+1e-50)
    SigmaHI = (1.0-fH2a+1e-50) * Sigmag
    return np.pi*G*SigmaHI**2/(4*alpha) * \
        (1.0+2.0*RH2 +
         np.sqrt((1+2*RH2)**2 + 
                 32*zetad*alpha*fwt*cw**2*rhos/(np.pi*G*SigmaHI**2)))

# GMC free-fall time and SF timescale, following KDM12
def tff(Sigmag):
    return np.pi**0.25/8.0**0.5*sigmagal / \
        (G*(SigmaGMC**3*Sigmag)**0.25)
def tSF(Sigmag):
    return tff(Sigmag)/epsff

# KMT model for fH2
def chiKMT(Z):
    return 3.1*(1+3.1*Z**0.365)/4.1
def tauc(Sigmag, Z, cfac=5.0):
    return 0.066*(cfac*Sigmag/(Msun/pc**2))*Z
def sKMT(Sigmag, Z, cfac=5.0):
    return np.log(1+0.6*chiKMT(Z)+0.01*chiKMT(Z)**2) / \
        (0.6*tauc(Sigmag,Z,cfac))
def fH2KMT(Sigmag, Z, cfac=5.0):
    return np.maximum(1.0-0.75*sKMT(Sigmag, Z, cfac) / 
                      (1.0+0.25*sKMT(Sigmag, Z, cfac)), 1e-10)

# SFR from original KMT model
def SigmaSFRKMT(Sigmag, Z, cfac=5.0):
    return fH2KMT(Sigmag, Z, cfac)*Sigmag/tSF(Sigmag)

# minimum possible n_CNM
TCNMmax = 243.
def nCNMmin(Sigmag, rhos, fH2a):
    return Pth(Sigmag, rhos, fH2a)/(1.1*kB*TCNMmax)

# chi in modified KMT model, including self-consistent solution for fH2
fdiss=0.1
sigmadR=3.2e-5
e0star=7.5e-4
def chinew(Sigmag, rhos, G0p, Z, fH2a):
    return np.minimum(chiKMT(Z), 
                      fdiss*sigmadR*e0star*c*G0p / 
                      nCNMmin(Sigmag,rhos,fH2a))
def snew(Sigmag, rhos, G0p, Z, fH2a, cfac=5.0):
    return np.log(1+0.6*chinew(Sigmag, rhos, G0p, Z, fH2a) + 
                  0.01*chinew(Sigmag, rhos, G0p, Z, fH2a)**2) / \
        (0.6*tauc(Sigmag,Z,cfac))
def fH2new_resid(fH2a, Sigmag, rhos, G0p, Z, cfac=5.0):
    return fH2a - np.maximum(
        1.0-0.75*snew(Sigmag, rhos, G0p, Z, fH2a, cfac) / 
        (1.0+0.25*snew(Sigmag, rhos, G0p, Z, fH2a, cfac)), 0.0)
def fH2new(Sigmag, rhos, G0p, Z, cfac=5.0):
    if hasattr(Sigmag, '__iter__'):
        lenSg=len(Sigmag)
    else:
        lenSg=1
    if hasattr(rhos, '__iter__'):
        lenr=len(rhos)
    else:
        lenr=1
    if hasattr(Z, '__iter__'):
        lenZ=len(Z)
    else:
        lenZ=1
    if hasattr(G0p, '__iter__'):
        lenG=len(G0p)
    else:
        lenG=1
    if hasattr(cfac, '__iter__'):
        lenc=len(cfac)
    else:
        lenc=1
    lens = set([lenSg, lenr, lenZ, lenG, lenc])
    if len(lens) > 2:
        raise ValueError(
            "inputs must be of equal length or length 1")
    lenmax=max(lens)
    if lenmax == 1:
        return 1.0e-20+brentq(fH2new_resid, 0, 1, 
                              args=(Sigmag, rhos, G0p, Z, cfac))
    else:
        outvec=np.zeros(lenmax)
        for i in range(lenmax):
            if lenSg > 1:
                Sigmagtmp = Sigmag[i]
            else:
                Sigmagtmp = Sigmag
            if lenr > 1:
                rhostmp = rhos[i]
            else:
                rhostmp = rhos
            if lenZ > 1:
                Ztmp = Z[i]
            else:
                Ztmp = Z
            if lenG > 1:
                Gtmp = G0p[i]
            else:
                Gtmp = G0p
            if lenc > 1:
                cfactmp = cfac[i]
            else:
                cfactmp = cfac
            outvec[i] = 1.0e-20 + \
                brentq(fH2new_resid, 0, 1,
                       args=(Sigmagtmp, rhostmp, Gtmp, Ztmp, cfactmp))
        return outvec

# Solver for G0' and SFR in new KMT model
def SigmaSFRG0(Sigmag, rhos, G0p, Z, cfac=5.0):
    return fH2new(Sigmag, rhos, G0p, Z, cfac)*Sigmag/tSF(Sigmag)
def G0presid(logG0p, Sigmag, rhos, Z, cfac):
    return logG0p - \
        np.log10(SigmaSFRG0(Sigmag, rhos, 10.**logG0p, Z, cfac)/SigmaSFR0)
def G0psol(Sigmag, rhos, Z, cfac=5.0):
    if hasattr(Sigmag, '__iter__'):
        lenSg=len(Sigmag)
    else:
        lenSg=1
    if hasattr(rhos, '__iter__'):
        lenr=len(rhos)
    else:
        lenr=1
    if hasattr(Z, '__iter__'):
        lenZ=len(Z)
    else:
        lenZ=1
    if hasattr(cfac, '__iter__'):
        lenc=len(cfac)
    else:
        lenc=1
    lens = set([lenSg, lenr, lenZ, lenc])
    if len(lens) > 2:
        raise ValueError(
            "inputs must be of equal length or length 1")
    lenmax=max(lens)
    if lenmax == 1:
        return 10.**brentq(G0presid, -100, 100, 
                           args=(Sigmag, rhos, Z, cfac))
    else:
        outvec=np.zeros(lenmax)
        for i in range(lenmax):
            if lenSg > 1:
                Sigmagtmp = Sigmag[i]
            else:
                Sigmagtmp = Sigmag
            if lenr > 1:
                rhostmp = rhos[i]
            else:
                rhostmp = rhos
            if lenZ > 1:
                Ztmp = Z[i]
            else:
                Ztmp = Z
            if lenc > 1:
                cfactmp = cfac[i]
            else:
                cfactmp = cfac
            outvec[i] = 10.** \
                        brentq(G0presid, -100, 100,
                               args=(Sigmagtmp, rhostmp, Ztmp, cfactmp))
        return outvec

def SigmaSFRnew(Sigmag, rhos, Z, cfac=5.0):
    G0p = G0psol(Sigmag, rhos, Z, cfac)
    return SigmaSFRG0(Sigmag, rhos, G0p, Z, cfac)

def fH2KMTnew(Sigmag, rhos, Z, cfac=5.0):
    G0p = G0psol(Sigmag, rhos, Z, cfac)
    return fH2new(Sigmag, rhos, G0p, Z, cfac)


