"""
This script generates plots of the star formation law from Krumholz+ 2017
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import matplotlib.colors as clrs
import matplotlib.colorbar as cbar
import astropy.io.ascii as asciitable
from astropy.units import pc, yr, kpc, Myr, Gyr, Msun, km, s, g, cm
from astropy.constants import G
import os.path as osp
import collections
from kmtnew import fH2KMTnew
import ma

# Choice of CO alpha factor scalings NORMALIZED TO DADDI'S CONVENTIONS
alphasb = 1.0      # starburst value
alphahiz = 1.0     # high-z disk
alphaz0 = 1.0      # z=0 scaling
alphathings = 4.6/4.4 # THINGS data value

# Location of data
datadir = osp.join('data', 'sfr')

# Data from Genzel+, MNRAS, 2010, 407, 2091
sigmag_genzel=[]
sigmasfr_genzel=[]
sigmagtorb_genzel=[]
sb_genzel=[]
fp=open(osp.join(datadir, 'genzel_ks.dat'), 'r')
for line in fp:
    if line[0]=='#':
        continue
    spl=line.split()
    if spl[0]=='Name':
        continue
    sigmag_genzel.append(10.0**float(spl[11]))
    sigmagtorb_genzel.append(10.0**float(spl[12])/(2.0*np.pi))
    sigmasfr_genzel.append(10.0**float(spl[13]))
    # Apply CO scalings
    if spl[0][0:3]=='SMM':
        # sub-mm galaxy, so use starburst scalings; Genzel uses alpha = 1.0
        # for starbursts, and Daddi's convention is 0.8
        sb_genzel.append(True)
        sigmag_genzel[-1] = sigmag_genzel[-1]*0.8/1.0*alphasb
        sigmagtorb_genzel[-1] = sigmagtorb_genzel[-1]*0.8/1.0*alphasb
    else:
        # non-sub-mm galaxy, so use normal high z scalings; Genzel uses
        # alpha = 3.2, Daddi uses 3.6
        sb_genzel.append(False)
        sigmag_genzel[-1] = sigmag_genzel[-1]*3.6/3.2*alphahiz
        sigmagtorb_genzel[-1] = sigmagtorb_genzel[-1]*3.6/3.2*alphahiz
fp.close()
sigmag_genzel=np.array(sigmag_genzel)*Msun/pc**2
sigmasfr_genzel=np.array(sigmasfr_genzel)*Msun/pc**2/Myr
sigmagtorb_genzel=np.array(sigmagtorb_genzel)*Msun/pc**2/Myr
torb_genzel=sigmag_genzel/sigmagtorb_genzel

# Data from Bouche+ 2007, ApJ, 671, 303
sigmag_bouche=[]
sigmasfr_bouche=[]
sigmagtorb_bouche=[]
fp=open(osp.join(datadir, 'KS_2_Bouche.dat'), 'r')
for line in fp:
    if line[0]=='#':
        continue
    spl=line.split()
    sigmag_bouche.append(10.0**float(spl[0])*alphasb)
    sigmagtorb_bouche.append(10.0**float(spl[2])/(2.0*np.pi) * alphasb)
    sigmasfr_bouche.append(10.0**float(spl[1]))
fp.close()
sigmag_bouche=np.array(sigmag_bouche)*Msun/pc**2
sigmasfr_bouche=np.array(sigmasfr_bouche)*Msun/pc**2/Myr
sigmagtorb_bouche=np.array(sigmagtorb_bouche)*Msun/pc**2/Myr
torb_bouche=sigmag_bouche/sigmagtorb_bouche

# Data from Daddi+ 2008, ApJL, 673, L21 and 2010, ApJ, 713, 686; z =
# 0.5 sample
sigmag_daddiz05=[]
sigmasfr_daddiz05=[]
sigmagtorb_daddiz05=[]
fp=open(osp.join(datadir, 'KS_2_Daddiz05.dat'), 'r')
for line in fp:
    if line[0]=='#':
        continue
    if line[0]=='\n':
        continue
    spl=line.split()
    sigmag_daddiz05.append(10.0**float(spl[1])*alphahiz)
    sigmagtorb_daddiz05.append(10.0**float(spl[2])*alphahiz)
    sigmasfr_daddiz05.append(10.0**float(spl[0]))
fp.close()
sigmag_daddiz05=np.array(sigmag_daddiz05)*Msun/pc**2
sigmasfr_daddiz05=np.array(sigmasfr_daddiz05)*Msun/pc**2/Myr
sigmagtorb_daddiz05=np.array(sigmagtorb_daddiz05)*Msun/pc**2/Myr
torb_daddiz05=sigmag_daddiz05/sigmagtorb_daddiz05

# Data from Daddi+ 2008, ApJL, 673, L21 and 2010, ApJ, 713, 686; z =
# 2 sample
sigmag_daddiz2=[]
sigmasfr_daddiz2=[]
sigmagtorb_daddiz2=[]
fp=open(osp.join(datadir, 'KS_2_Daddi.dat'), 'r')
for line in fp:
    if line[0]=='#':
        continue
    if line[0]=='\n':
        continue
    spl=line.split()
    sigmag_daddiz2.append(float(spl[0])*alphahiz)
    sigmagtorb_daddiz2.append(float(spl[2])* alphahiz)
    sigmasfr_daddiz2.append(float(spl[1]))
fp.close()
sigmag_daddiz2=np.array(sigmag_daddiz2)*Msun/pc**2
sigmasfr_daddiz2=np.array(sigmasfr_daddiz2)*Msun/pc**2/Myr
sigmagtorb_daddiz2=np.array(sigmagtorb_daddiz2)*Msun/pc**2/Myr
torb_daddiz2=sigmag_daddiz2/sigmagtorb_daddiz2

# Data from Tacconi+ 2013, ApJ, 768, 74
sigmag_tacconi=[]
sigmasfr_tacconi=[]
torb_tacconi=[]
fp=open(osp.join(datadir, 'tacconi12.txt'), 'r')
fp.readline()
fp.readline()
for line in fp:
    spl=line.split()
    if spl[2]=='...':
        continue
    sigmag_tacconi.append(10.**float(spl[-2]))
    sigmasfr_tacconi.append(10.**float(spl[-1]))
    torb_tacconi.append(2*np.pi*float(spl[3])/float(spl[2]))
fp.close()
sigmag_tacconi=np.array(sigmag_tacconi)*Msun/pc**2
sigmasfr_tacconi=np.array(sigmasfr_tacconi)*Msun/pc**2/Myr
torb_tacconi=np.array(torb_tacconi)*kpc/(km/s)

# ULIRGs from Kennicutt, 1998, ApJ, 498, 541
sigmag_kenn_ulirg=[]
sigmasfr_kenn_ulirg=[]
torb_kenn_ulirg=[]
fp=open(osp.join(datadir, 'KS_2_KennUlirgs.dat'), 'r')
for line in fp:
    if line[0]=='#':
        continue
    if line[0]=='\n':
        continue
    spl=line.split()
    if spl[2]=='-1':
        continue
    sigmag_kenn_ulirg.append(10.0**float(spl[0])*alphasb)
    sigmasfr_kenn_ulirg.append(10.0**float(spl[1]))
    torb_kenn_ulirg.append(float(spl[2])*100)
fp.close()
sigmag_kenn_ulirg=np.array(sigmag_kenn_ulirg)*Msun/pc**2
sigmasfr_kenn_ulirg=np.array(sigmasfr_kenn_ulirg)*Msun/pc**2/Myr
torb_kenn_ulirg=np.array(torb_kenn_ulirg)*Myr

# Normal spirals from Kennicutt, 1998, ApJ, 498, 541
sigmag_kenn_spiral=[]
sigmasfr_kenn_spiral=[]
torb_kenn_spiral=[]
fp=open(osp.join(datadir, 'KS_2_KennSpirals.dat'), 'r')
for line in fp:
    if line[0]=='#':
        continue
    if line[0]=='\n':
        continue
    spl=line.split()
    if spl[2]=='-1':
        continue
    sigmag_kenn_spiral.append(10.0**float(spl[0])*alphasb)
    sigmasfr_kenn_spiral.append(10.0**float(spl[1]))
    torb_kenn_spiral.append(float(spl[2])*100)
fp.close()
sigmag_kenn_spiral=np.array(sigmag_kenn_spiral)*Msun/pc**2
sigmasfr_kenn_spiral=np.array(sigmasfr_kenn_spiral)*Msun/pc**2/Myr
torb_kenn_spiral=np.array(torb_kenn_spiral)*Myr

# Pixel data from Bigiel et al., 2010, AJ, 140, 1194
try:
    indata
except:
    indata=asciitable.read(osp.join(datadir,'bigiel1.txt'),
                           Reader=asciitable.Cds)
    outdata=np.loadtxt(osp.join(datadir,'things_sflaw_outer_krumholz.txt'),
                       delimiter='&',
                       skiprows=18, usecols=(1,2))

    # Count number of points per galaxy to set weight
    fp=open(osp.join(datadir,'things_sflaw_outer_krumholz.txt'), 'r')
    for i in range(18):
        fp.readline()
    galnameout=[]
    for line in fp:
        galnameout.append(line.partition('&')[0].strip())
    fp.close()
    outnamectr=collections.Counter(galnameout)
    outwgt=np.zeros(len(outdata))
    for n in outnamectr:
        i1=galnameout.index(n)
        i2=len(galnameout)-galnameout[::-1].index(n)
        outwgt[i1:i2] = 1.0/outnamectr[n]
    galnamein=indata['Name'].tolist()
    innamectr=collections.Counter(galnamein)
    inwgt=np.zeros(len(indata))
    for n in innamectr:
        i1=galnamein.index(n)
        i2=len(galnamein)-galnamein[::-1].index(n)
        inwgt[i1:i2] = 1.0/innamectr[n]

    # Get histogram of inner galaxy data
    nbins=50
    hiin=ma.filled(10.**indata['logHI'], 1e-50)
    h2in=ma.filled(10.**indata['logH2'], 1e-50)
    gasin=hiin+h2in
    loggasin=np.log10(gasin)
    logh2in=np.log10(h2in)
    logsfrin=ma.filled(indata['logSFR'], -50)
    histin, xedgein, \
        yedgein = \
        np.histogram2d(loggasin, logsfrin,
                       bins=[nbins,nbins],
                       range=[[-0.5,3.1], [-7,0]])
    xcenin=0.5*(xedgein[:-1]+xedgein[1:])
    ycenin=0.5*(yedgein[:-1]+yedgein[1:])
    histin=histin/np.amax(histin)
    histh2in, dummy, \
        dummy = \
        np.histogram2d(logh2in, logsfrin, bins=[nbins,nbins],
                       range=[[-0.5,3.1], [-7,0]])
    histh2in=histh2in/np.amax(histh2in)
    bbox_in=[xedgein[0], xedgein[-1], yedgein[0], yedgein[-1]]

    # Get histogram of outer galaxy data
    loggasout=outdata[:,0]
    logsfrout=outdata[:,1]
    histout, xedgeout, \
        yedgeout = \
        np.histogram2d(loggasout, logsfrout,
                       bins=[nbins,nbins], weights=outwgt,
                       range=[[-0.5,3.1], [-7,0]])
    xcenout=0.5*(xedgeout[:-1]+xedgeout[1:])
    ycenout=0.5*(yedgeout[:-1]+yedgeout[1:])
    histout=histout/np.amax(histout)
    bbox_out=[xedgeout[0], xedgeout[-1], yedgeout[0], yedgeout[-1]]

    # Bigiel's median and scatter
    binctr=np.array([-0.150515, 0.150515, 0.451545, 0.752575])
    binmed=np.array([-5.50282, -5.05890, -4.47417, -3.98886])
    binscatter=np.array([1.5, 1.2, 0.64, 0.46])


# Leroy et al., 2013, AJ, 146, 19 data
try:
    histh2in_hera
except:
    fp=open(osp.join(datadir,'compile_lit.txt'), 'r')
    h2_hera=[]
    sf_hera=[]
    for line in fp:
        if line.strip()[0] == '#':
            continue
        spl=line.split()
        if spl[-1] == 'HERACLES':
            h2_hera.append(float(spl[0]))
            sf_hera.append(float(spl[1]))
    h2_hera=np.array(h2_hera)
    sf_hera=np.array(sf_hera)
    histh2in_hera, dummy, \
        dummy = \
        np.histogram2d(h2_hera, sf_hera, bins=[nbins,nbins],
                       range=[[-0.5,3.1], [-7,0]])
    histh2in_hera=histh2in_hera/np.amax(histh2in_hera)

# Construct an image from the Leroy+ 2013 data set
img=np.ones((histin.shape[0], histin.shape[1], 3))
nrm=clrs.Normalize(vmin=-2, vmax=0, clip=True)
scalefac = 1.
img[:,:,1]=img[:,:,1]-nrm(np.transpose(np.log10(histin)))*scalefac
img[:,:,0]=img[:,:,0]-nrm(np.transpose(np.log10(histin)))*scalefac
img[:,:,1]=img[:,:,1]-nrm(np.transpose(np.log10(histout)))*scalefac
img[:,:,2]=img[:,:,2]-nrm(np.transpose(np.log10(histout)))*scalefac
img[img < 0]=0.0


# Generate theory curves

# Fiducial parameter choices
Q = 1.0
beta = 0.5
fgP = 0.5
fgQ = 0.5
phimp = 1.4
epsff = 0.015
tsf = 2.0*Gyr
chi = 3.1
fc = 5.0
pmstar = 3e3*km/s
eta = 1.5
phiQ = 2.0
phith = 1.0
phia = 2.0

# Parameters we will vary
Sigma_g = np.logspace(-1, 5, 200)*Msun/pc**2
Sigma_g_torb = np.logspace(-3, 4, 200)*Msun/pc**2/Myr
torb = np.logspace(np.log10(5),np.log10(500),11)*Myr

# Fiducial model
Sigma_sfr_fid1 = []
for t in torb:
    rho_min = (2.0*np.pi/t)**2 * np.sqrt(2*beta+1) / (4.0*np.pi*G)
    fH2 = fH2KMTnew(np.array(Sigma_g.to(g/cm**2)),
                    2*float(rho_min/(g/cm**3)),
                    0.3333, cfac=fc)
    Sigma_sfr_fid1.append(
        fH2 * Sigma_g *
        np.maximum(4*epsff*fgQ/(np.pi*Q) *
                   np.sqrt(2*(1+beta)/(3*fgP*phimp)) * 2.0*np.pi/t,
                   1.0/tsf))
Sigma_sfr_fid2 = []
for t in torb:
    Sg = Sigma_g_torb * t
    rho_min = (2.0*np.pi/t)**2 * np.sqrt(2*beta+1) / (4.0*np.pi*G)
    fH2 = fH2KMTnew(np.array(Sg.to(g/cm**2)),
                    2*float(rho_min/(g/cm**3)),
                    0.3333, cfac=fc)
    Sigma_sfr_fid2.append(
        fH2 * Sigma_g_torb *
        np.maximum(8*epsff*fgQ/Q *
                   np.sqrt(2*(1+beta)/(3*fgP*phimp)),
                   t/tsf)
        )

# Ostriker & Shetty-like model with fixed sigma_g
Sigma_sfr_os1 = [np.pi*G*eta*phimp**0.5*phiQ*phith/pmstar *
                 Sigma_g**2]*len(torb)
Sigma_sfr_os2 = []
for t in torb:
    Sg = Sigma_g_torb * t
    Sigma_sfr_os2.append(np.pi*G*eta*phimp**0.5*phiQ*phith/pmstar *
                         Sg**2)

# Faucher-Giguere, Quatert, & Hopkins-like model with varying
# epsilon_ff
Sigma_sfr_fqh1 = [np.pi**2*G*eta*phimp**0.5*phiQ*phith*Q/(fgP*pmstar) *
                  Sigma_g**2]*len(torb)
Sigma_sfr_fqh2 = []
for t in torb:
    Sg = Sigma_g_torb * t
    Sigma_sfr_fqh2.append(
        np.pi**2*G*eta*phimp**0.5*phiQ*phith*Q/(fgP*pmstar) *
        Sg**2)



# Plot data and models; first whole-galaxy data

# Set up plot
fig = plt.figure(1, figsize=(6,5))
plt.clf()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
try:
    # Depends on matplotlib version
    plt.rc('xtick', direction='in', top=True)
    plt.rc('ytick', direction='in', right=True)
except:
    pass
ms=5
colors=['r', 'g', 'b', 'c', 'm', 'y']

# Overall y axis
ax = fig.add_subplot(1,1,1)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(top='off', bottom='off', left='off', right='off')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_ylabel(r'$\log\,\dot{\Sigma}_*$ [$M_\odot$ '
              '$\mathrm{pc}^{-2}$ $\mathrm{Myr}^{-1}$]',
              labelpad=24)


# Sigma_SFR vs Sigma_g, fiducial model
ax = fig.add_subplot(2,2,1)

# Data
pdata = []
labeldata = []
p,=plt.plot(np.log10(sigmag_kenn_ulirg/(Msun/pc**2)),
            np.log10(sigmasfr_kenn_ulirg/(Msun/pc**2/Myr)), 
            colors[0]+'s', ms=ms, mec='k')
pdata.append(p)
labeldata.append('K98')
plt.plot(np.log10(sigmag_kenn_spiral/(Msun/pc**2)),
         np.log10(sigmasfr_kenn_spiral/(Msun/pc**2/Myr)), 
         colors[0]+'s', ms=ms, mec='k')
p,=plt.plot(np.log10(sigmag_genzel/(Msun/pc**2)),
            np.log10(sigmasfr_genzel/(Msun/pc**2/Myr)), 
            colors[2]+'^', ms=ms, mec='k')
pdata.append(p)
labeldata.append('G10')
p,=plt.plot(np.log10(sigmag_bouche/(Msun/pc**2)),
            np.log10(sigmasfr_bouche/(Msun/pc**2/Myr)), 
            colors[3]+'p', ms=ms, mec='k')
pdata.append(p)
labeldata.append('B07')
p,=plt.plot(np.log10(sigmag_daddiz05/(Msun/pc**2)),
            np.log10(sigmasfr_daddiz05/(Msun/pc**2/Myr)), 
            colors[4]+'*', ms=ms, mec='k')
pdata.append(p)
labeldata.append('D08, D10')
plt.plot(np.log10(sigmag_daddiz2/(Msun/pc**2)),
         np.log10(sigmasfr_daddiz2/(Msun/pc**2/Myr)), 
         colors[4]+'*', ms=ms, mec='k')
p,=plt.plot(np.log10(sigmag_tacconi/(Msun/pc**2)),
            np.log10(sigmasfr_tacconi/(Msun/pc**2/Myr)), 
            colors[5]+'h', ms=ms, mec='k')
pdata.append(p)
labeldata.append('T13')

# Fidicual model
pmod = []
labelmod = []
for i in range(len(torb)):
    p,=plt.plot(np.log10(Sigma_g/(Msun/pc**2)),
                np.log10(phia*Sigma_sfr_fid1[i]/(Msun/pc**2/Myr)),
                lw=2,
                color = cmap.Greens(0.3 + 0.7*float(i)/(len(torb)-1)))

# Add legend
leg1 = plt.legend(pdata[:3], labeldata[:3], loc='upper left',
                  prop={"size":10}, numpoints=1)
plt.legend(pdata[3:], labeldata[3:], loc='lower right',
                  prop={"size":10}, numpoints=1)
plt.gca().add_artist(leg1)

# Adjust axes
plt.xlim([0,4.9])
plt.ylim([-4,4.8])
plt.setp(ax.get_xticklabels(), visible=False)


# Sigma_SFR vs Sigma_g / t_orb, fiducial model
ax = fig.add_subplot(2,2,2)

# Data
p,=plt.plot(np.log10(sigmag_kenn_ulirg/(Msun/pc**2)/(torb_kenn_ulirg/Myr)),
            np.log10(sigmasfr_kenn_ulirg/(Msun/pc**2/Myr)), 
            colors[0]+'s', ms=ms, mec='k')
plt.plot(np.log10(sigmag_kenn_spiral/(Msun/pc**2)/(torb_kenn_spiral/Myr)),
         np.log10(sigmasfr_kenn_spiral/(Msun/pc**2/Myr)), 
         colors[0]+'s', ms=ms, mec='k')
p,=plt.plot(np.log10(sigmag_genzel/(Msun/pc**2)/(torb_genzel/Myr)),
            np.log10(sigmasfr_genzel/(Msun/pc**2/Myr)), 
            colors[2]+'^', ms=ms, mec='k')
p,=plt.plot(np.log10(sigmag_bouche/(Msun/pc**2)/(torb_bouche/Myr)),
            np.log10(sigmasfr_bouche/(Msun/pc**2/Myr)), 
            colors[3]+'p', ms=ms, mec='k')
p,=plt.plot(np.log10(sigmag_daddiz05/(Msun/pc**2)/(torb_daddiz05/Myr)),
            np.log10(sigmasfr_daddiz05/(Msun/pc**2/Myr)), 
            colors[4]+'*', ms=ms, mec='k')
plt.plot(np.log10(sigmag_daddiz2/(Msun/pc**2)/(torb_daddiz2/Myr)),
         np.log10(sigmasfr_daddiz2/(Msun/pc**2/Myr)), 
         colors[4]+'*', ms=ms, mec='k')
p,=plt.plot(np.log10(sigmag_tacconi/(Msun/pc**2)/(torb_tacconi/Myr)),
            np.log10(sigmasfr_tacconi/(Msun/pc**2/Myr)), 
            colors[5]+'h', ms=ms, mec='k')

# Fidicual model
pmod = []
labelmod = []
for i in range(len(torb)):
    p,=plt.plot(np.log10(Sigma_g_torb/(Msun/pc**2/Myr)),
                np.log10(phia*Sigma_sfr_fid2[i]/(Msun/pc**2/Myr)),
                lw=2,
                color = cmap.Greens(0.3 + 0.7*float(i)/(len(torb)-1)))
    if i == 0 or i == len(torb)-1:
        pmod.append(p)
        labelmod.append('T+F, $t_{{\mathrm{{orb}}}} = {:d}$ Myr'.
                        format(int(round(torb[i].to(Myr).value))))

# Add legend
plt.legend(pmod, labelmod, loc='upper left',
           prop={"size":10}, numpoints=1)

# Adjust axes
plt.xlim([-3,4])
plt.ylim([-4,4.8])
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)

# Add label on right
ax=plt.twinx()
plt.xlim([-3,4])
plt.ylim([-4,4.8])
plt.setp(ax.get_yticklabels(), visible=False)
plt.ylabel(r'$\mathrm{Transport+feedback}$')


# Sigma_SFR vs Sigma_g, no transport model
ax = fig.add_subplot(2,2,3)

# Data
pdata = []
labeldata = []
p,=plt.plot(np.log10(sigmag_kenn_ulirg/(Msun/pc**2)),
            np.log10(sigmasfr_kenn_ulirg/(Msun/pc**2/Myr)), 
            colors[0]+'s', ms=ms, mec='k')
plt.plot(np.log10(sigmag_kenn_spiral/(Msun/pc**2)),
         np.log10(sigmasfr_kenn_spiral/(Msun/pc**2/Myr)), 
         colors[0]+'s', ms=ms, mec='k')
p,=plt.plot(np.log10(sigmag_genzel/(Msun/pc**2)),
            np.log10(sigmasfr_genzel/(Msun/pc**2/Myr)), 
            colors[2]+'^', ms=ms, mec='k')
p,=plt.plot(np.log10(sigmag_bouche/(Msun/pc**2)),
            np.log10(sigmasfr_bouche/(Msun/pc**2/Myr)), 
            colors[3]+'p', ms=ms, mec='k')
p,=plt.plot(np.log10(sigmag_daddiz05/(Msun/pc**2)),
            np.log10(sigmasfr_daddiz05/(Msun/pc**2/Myr)), 
            colors[4]+'*', ms=ms, mec='k')
plt.plot(np.log10(sigmag_daddiz2/(Msun/pc**2)),
         np.log10(sigmasfr_daddiz2/(Msun/pc**2/Myr)), 
         colors[4]+'*', ms=ms, mec='k')
p,=plt.plot(np.log10(sigmag_tacconi/(Msun/pc**2)),
            np.log10(sigmasfr_tacconi/(Msun/pc**2/Myr)), 
            colors[5]+'h', ms=ms, mec='k')

# No transport model
pmod = []
labelmod = []
p,=plt.plot(np.log10(Sigma_g/(Msun/pc**2)),
            np.log10(phia*Sigma_sfr_os1[0]/(Msun/pc**2/Myr)),
            lw=2,
            color = cmap.Reds(1.0))

# Adjust axes
plt.xlim([0,4.9])
plt.ylim([-4,4.8])
plt.xlabel(r'$\log\,\Sigma_{\mathrm{g}}$ [$M_\odot$ $\mathrm{pc}^{-2}$]')


# Sigma_SFR vs Sigma_g / t_orb, no transport model
ax = fig.add_subplot(2,2,4)

# Data
p,=plt.plot(np.log10(sigmag_kenn_ulirg/(Msun/pc**2)/(torb_kenn_ulirg/Myr)),
            np.log10(sigmasfr_kenn_ulirg/(Msun/pc**2/Myr)), 
            colors[0]+'s', ms=ms, mec='k')
plt.plot(np.log10(sigmag_kenn_spiral/(Msun/pc**2)/(torb_kenn_spiral/Myr)),
         np.log10(sigmasfr_kenn_spiral/(Msun/pc**2/Myr)), 
         colors[0]+'s', ms=ms, mec='k')
p,=plt.plot(np.log10(sigmag_genzel/(Msun/pc**2)/(torb_genzel/Myr)),
            np.log10(sigmasfr_genzel/(Msun/pc**2/Myr)), 
            colors[2]+'^', ms=ms, mec='k')
p,=plt.plot(np.log10(sigmag_bouche/(Msun/pc**2)/(torb_bouche/Myr)),
            np.log10(sigmasfr_bouche/(Msun/pc**2/Myr)), 
            colors[3]+'p', ms=ms, mec='k')
p,=plt.plot(np.log10(sigmag_daddiz05/(Msun/pc**2)/(torb_daddiz05/Myr)),
            np.log10(sigmasfr_daddiz05/(Msun/pc**2/Myr)), 
            colors[4]+'*', ms=ms, mec='k')
plt.plot(np.log10(sigmag_daddiz2/(Msun/pc**2)/(torb_daddiz2/Myr)),
         np.log10(sigmasfr_daddiz2/(Msun/pc**2/Myr)), 
         colors[4]+'*', ms=ms, mec='k')
p,=plt.plot(np.log10(sigmag_tacconi/(Msun/pc**2)/(torb_tacconi/Myr)),
            np.log10(sigmasfr_tacconi/(Msun/pc**2/Myr)), 
            colors[5]+'h', ms=ms, mec='k')

# No-transport model
pmod = []
labelmod = []
for i in range(len(torb)):
    p,=plt.plot(np.log10(Sigma_g_torb/(Msun/pc**2/Myr)),
                np.log10(phia*Sigma_sfr_os2[i]/(Msun/pc**2/Myr)),
                lw=2,
                color = cmap.Reds(0.5 + 0.5*float(i)/(len(torb)-1)))
    if i == 0 or i == len(torb)-1:
        pmod.append(p)
        labelmod.append('NT, $t_{{\mathrm{{orb}}}} = {:d}$ Myr'.
                        format(int(round(torb[i].to(Myr).value))))

# Add legend
plt.legend(pmod, labelmod, loc='upper left',
           prop={"size":10}, numpoints=1)

# Adjust axes
plt.xlim([-3,4])
plt.ylim([-4,4.8])
plt.setp(ax.get_yticklabels(), visible=False)
plt.xlabel(r'$\log\,\Sigma_{\mathrm{g}}/t_{\mathrm{orb}}$ [$M_\odot$'
           ' $\mathrm{pc}^{-2}$'
           ' $\mathrm{Myr}^{-1}$]')

# Add label on right
ax=plt.twinx()
plt.ylim([-4,4.8])
plt.setp(ax.get_yticklabels(), visible=False)
plt.ylabel('No-transport')

# Adjust spacing
plt.subplots_adjust(hspace=0, wspace=0, top=0.95)

# Save
plt.savefig('sflaw_unresolved.pdf')


# Now plot resolved data
fig = plt.figure(2, figsize=(5,3.5))
plt.clf()
ax=plt.subplot(111)
plots = []
labels = []

# Observations
plt.imshow(img, extent=bbox_in,
           interpolation='nearest', aspect='auto', origin='lower')
prd = Rectangle((0,0), 1, 1, fc='r')
pbl = Rectangle((0,0), 1, 1, fc='b')
plots.append(pbl)
plots.append(prd)
labels.append('L13')
labels.append('B10')
p1=plt.errorbar(binctr, binmed, yerr=binscatter, marker='o', ms=ms, mfc='r', 
                mec='k', ecolor='r', elinewidth=2, capsize=4, capthick=2, 
                fmt='o')
plots.append(p1)
labels.append('B10 (med)')

# Fiducial model
pmod = []
labelmod = []
idx = np.where(torb >= 49*Myr)[0]
for i in range(len(idx)):
    p,=plt.plot(np.log10(Sigma_g/(Msun/pc**2)),
                np.log10(Sigma_sfr_fid1[idx[i]]/(Msun/pc**2/Myr)),
                lw=2,
                color = cmap.Greens(0.5 + 0.5*float(i)/(len(idx)-1)))
    if i==0 or i == len(idx)-1:
        pmod.append(p)
        labelmod.append('T+F, $t_{{\mathrm{{orb}}}} = {:d}$ Myr'.
                        format(int(round(torb[idx[i]].to(Myr).value))))
        
# No transport model
p,=plt.plot(np.log10(Sigma_g/(Msun/pc**2)),
            np.log10(Sigma_sfr_os1[0]/(Msun/pc**2/Myr)),
            lw=2, ls='--',
            color = 'k')
pmod.append(p)
labelmod.append('NT')

# Legends
leg=plt.legend(plots, labels, loc='upper left',
               prop={"size":10}, numpoints=1)
plt.legend(pmod, labelmod, loc='lower right',
           prop={"size":10}, numpoints=1)
ax.add_artist(leg)
 
# Adjust plot
plt.xlim([-0.5,3])
plt.ylim([-6,0])
plt.subplots_adjust(left=0.125, right=0.8, bottom=0.15, top=0.9)

# Add labels
plt.xlabel(r'$\log\,\Sigma_{\mathrm{g}}$ [$M_\odot$ $\mathrm{pc}^{-2}$]')
plt.ylabel(r'$\log\,\dot{\Sigma}_*$ [$M_\odot$ $\mathrm{pc}^{-2}$ $\mathrm{Myr}^{-1}$]')

# Add colorbars
cdictbl={ 'red' : [ (0.0, 1.0, 1.0),
                    (1.0, 0.0, 1.0-scalefac) ],
          'green' : [ (0.0, 1.0, 1.0),
                      (1.0, 0.0, 1.0-scalefac) ],
          'blue' : [ (0.0, 0.0, 1.0),
                     (1.0, 1.0, 1.0) ] }
cdictrd={ 'red' : [ (0.0, 0.0, 1.0),
                    (1.0, 1.0, 1.0) ],
          'green' : [ (0.0, 0.0, 1.0),
                      (1.0, 0.0, 1.0-scalefac) ],
          'blue' : [ (0.0, 0.0, 1.0),
                     (1.0, 0.0, 1.0-scalefac) ] }
cmapbl=clrs.LinearSegmentedColormap('bl', cdictbl)
cmaprd=clrs.LinearSegmentedColormap('rd', cdictrd)
cbarwidth=0.025
pts=ax.get_axes().get_position().get_points()
axcbar1=fig.add_axes([pts[1,0], pts[0,1], cbarwidth, pts[1,1]-pts[0,1]], label='bar1')
norm=clrs.Normalize(vmin=-2, vmax=0.0)
cbar.ColorbarBase(axcbar1, norm=norm, orientation='vertical',
                  ticks=[-2,-1.5,-1.0,-0.5,0], cmap=cmapbl)
plt.setp(axcbar1.get_yticklabels(), visible=False)
axcbar2=fig.add_axes([pts[1,0]+cbarwidth, pts[0,1], cbarwidth, pts[1,1]-pts[0,1]], label='bar2')
cbar.ColorbarBase(axcbar2, norm=norm, orientation='vertical', 
                  ticks=[-2,-1.5,-1.0,-0.5,0], cmap=cmaprd)
axcbar2.set_ylabel('Log point density')
axcbar2.yaxis.set_ticks_position('right')
axcbar2.yaxis.set_label_position('right')

# Save
plt.savefig('sflaw_resolved.pdf')
