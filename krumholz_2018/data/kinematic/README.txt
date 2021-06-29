Below we list the data sources for each kinematic data set, along with
the assumptions made in extracting the quantities we need.

epinat08.txt -- data from the GHASP survey, as repored by Epinat+
2008, MNRAS, 390, 446; extracted from the VizieR database, accessed at
http://vizier.cfa.harvard.edu/viz-bin/VizieR-3?-source=J/MNRAS/390/466/galaxies
; data have been 999-filled, so values of 99 for the velocity
dispersion correspond to missing data; columns are (1) D = distance
[Mpc], (2) sigma = velocity dispersion [km/s], (3) Ha_flux = Halpha
flux [10^-16 W / m^2]; when plotting these values we do not correct
the Halpha velocity dispersions for thermal broadening, because
Epinat's fitting method is not sensitive to it, as they are
effectively only fitting the centroid velocity at each pixel

epinat09.txt -- data from Epinat et al., 2009, MNRAS, 504, 789, copied
by hand from their tables 3 and 5; columns are (1) SFR = star formation
rate [Msun/yr]; this is the Halpha-based estimate given in their table
3; (2) sigma = velocity dispersion [km/s]; this is the value given as
the local mean velocity dispersion sigma_0 in their table 5

cresci09.txt -- data from the SINS survey, Cresci et al., 2009, ApJ,
697, 115, copied by hand from their tables 1 and 2; columns are (1)
galaxy name, (2) SFR = star formation rate [Msun/yr], (3) vdisp =
velocity dispersion [km/s], (4) fgas = gas fraction

law09.txt -- data from Law et al., 2009, ApJ, 697, 2057; columns are
(1) SFR = star formation rate [Msun/yr]; the value given is their
estimate based on nebular emission (SFR_neb in their tables), (2)
sigma = velocity dispersion [km/s]; this value is their sigma_mean,
using an average in cases where they give more than 1 value per
galaxy; (3) Ms = stellar mass [10^10 Msun], (4) Mg = gas mass [10^10
Msun], with a value of -99 indicating a non-detection

jones10.txt -- lensed galaxies from Jones et al., 2010, MNRAS, 404,
1247, copied by hand from their table 1; columns are (1) SFR = star
formation rate [Msun/yr], (2) sigma = velocity dispersion [km/s]

lemoine-busserolle10.txt -- data from Lemoine-Busserolle et al., 2010,
MNRAS, 401, 1657, from their tables 5 and 6; columns are (1) SFR =
star formation rate [Msun/yr], (2) sigma = velocity dispersion [km/s];
this is taken to be the quantity they call sigma_mean, (3) fgas = gas
fraction

wisnioski11a.txt -- data from the WiggleZ survey, Wisnioski et al.,
2011, MNRAS, 417, 2601, copied by hand from their tables 3 and
4. Columns are (1) LHa = Halpha luminosity [10^41 erg/s], (2) sigma =
velocity dispersion [km/s]; this is their sigma_mean rather than their
sigma_net

ianjamasimanana12.txt -- data from Ianjamasimanana et al., 2012, AJ,
144, 96, extracted by hand; columns are (1) Name = galaxy name, (2)
vdisp = velocity dispersion [km/s]; this is their single Gaussian fit
value; (3) SFR = star formation rate [Msun/yr], (4) logM* = log
stellar mass [Msun], (5) = log HI mass [Msun], (6) = log H2 mass
[Msun]; masses are taken from Walter et al., 2008, AJ, 136, 2563 (the
THINGS survey paper)

stilp13.txt -- data come from Stilp et al., 2013, ApJ, 765, 136, their
tables 3 and 4; columns are (1) log_Mb = log baryon mass [Msun], (2)
log_MHI = log HI mass [Msun], (3) log_Ms = log stellar mass [Msun],
(4) SFR = star formation rate [Msun/yr], (5) sigma = velocity
dispersion (their sigma_central) [km/s]

green14.txt -- data from Green et al., 2014, MNRAS, 437, 1070;
extracted from the VizieR database at
http://vizier.cfa.harvard.edu/viz-bin/VizieR-3?-source=J/MNRAS/437/1070/sample;
columns are (1) Ms = stellar mass [10^9 Msun], (2) SFRB04 = star
formation rate from Brichmann+ 2004 [Msun/yr], (3) SFRHa = star
formation rate based on Halpha [Msun/yr], (4) sigma = velocity
dispersion [km/s]

moiseev15.csv -- data from Moiseev et al., 2015, MNRAS, 449, 3568;
extracted by hand; columns are (1) galaxy name, (2) D = distance
[Mpkc], (3) log_LHa = log Ha luminosity [erg/s], (4) sigma = velocity
dispersion [km/s], (5) sigma err = error on velocity dispersion
[km/s]; note that the sigma values listed have had 9.1 km/s subtracted
in quadrature in order correct for thermal broadening, unlike any of
the other data sets

varidel16.txt -- data from Varidel et al., 2016, PASA, 33, e006,
copied by hand from their tables 1 and 2; columns are (1) SFR = star
formation rate [Msun/yr]; (2) sigma = velocity disperison [km/s]. For
the SFR we use their IFU-based estimate, while for velocity dispersion
we use their flux-weighted mean (sigma_m in their notation)

di-teodoro16.txt -- data from Di-Teodoro et al., 2016, A&A, 594, A77,
analysing the KMOS survey; data copied directly from their table 1;
columns are (1) SFR = star formation rate [Msun/yr], (2) sigma =
velocity dispersion [km/s]

johnson17.fits -- measurements from the KROSS survey, Johnson et al.,
2017, in prep, provided by H. Johnson. Data are contained in the 2nd
FITS HDU; column headers are 'SFR' = star formation rate [Msun/yr] and
'SIGMA0' = velocity disperison [km/s]

ulirgs.txt -- a compilation of data on local ULIRGs. Data on Arp220
are from Scoville et al., 2017, ApJ, 836, 66, assuming SFRs are
identical for Arp 220's 2 nuclei, since they have nearly equal IR
luminosities; NGC6240 is from Scoville et al., 2015, ApJ, 800, 70; for
the remainder, velocity dispersions are from Downes & Solomon, 1998,
ApJ, 507, 615, and SFRs are derived from the IR luminosity given in
Veilleux et al., 2009, ApJS, 182, 628 or, if that is not available,
Sanders et al., 2003, AJ, 126, 1607, converted to a SFR using the
conversion of Kennicutt & Evans, 2012, ARA&A, 50, 31; for the Veilleux
et al. IR luminosities we use the AGN fraction they give to correct
the IR luminosity before converting, while for the Sanders et
al. luminosities we adopt an AGN fraction of 50%. Columns are
(1) name = galaxy name, (2) SFR = star formation rate [Msun/yr], (3)
sigma = velocity dispersion [km/s], (4) Reference = references


