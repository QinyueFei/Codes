This README explains the data files contained in this directory. The
compilation here is partly from E. Daddi, with substantial additions
from M. Krumholz.

genzel_ks.dat -- data taken from Genzel et al., 2010, MNRAS, 407,
2091. Column names in the file should be fairly self-explanatory, and
match those given in the paper. This file is originally from E. Daddi.

KS_2_Bouche.dat -- data from Bouche et al., 2007, ApJ, 671, 303. First
column is log Sigma_gas [Msun/pc^2], second is log Sigma_SFR
[Msun/pc^2/Myr], third is log Sigma_gas / t_orb [Msun/pc^2/Myr]. This
file is originally from E. Daddi.

KS_2_Daddiz05.dat -- data from Daddi et al., 2008, ApJL, 673, L21 and
2010, ApJ, 713, 686, for galaxies at z = 0.5. Columns are the same as
for KS_2_Bouche.dat. This file is originally from E. Daddi.

KS_2_Daddi.dat -- same as KS_2_Daddiz05.dat, but for the z = 2 sample.

tacconi12.txt -- data from Tacconi et al., 2013, ApJ, 768, 74, as
transcribed by M. Krumholz. Column names and contents are identical to
the table in their paper.

KS_2_KennUlirgs.dat -- data on local ULIRGs from Kennicutt, 1998, ApJ,
498, 541, with SFRs adjusted from a Salpeter to a Chabrier IMF. File
format is the same as KS_2_Bouche.dat. This file is from E. Daddi.

KS_2_KennSpirals.dat -- same as KS_2_KennUlirgs.dat, but for the
sample of local spirals.

bigiel1.txt -- data on the galaxy sample of Bigiel et al., 2010, AJ,
140, 1194, taken from the CDS website entry associated with this
paper; format is standard CDS, explained in file header

things_sflaw_outer_krumholz.txt -- data on the galaxy sample of Bigiel
et al., 2010, AJ, 140, 1194, provided by F. Bigiel to
M. Krumholz. Each entry represents an individual pointing. First
column is galaxy name, second is log HI surface density [Msun/pc^2],
third is log SFR surface density [Msun/pc^2/Myr]

compile_lit.txt -- data from Leroy et al., 2013, AJ, 146, 19, provided
by A. Leroy to M. Krumholz. Each entry represents an individual pixel
in the HERACLES survey or another comparable survey; first column is
log gas surface density [Msun/pc^2; note: file header says H2, but
this is an error], second column is log SFR surface density
[Msun/pc^2/Myr], third is data source
