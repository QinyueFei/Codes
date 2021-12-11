import numpy as np
import matplotlib.pyplot as plt

def load_Wilson(dir):
    #dir = '/home/qyfei/Desktop/Codes/CODES/literature/'
    Wilson19 = np.loadtxt(dir+'Wilson19.txt', dtype='str')

    Disp_CO_Wilson = np.array(Wilson19[:,3], dtype='float')
    Sigma_H2_Wilson = np.power(10, np.array(Wilson19[:,5], dtype='float'))
    return Sigma_H2_Wilson, Disp_CO_Wilson

def load_Bolatto(dir):
    #
    Bolatto17_01 = np.loadtxt(dir+'Bolatto17_1.txt', dtype='str')
    Bolatto17_02 = np.loadtxt(dir+'Bolatto17_2.txt', dtype='str')

    Levy18 = np.loadtxt(dir+'Levy18.txt', dtype='str')
    Name_CO_Levy = Levy18[:,0]
    Disp_CO_Levy = np.array(Levy18[:,16], dtype='float')
    Sigma_s_Levy = np.array(Levy18[:,14], dtype='float')
    N = []
    M = []
    for i in range(len(Name_CO_Levy)):
        N.append(np.where(Bolatto17_01[:,0]==Name_CO_Levy[i])[0][0])
        M.append(np.where(Bolatto17_02[:,0]==Name_CO_Levy[i])[0][0])

    Bolatto_H2_Mass = 10**np.array(Bolatto17_01[N][:,5], dtype='float')
    Bolatto_Rhmol = np.array(Bolatto17_02[M][:,7], dtype='float')
    Sigma_H2_Bolatto = 0.5*Bolatto_H2_Mass/(np.pi*Bolatto_Rhmol**2)/1e6

    return Sigma_H2_Bolatto, Sigma_s_Levy, Disp_CO_Levy

'''# Plot literature data
plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
ax.errorbar(Sigma_H2_Wilson, Disp_CO_Wilson, fmt='ms', mfc='none', ms=8, mew=2, zorder=3, label='z$\sim$0 ULIRGs')
ax.errorbar(Sigma_H2_Bolatto, Disp_CO_Levy, fmt='bs', mfc='none', ms=8, mew=2, zorder=3, label='z$\sim$0 SFGs')
ax.loglog()
ax.set_xlim(10, 10**4.5)
ax.set_ylim(3, 210)
ax.legend(loc='lower right', fontsize=18, frameon=False)
plt.show()'''