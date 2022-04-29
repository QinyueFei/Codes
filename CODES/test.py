# %%
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c

# %%
from Dynamics.models import *
from scipy.misc import derivative

plt.rc('text', usetex=True)
plt.rc('font', family='dejavuserif', size=25)
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)

# %%
G = 4.302e-6
r = np.linspace(1e-3, 10.0, 10000)
v_bulge = V_b(r, 10.72, 1.61, 1.69)
v_disk = V_d(r, 10.64, 10.97)
MBH = 1e7
v_BH = np.sqrt(G*MBH/r)

v_circ = np.sqrt(v_bulge**2 + v_disk**2 + v_BH**2)

# %%
# Omega = v_circ / r
def Omega2(x):
    v_circ2 =V_b(x, 10.72, 1.61, 1.69)**2 + V_d(x, 10.64, 10.97)**2 + G*MBH/x
    return v_circ2/x**2

kappa = np.sqrt(r * derivative(Omega2, r, dx=1e-8) + 4 * Omega2(r))

# %%
plt.figure(figsize=(8, 6))
ax = plt.subplot(111)
# ax.plot(r, v_bulge, 'r-')
# ax.plot(r, v_disk, 'b--')
ax.plot(r, v_circ, 'k-')
ax.plot(r, np.sqrt(Omega2(r)), 'k--')
ax.plot(r, np.sqrt(Omega2(r)) - kappa/2, 'r--')
ax.plot(r, np.sqrt(Omega2(r)) + kappa/2, 'r--')

ax.set_xlabel('r [kpc]')
ax.set_ylabel(r'v [km/s], $\Omega$ [km/s/kpc]')
ax.set_xlim(1e-3, 10)
ax.set_ylim(0, 400)
ax.semilogx()
# %%
