import numpy as np
def Sigma_disk(Sigma_, radius_, re_):
    ## Sigma, re = np.power(10, 10.64)/40.965, 10.97
    Sigma, radius, re = Sigma_, radius_, re_
    Sigma_e = Sigma
    #print(np.log10(trapz(Sigma_s, 2*np.pi*R)))

    Sigma_s = Sigma_e*np.exp(-1.68*radius/re)/1e6
    return Sigma_s
