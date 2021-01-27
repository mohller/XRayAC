import numpy as np
from numpy import pi, log, sqrt

mp = 938.  # MeV
Na = 6.02e23  # particules per mole (Avogadro number)
Mu = 1.  # g/mol (molar mass constant)
e = 1.6e-19  # Coulomb (electron charge)
ke = 9e9  # Nm2/C2
# konst = e**2/4/pi/eps0 = e**2 * ke, in units N*m**2 = J*m = 1/e eV*m = 1E-4/e eV*m
konst = e * ke * 1e-4 # MeV * cm 

def bethe_bloch(KE, Z=3., A=6., rho=1., relativistic=True):
    """Returns the differential energy loss per depth
    for protons of speed v (not relativistic) in a medium
    with density rho as a function of the kinetic energy 
    KE = m * v**2 / 2 in MeV units

    https://en.wikipedia.org/wiki/Bethe_formula (non relativistic formula eq(2))
    """ 
    M = A  # grams / mol (molar mass)
    rho2 = rho * 1e-3  # to g/mm3
    n = (Na * rho / M ) * Z
    I = 10. * Z * 1e-6  # MeV (mean excitation potential of Block, see wikipedia)
    
    if relativistic:
        beta2 = 1 - (KE / mp + 1)**-2  # from relativistic relation: gamma*mp = mp + KE
        return 4 * pi / mp * n * Z**2 / beta2 * konst**2 * (log(2 * mp * beta2 / I / (1 - beta2)) - beta2)
    else:  # classic
        beta2 = 2 * KE / mp # from classic relation KE = mo * beta**2 * c**2 / 2
        # return 4 * pi * n * Z**2 / mp / v**2 *(e**2/4/pi/eps0)**2 * log(2 * mp * v**2/I)
        return 2 * pi * n * Z**2 / KE * konst**2 * log(4 * KE / I)  # low energy limit



def stopping_power(KE, Z=3., A=6., rho=1.):
    """Returns the differential energy loss per thickness
    for protons of kinetic energy KE (not relativistic) in 
    a medium with density rho
    KE = m * v**2 / 2 in MeV units

    https://en.wikipedia.org/wiki/Bethe_formula (non relativistic formula eq(2))
    https://physics.nist.gov/PhysRefData/Star/Text/programs.html
    """ 
    return bethe_bloch(KE, Z, A, rho) / rho