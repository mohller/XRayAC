import numpy as np
from numpy import pi, log, sqrt

mp = 938.  # MeV
Na = 6.02e23  # particules per mole (Avogadro number)
Mu = 1.  # g/mol (molar mass constant)
e = 1.6e-19  # Coulomb (electron charge)
ke = 9e9  # Nm2/C2
# konst = e**2/4/pi/eps0 = e**2 * ke, in units N*m**2 = J*m = 1/e eV*m = 1E-4/e eV*m
konst = e * ke * 1e-4 # MeV * cm 

def bethe_bloch(KE, Z=3., A=6., rho=1., relativistic=True, I=None):
    """Returns the differential energy loss per depth
    for protons of speed v (not relativistic) in a medium
    with density rho as a function of the kinetic energy 
    KE = m * v**2 / 2 in MeV units

    https://en.wikipedia.org/wiki/Bethe_formula (non relativistic formula eq(2))
    """ 
    M = A  # grams / mol (molar mass)
    rho2 = rho * 1e-3  # to g/mm3
    n = (Na * rho / M ) * Z
    if I is None:
        I = 10. * Z * 1e-6  # MeV (mean excitation potential of Block, see wikipedia)
    
    if relativistic:
        beta2 = 1 - (KE / mp + 1)**-2  # from relativistic relation: gamma*mp = mp + KE
        return 4 * pi / mp * n * Z**2 / beta2 * konst**2 * (log(2 * mp * beta2 / I / (1 - beta2)) - beta2)
    else:  # classic
        beta2 = 2 * KE / mp # from classic relation KE = mo * beta**2 * c**2 / 2
        # return 4 * pi * n * Z**2 / mp / v**2 *(e**2/4/pi/eps0)**2 * log(2 * mp * v**2/I)
        return 2 * pi * n * Z**2 / KE * konst**2 * log(4 * KE / I)  # low energy limit



def stopping_power(KE, Z=3., A=6., rho=1., **kwargs):
    """Returns the differential energy loss per thickness
    for protons of kinetic energy KE (not relativistic) in 
    a medium with density rho
    KE = m * v**2 / 2 in MeV units

    https://en.wikipedia.org/wiki/Bethe_formula (non relativistic formula eq(2))
    https://physics.nist.gov/PhysRefData/Star/Text/programs.html
    """ 
    return bethe_bloch(KE, Z, A, rho, **kwargs) / rho


def burells_formula(energies, material=None):
    """Calculates the range for protons with the given energies
    in the given material. Implementation of the equations 14-17 
    in the paper Burell1964.

    Arguments:
    ----------
    energies: numpy array, energies for which the range is calculated (MeV)
    material: tuple (Z, A), the proton and mass number of the desired material
              any of the materials given as string:
              `carbon', 
              `aluminum', 
              `iron', 
              `copper', 
              `tungsten' 

    Reference:
    Burell1964 - M. O. Burrell.  The calculation of proton penetration and doserates.  1964
    """
    materials = {
        'carbon'   : (1.78, 2.33E-3, 2.0E-6),
        'aluminum' : (1.78, 2.77E-3, 2.5E-6),  
        'iron'     : (1.75, 3.70E-3, 2.6E-6),
        'copper'   : (1.75, 3.85E-3, 2.7E-6),
        'silver'   : (1.75, 4.55E-3, 3.7E-6),
        'tungsten' : (1.75, 5.50E-3, 4.2E-6)
    }

    if material is None:
        r, a, b = materials['carbon']
    elif material in materials:
        r, a, b = materials[material]
    elif len(material) == 2:
        Z, A = material
        if Z <= 20:
            r = 1.78
            a = 1.53E-3 + 2.33E-4*np.sqrt(A)
            b = 8E-7 + 5E-7*np.sqrt(Z)
        elif Z > 20:
            r = 1.75
            a = 1.6E-3 + 2.89E-4*np.sqrt(A)
            b = 5.16E-7*np.sqrt(Z)

    proton_range = a/2/b * np.log(1 + 2*b*energies**r)

    return energies, proton_range

