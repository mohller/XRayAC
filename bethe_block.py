# from cmath import sqrt
import numpy as np
from scipy.integrate import cumtrapz
# import molmass
from numpy import sqrt, log, logspace
from scipy.constants import pi, c, epsilon_0, N_A, e, m_e, m_p, eV, milli, mega, centi


def bethe_bloch(KE, Z=3., A=6., rho=1., I=None):
    """Returns the differential energy loss per depth
    for protons of speed v (not relativistic) in a medium
    with density rho as a function of the kinetic energy 
    KE = m * v**2 / 2 in MeV units

    https://en.wikipedia.org/wiki/Bethe_formula (non relativistic formula eq(2))

    Arguments:
    ==========
    KE  : Kinetic Energy in MeV
    Z   : Material's atomic number
    A   : Material's mass number (and approximate molar mass in g/mol)
    rho : density in g/cm^3

    Returns
    =======
    dEdx : positive value of energy loss per unit distance (MeV/mm)
    """
    Zp = 1
    E = KE + m_p * c**2 * mega * eV
    n = N_A * Z * (rho / centi**3) / A # electron density in m^-3
    
    if I is None:
        I = 10. * Z # eV (mean excitation potential of Block, see wikipedia)
    
    beta2 = 1 - (1 + E * eV * mega / m_p / c**2)**-2  # from relativistic relation: gamma*mp = mp + KE
    Zp *= 1 - np.exp(-125*sqrt(beta2)*Zp**(-2./3)) # from https://de.wikipedia.org/wiki/Bethe-Formel

    constant_term = 2 * pi * n * Zp**2 / m_e / c**2 * (e**2 / 4 / pi / epsilon_0)**2
    
    dEdx = 2 * constant_term / beta2 * (log(2 * m_e*c**2/eV * beta2 / I / (1 - beta2)) - beta2)

    # Advanced formula with further relativistic corrections
    # See also Wm from https://journals.sagepub.com/doi/pdf/10.1093/jicru_os25.2.6
    # dEdx = 2 * constant_term / beta2 * (
    #     log(2 * m_e*c**2/eV * beta2 / I / (1 - beta2))
    #     - 1/2. * log(1 + 2/sqrt(1-beta2)*m_e/m_p + (m_e/m_p)**2) 
    #     - beta2)

    return dEdx / eV / mega * milli


def bethe_bloch_aluminum_semicorrected(KE):
    """A specific correction valid between 1-100 MeV for protons on aluminum
    which explains the difference between my implementation of the bethe formula
    in the bethe_block function, and the comparison values from NIST and wikipedia.

    The correction here removes the ionization potential I, and adds a 5.25 value
    to the logarithmic term of the expression in bethe_bloch.

    Value 5.25 obtained from http://www.srim.org/SRIM/SRIMPICS/IPLOTS/IPLOT13.gif

    This function should be removed once a suitable expression or table of values
    for the shell correction is found. A possible implementation could be produced
    using the publication JETP Letters volume 94, Article number: 1 (2011) 

    A detailed discussion can be found in 
    Ziegler, J. F., Journal of Applied Physics 85, 1249 (1999) while an the actual
    values and formulas used in PSTAR program of NIST are available in
    https://journals.sagepub.com/toc/crub/os-25/2 which is unaccessible to me.

    Arguments:
    ==========
    KE  : Kinetic Energy in MeV

    Returns
    =======
    dEdx : positive value of energy loss per unit distance (MeV/mm)
    """
    Z = 13
    A = 27
    rho = 2.69
    Zp = 1
    E = KE + m_p * c**2 * mega * eV
    n = N_A * Z * (rho / centi**3) / A # electron density in m^-3
    
    beta2 = 1 - (1 + E * eV * mega / m_p / c**2)**-2  # from relativistic relation: gamma*mp = mp + KE
    Zp *= 1 - np.exp(-125*sqrt(beta2)*Zp**(-2./3)) # from https://de.wikipedia.org/wiki/Bethe-Formel

    constant_term = 2 * pi * n * Zp**2 / m_e / c**2 * (e**2 / 4 / pi / epsilon_0)**2
    
    dEdx = 2 * constant_term / beta2 * (log(2 * m_e*c**2/eV * beta2 / (1 - beta2)) - 5.25 - beta2)

    return dEdx / eV / mega * milli


def stoping_power(KE, Z=3., A=6., rho=1., **kwargs):
    """Returns the differential energy loss per thickness
    for protons of kinetic energy KE (not relativistic) in 
    a medium with density rho
    KE = m * v**2 / 2 in MeV units

    https://en.wikipedia.org/wiki/Bethe_formula (non relativistic formula eq(2))
    https://physics.nist.gov/PhysRefData/Star/Text/programs.html

    Arguments:
    ==========
    KE  : Kinetic Energy in MeV
    Z   : Material's atomic number
    A   : Material's mass number (and approximate molar mass in g/mol)
    rho : density in g/cm^3

    Returns
    =======
    dEdx : energy loss per unit distance (MeV/mm)
    """ 
    return bethe_bloch(KE, Z, A, rho, **kwargs) / rho


def CE_range(E0=10, *args, **kwargs):
    """Returns the range assuming a continuos approximation.
    The result is the integration of stoping_power from 0 to an estimated
    maximal thickness. Then finding the thickness t0 for which the integral
    is equal to the total initial energy E0.

    Arguments:
    ----------
    E0: [float], total initial energy of protons in MeV

    Returns:
    --------
    dEdx : energy loss per unit distance (MeV/mm)
    """
    logE0 = np.log10(E0)
    logEcut = np.max([logE0 - 3, -3])

    E = logspace(logEcut, logE0, 10*int(logE0 - logEcut))

    SP = bethe_bloch(E, *args, **kwargs)
    
    return E[1:], cumtrapz(1./SP, E)


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

