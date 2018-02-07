
import scipy
import numpy as np

def dy1(x_vec, msc_mat, alpha):
    """

    Parameters
    ----------
    x_vec
    msc_mat
    alpha

    Returns
    -------

    """
    dy = alpha * (msc_mat[1] * (-np.sin(alpha * x_vec) - np.sinh(alpha * x_vec)) +
                  msc_mat[3] * (np.cos(alpha * x_vec) - np.cosh(alpha * x_vec)))
    return dy


def dy2(x_vec, msc_mat, alpha):
    dy = alpha * (msc_mat[0] * (-np.sin(alpha * x_vec) + np.sinh(alpha * x_vec)) +
                  msc_mat[2] * (np.cos(alpha * x_vec) + np.cosh(alpha * x_vec)))
    return dy


def solve_charac(L, L_1, L_2, f_c1, f_1, Q_c1, Q_1, gamma_ratio, k_C, h, phi, var='kstarcomplex'):
    """

    Parameters
    ----------
    L
    L_1
    L_2
    f_c1
    f_1
    Q_c1
    Q_1
    gamma_ratio
    k_C
    h
    phi
    var

    Returns
    -------

    """

    if var == 'kstarcomplex':
        def solve_func(kstarcomplex):
            return charac(L, L_1, L_2, f_c1, f_1, Q_c1, Q_1, kstarcomplex, gamma_ratio, k_C, h, phi, mode='solve')

    return solve_func


def charac(L, L_1, L_2, f_c1, f_1, Q_c1, Q_1, kstarcomplex, gamma_rat, k_C, h, phi, mode='eval'):
    """
    Calculate the characteristic equation

    Parameters
    ----------
    L
    L_1
    L_2
    f_c1
    f_1
    Q_c1
    Q_1
    kstarcomplex
    gamma_rat
    k_C
    h
    phi
    mode

    Returns
    -------

    """
    if mode == 'solve':
        f = f_c1 / f_1 + 1j * (f_c1 / f_1) / (2 * Q_c1)
    else:
        f = f_c1 / f_1

    # Dispersion Relation
    alpha = (1.8751 / L) * (f**2 - 1j * f / Q_1)**(1/4)

    # Calc trig terms
    trig_dict = calc_trig_relations(L, L_1, L_2, alpha)
    cchfull = trig_dict['cchfull']
    mixplus2 = trig_dict['mixplus2']
    cchminus1 = trig_dict['cchminus1']
    cchplus2 = trig_dict['cchplus2']
    mixplus1 = trig_dict['mixplus1']
    ssh1 = trig_dict['ssh1']
    ssh2 = trig_dict['ssh2']
    mixminus2 = trig_dict['mixminus2']
    mixminus1 = trig_dict['mixminus1']

    # Contact function calculations
    kstar = np.real(kstarcomplex) / k_C
    damp_contact = np.imag(kstarcomplex) * f_1 / f_c1 / k_C

    # Secondary parameter calculation
    k_lat = gamma_rat * kstar

    # Calculation of damping components for the contact functions
    p = (L_1 / L) * (3 * damp_contact) / 1.8751**2
    p_lat = (L_1 / L) * (3 * damp_contact * gamma_rat) / 1.8751**2

    # Calculate contact functions
    cont = 3 * kstar / k_C + 1j * p * (alpha * L_1)**2
    cont_lat = 3 * k_lat / k_C + 1j * p_lat * (alpha * L_1)**2

    # Auxiliary functions
    T = h**2 / L_1**3 * (cont * np.sin(phi)**2 + cont_lat * np.cos(phi)**2)
    X = h * np.sin(phi) * np.cos(phi) * (cont_lat - cont) / L_1**3
    U = (cont * np.cos(phi)**2 + cont_lat * np.sin(phi)**2) / L_1**3

    # Characteristic equation (Denominator of Determinant)
    den_buff = 2 * (-2 * alpha**5 * cchfull +
                    alpha**4 * T * (mixplus2 * cchminus1 - cchplus2 * mixplus1) -
                    2 * alpha**3 * X * (cchplus2 * ssh1 + ssh2 * cchminus1) +
                    alpha**2 * U * (mixminus2 * cchminus1 - cchplus2 * mixminus1) -
                    alpha * (T * U - X**2) * cchplus2 * cchminus1)

    return den_buff


def calc_trig_relations(L, L_1, L_2, alpha):
    """

    Parameters
    ----------
    L
    L_1
    L_2
    alpha

    Returns
    -------
    trig_dict

    """
    splus1 = np.sin(alpha * L_1) + np.sinh(alpha * L_1)
    splus2 = np.sin(alpha * L_2) + np.sinh(alpha * L_2)
    sminus1 = np.sin(alpha * L_1) - np.sinh(alpha * L_1)
    sminus2 = np.sin(alpha * L_2) - np.sinh(alpha * L_2)
    cplus1 = np.cos(alpha * L_1) + np.cosh(alpha * L_1)
    cplus2 = np.cos(alpha * L_2) + np.cosh(alpha * L_2)
    cminus1 = np.cos(alpha * L_1) - np.cosh(alpha * L_1)
    cminus2 = np.cos(alpha * L_2) - np.cosh(alpha * L_2)
    ssh1 = np.sin(alpha * L_1) * np.sinh(alpha * L_1)
    ssh2 = np.sin(alpha * L_2) * np.sinh(alpha * L_2)
    cch1 = np.cos(alpha * L_1) * np.cosh(alpha * L_1)
    cch2 = np.cos(alpha * L_2) * np.cosh(alpha * L_2)
    cchplus1 = 1 + cch1
    cchplus2 = 1 + cch2
    cchminus1 = 1 - cch1
    cchminus2 = 1 - cch2
    cchfull = 1 + np.cos(alpha * L) * np.cosh(alpha * L)
    mixplus1 = np.sin(alpha * L_1) * np.cosh(alpha * L_1) + np.cos(alpha * L_1) * np.sinh(alpha * L_1)
    mixplus2 = np.sin(alpha * L_2) * np.cosh(alpha * L_2) + np.cos(alpha * L_2) * np.sinh(alpha * L_2)
    mixminus1 = np.sin(alpha * L_1) * np.cosh(alpha * L_1) - np.cos(alpha * L_1) * np.sinh(alpha * L_1)
    mixminus2 = np.sin(alpha * L_2) * np.cosh(alpha * L_2) - np.cos(alpha * L_2) * np.sinh(alpha * L_2)

    trig_dict = {'splus1': splus1,
                 'splus2': splus2,
                 'sminus1': sminus1,
                 'sminus2': sminus2,
                 'cplus1': cplus1,
                 'cplus2': cplus2,
                 'cminus1': cminus1,
                 'cminus2': cminus2,
                 'ssh1': ssh1,
                 'ssh2': ssh2,
                 'cch1': cch1,
                 'cch2': cch2,
                 'cchplus1': cchplus1,
                 'cchplus2': cchplus2,
                 'cchminus1': cchminus1,
                 'cchminus2': cchminus2,
                 'cchfull': cchfull,
                 'mixplus1': mixplus1,
                 'mixplus2': mixplus2,
                 'mixminus1': mixminus1,
                 'mixminus2': mixminus2}

    return trig_dict