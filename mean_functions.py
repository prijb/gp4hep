# Import mean functions from here
import numpy as np
import numba

# Function
@numba.njit
def DSCB_pdf(x, a, mu, sigma, n_low, alpha_low, n_high, alpha_high):
    """
    Double sided Crystal-Ball
    
    https://arxiv.org/abs/1606.03833
    
    [floating normalization]
    
    Args:
        par: mu > 0, sigma > 0, n_low > 1, alpha_low > 0, n_high > 1, alpha_high > 0
    """
    # Piece wise definition
    y = np.zeros(len(x))
    t = (x - mu) / max(sigma, 1E-12)
    
    ind_0 = (-alpha_low <= t) & (t <= alpha_high)
    ind_1 = t < -alpha_low
    ind_2 = t >  alpha_high

    y[ind_0] = np.exp(- 0.5 * t[ind_0]**2)
    y[ind_1] = np.exp(- 0.5 * alpha_low**2)  * (alpha_low / n_low   * (n_low / alpha_low   - alpha_low  - t[ind_1]))**(-n_low)
    y[ind_2] = np.exp(- 0.5 * alpha_high**2) * (alpha_high / n_high * (n_high / alpha_high - alpha_high + t[ind_2]))**(-n_high)
    
    return a * y

@numba.njit
def mod_exp_simpson(x, p0, p1, p2, p3, p4):
    dx = (x[1] - x[0]) * 0.5
    h = (x[1] - x[0])/3
    xa = x - dx
    xb = x - dx + h
    xc = x - dx + 2*h
    xd = x + dx

    xa = xa / 13600
    xb = xb / 13600
    xc = xc / 13600
    xd = xd / 13600

    ya = p0 * np.exp((p1 * (xa ** p2)) + (p3 * ((1 - xa) ** p4)))
    yb = p0 * np.exp((p1 * (xb ** p2)) + (p3 * ((1 - xb) ** p4)))
    yc = p0 * np.exp((p1 * (xc ** p2)) + (p3 * ((1 - xc) ** p4)))
    yd = p0 * np.exp((p1 * (xd ** p2)) + (p3 * ((1 - xd) ** p4)))

    y = (1.0/8.0)*(ya + (3*yb) + (3*yc) + yd)

    return y

@numba.njit
def poly_ext_simpson(x, p0, p1, p2, p3, p4):
    dx = (x[1] - x[0]) * 0.5
    h = (x[1] - x[0])/3
    xa = x - dx
    xb = x - dx + h
    xc = x - dx + 2*h
    xd = x + dx

    xa = xa / 13600
    xb = xb / 13600
    xc = xc / 13600
    xd = xd / 13600

    ya = p0 * ((1 - xa) ** p1) * (1 + (p4*xa)) * (xa ** -(p2 + p3 * np.log(xa)))
    yb = p0 * ((1 - xb) ** p1) * (1 + (p4*xb)) * (xb ** -(p2 + p3 * np.log(xb)))
    yc = p0 * ((1 - xc) ** p1) * (1 + (p4*xa)) * (xa ** -(p2 + p3 * np.log(xc)))
    yd = p0 * ((1 - xd) ** p1) * (1 + (p4*xa)) * (xa ** -(p2 + p3 * np.log(xd)))

    y = (1.0/8.0)*(ya + (3*yb) + (3*yc) + yd)
    return y