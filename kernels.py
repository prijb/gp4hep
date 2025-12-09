# Import kernels functions from here
import numpy as np
import numba
import scipy

# Scale function 
def scale_func(x, b=5e-2, c=30.0):
    return b*x + c

# Simplest kernel (RBF)
# Takes N x M arrays where N is the number of points and M is the dimensionality of each point
def exponentiated_quadratic(xa, xb, variance, scale):
    """Exponentiated quadratic  with K=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')/scale**2
    return variance*np.exp(sq_norm)

# Modified Gibbs kernel 
def modified_gibbs(xa, xb, variance, a, b, c, d):
    scale_a = scale_func(xa, b, c)
    scale_b = scale_func(xb, b, c)

    # Operations between x values
    cdist = scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    xa_broadcast = xa[:, None, :]
    xb_broadcast = xb[None, :, :]
    x_sum = xa_broadcast + xb_broadcast

    # Operations between scales
    scale_a = scale_a[:, None, :]
    scale_b = scale_b[None, :, :]
    scale_quadrature = scale_a**2 + scale_b**2
    scale_diff = scale_a - scale_b
    scale_product = scale_a*scale_b

    # Shape down to 2D (similar dimension to cdist)
    reshape_shape = (cdist.shape[0], cdist.shape[1])
    x_sum = x_sum.reshape(*reshape_shape)
    scale_quadrature = scale_quadrature.reshape(*reshape_shape)
    scale_diff = scale_diff.reshape(*reshape_shape)
    scale_product = scale_product.reshape(*reshape_shape)

    # Calculate the kernel
    amplitude = variance*np.exp((d - x_sum)/(2*a))
    scale_term = np.sqrt(2*scale_product/scale_quadrature)
    exp_term = np.exp(-cdist/scale_quadrature)

    return amplitude*scale_term*exp_term