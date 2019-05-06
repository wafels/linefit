#
# Intensity respecting line fitting
#
# Use the emcee
#
import numpy as np
from numpy.random import poisson
from scipy.integrate import quad
import matplotlib.pyplot as plt
import emcee



def emission_model_at_sun(x, a, p, w):
    """

    :param x:
    :param a:
    :param p:
    :param w:
    :return:
    """
    onent = (x-p) / w
    normalize = 1.0 / (np.sqrt(2*np.pi) * w)
    return a * normalize * np.exp(-0.5*onent**2)


def emission_model_at_sun_simple_gaussian(x, a, p, w):
    """

    :param x:
    :param a:
    :param p:
    :param w:
    :return:
    """
    onent = (x-p) / w
    return a * np.exp(-0.5*onent**2)


def integrate_across_one_bin(lhs, rhs, args):
    answer = quad(emission_model_at_sun_simple_gaussian, lhs, rhs, args=args)
    return answer[0]


def integral_across_all_bins(bins_edges, args):
    """

    :param bins:
    :param args:
    :return:
    """
    integral = []
    for i in range(0, bin_edges.shape[1]):
        lhs = bins_edges[0, i]
        rhs = bins_edges[1, i]
        integral.append(integrate_across_one_bin(lhs, rhs, args))
    return np.asarray(integral)


def equally_spaced_bins(inner_value=1, outer_value=2, nbins=100):
    """
    Define a set of equally spaced bins between the specified inner and outer
    values.  The inner value must be strictly less than the outer value.
    Parameters
    ----------
    inner_value : ``float`
        The inner value of the bins.
    outer_value : ``float`
        The outer value of the bins.
    nbins : ``int`
        Number of bins
    Returns
    -------
    An array of shape (2, nbins) containing the bin edges.
    """
    if inner_value >= outer_value:
        raise ValueError('The inner value must be strictly less than the outer value.')

    if nbins <= 0:
        raise ValueError('The number of bins must be strictly greater than 0.')

    bin_edges = np.zeros((2, nbins))
    bin_edges[0, :] = np.arange(0, nbins)
    bin_edges[1, :] = np.arange(1, nbins+1)

    return inner_value + bin_edges * (outer_value - inner_value) / nbins


# Log of the uniform prior
def ln_uniform(theta, lower, upper):
    if lower < theta < upper:
        return 0.0
    return -np.inf

# Log of the Poisson
def ln_poisson(mean):
    

#
# Create some fake data
#
# wavelength range
fx_start = -10
fx_end = 10
fnum = 23
bin_edges = equally_spaced_bins(inner_value=fx_start, outer_value=fx_end, nbins=fnum)


# properties of the emission at the Sun
fa = 1000
fp = 0
fw = 5

# Make the noisy data
real = integral_across_all_bins(bin_edges, (fa, fp, fw))
observed = poisson(real)

plt.plot(real)
plt.plot(observed)
plt.show()


