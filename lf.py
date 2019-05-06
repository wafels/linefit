#
# line fitting program
#
import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln
import pymc


def ln_poisson(lam, k):
    """
    Log of the Poisson distribution
    :param lam:
    :param k:
    :return: log of the Poisson distribution
    """
    return k*np.log(lam) - lam - gammaln(k+1)



def integrand(x, c, a, p, w):
    onent = (x-p) / w
    normalize = 1.0 / (np.sqrt(2*np.pi) * w)
    return c + a * normalize * np.exp(-0.5*onent**2)


def integrate_across_one_bin(lhs, rhs, args):
    answer = quad(integrand, lhs, rhs, args=args)
    return answer[0]


def integral_across_all_bins(bins, args):
    integral = []
    for bin_edges in bins:
        integral.append(integrate_across_one_bin(bin_edges[0], bin_edges[1], args))
    return np.asarray(integral)


# -----------------------------------------------------------------------------
# Gaussian plus a constant
#
def gaussian_plus_constant(bins, observed_counts_per_bin, init=None):
    """Assumes the line can be modeled using a constant and a Gaussian
    """
    if init is None:
        constant = pymc.Uniform('constant',
                                lower=0.0,
                                upper=6.0,
                                doc='constant')

        amplitude = pymc.Uniform('amplitude',
                                 lower=0.0,
                                 upper=10.0,
                                 doc='amplitude')

        position = pymc.Uniform('position',
                                lower=-20.0,
                                upper=10.0,
                                doc='position')

        width = pymc.Uniform('width',
                             lower=-20.0,
                             upper=10.0,
                             doc='width')
    else:
        raise ValueError('Not implemented yet')

    # Model for the emission line
    @pymc.deterministic(plot=False)
    def modeled_emission(c=constant,
                         a=amplitude,
                         p=position,
                         w=width,
                         bins=bins):
        # A pure and simple power law model
        out = integral_across_all_bins(bins, (c, a, p, w))
        return out

    #
    @pymc.potential
    def constrain_total_emission():
        total_observed_emission = np.sum(observed_counts_per_bin)
        total_fit_emission = np.sum(modeled_emission(c=constant,
                         a=amplitude,
                         p=position,
                         w=width,
                         bins=bins))
        return

    spectrum = pymc.Poisson('emission',
                            mu=modeled_emission,
                            value=observed_counts_per_bin,
                            observed=True)

    # Need to add in the potential constraint

    predictive = pymc.Poisson('predictive',
                              mu=modeled_emission)
    # MCMC model
    return locals()
