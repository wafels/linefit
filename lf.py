#
# line fitting program
#
import numpy as np
import pymc

def line_profile(x, a):
    f = a[0]
    return f

def integral_of_gaussian_plus_constant(lambda_bins, parameters):



# -----------------------------------------------------------------------------
# Gaussian plus a constant
#
def Log_splwc(bins, observed_counts_per_bin, init=None):
    """Power law with a constant.  This model assumes that the power
    spectrum is made up of a power law and a constant background.  At high
    frequencies the power spectrum is dominated by the constant background.
    """
    if init is None:
        constant = pymc.Uniform('constant',
                                lower=0.0,
                                upper=6.0,
                                doc='power law index')

        amplitude = pymc.Uniform('power_law_norm',
                                lower=0.0,
                                upper=10.0,
                                doc='power law normalization')

        position = pymc.Uniform('position',
                                lower=-20.0,
                                upper=10.0,
                                doc='background')

        width = pymc.Uniform('width',
                             lower=-20.0,
                             upper=10.0,
                             doc='background')
    else:
        constant = pymc.Uniform('constant',
                                lower=0.0,
                                upper=6.0,
                                doc='power law index')

        amplitude = pymc.Uniform('power_law_norm',
                                lower=0.0,
                                upper=10.0,
                                doc='power law normalization')

        position = pymc.Uniform('position',
                                lower=-20.0,
                                upper=10.0,
                                doc='background')

        width = pymc.Uniform('width',
                             lower=-20.0,
                             upper=10.0,
                             doc='background')

    # Total counts
    total_counts = np.sum(observed_counts_per_bin)

    # Model for the power law spectrum
    @pymc.deterministic(plot=False)
    def modeled_emission(c=constant,
                         a=amplitude,
                         p=position,
                         w=width,
                         bins=bins):
        # A pure and simple power law model#
        out = integral_of_gaussian_plus_constant(bins, [c, a, p, w])
        return out

    spectrum = pymc.Poisson('spectrum',
                            mu=modeled_emission,
                            value=observed_counts_per_bin,
                            observed=True)

    predictive = pymc.Poisson('predictive',
                              mu=modeled_emission)
    # MCMC model
    return locals()