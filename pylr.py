import matplotlib.pyplot as plt
import numpy
from scipy import stats as sts


def tippett_plot(lrs_on_target, lrs_off_target):

    lr_sorted_on_target = (sorted(lrs_on_target))  # sort for nice plotting
    lr_sorted_off_target = (sorted(lrs_off_target))

    num_off_target_compared = len(lr_sorted_off_target)
    cumulative_proportion_off_target = [x * 1.0 / num_off_target_compared for x in range(0, len(lr_sorted_off_target))]

    num_on_target_compared = len(lr_sorted_on_target)
    cumulative_proportion_on_target = [x * 1.0 / num_on_target_compared for x in range(0, len(lr_sorted_on_target))]

    plt.ylim((-0.1, 1.1))
    plt.plot(lr_sorted_off_target[::-1], cumulative_proportion_off_target, 'b-', label='Off target group')
    # [::-1] #reverse off-target array for nice plot
    plt.plot(lr_sorted_on_target, cumulative_proportion_on_target, 'r-', label='On target group')
    plt.plot((1.0, 1.0), (-0.1, 1.1), 'g-')
    plt.legend(loc='upper right')
    plt.ylabel('proportion of observed LR values in test set')
    plt.xlabel('log10 LR value for On-target/Off-target group membership')
    plt.show()


def neg_log_sig(log_odds):
    neg_log_odds = [-1.0 * x for x in log_odds]
    e = numpy.exp(neg_log_odds)
    return [numpy.log(1 + f) for f in e if f < (f + 1)]


def cllr(lrs_on_target, lrs_off_target):
    # based on Niko Brummer's original implementation:
    # Niko Brummer and Johan du Preez, Application-Independent Evaluation of Speaker Detection"
    # Computer Speech and Language, 2005
    lrs_on_target = numpy.log(lrs_on_target[~numpy.isnan(lrs_on_target)])
    lrs_off_target = numpy.log(lrs_off_target[~numpy.isnan(lrs_off_target)])

    c1 = numpy.mean(neg_log_sig(lrs_on_target)) / numpy.log(2)
    c2 = numpy.mean(neg_log_sig(-1.0 * lrs_off_target)) / numpy.log(2)
    return (c1 + c2) / 2


def fit_parametric(data, dist_type):
    scipy_dists = ['beta', 'binom', 'cauchy', 'chi2', 'expon', 'f', 'gamma', 'geom', 'hypergeom', 'invgamma', 'lognorm',
                   'logistic', 'nbinom', 'norm', 'poisson', 'unif', 't', 'exponweib']

    if dist_type not in scipy_dists:
        raise ValueError('parametric density function must be one of: %s' % scipy_dists)

    mydist = getattr(sts, dist_type)  # instantiate an empty distribution object of type scipy.stats.$dist_type
    param = mydist.fit(data)
    mydist = getattr(sts, dist_type)([i for i in param])
    #  this is broken,
    #  must disambiguate between distributions that require multiple parameters and
    #  multivariate parameters passed to a distribution of higher than dimension 1
    return mydist

# TODO: add goodness of fit metrics iff specific distributions are found and warnign otherwise:
# jarque_bera(x)
# see https://docs.scipy.org/doc/scipy/reference/stats.html

version = '0.1'
