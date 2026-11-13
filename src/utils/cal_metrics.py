import numpy as np
from scipy.special import ndtr, ndtri
from scipy.optimize import root_scalar


class GaussianMixtureDistributionNumpy:
    def __init__(self, means: np.array, scales: np.array, weights: np.array):
        self.means = means
        self.scales = scales
        self.weights = weights
        self.K = len(means)


    def objective(self, x, p):
        return np.inner(self.weights, ndtr((x - self.means) / self.scales)) - p # CDF - p


    def optim_range(self, p):
        quantiles = ndtri(p)*self.scales + self.means
        a, b = np.min(quantiles), np.max(quantiles)
        return (a + b) / 2, [a, b]


    def ppf(self, p):
        if self.K == 1:
            return ndtri(p)*self.scales + self.means

        x0, bracket = self.optim_range(p)
        obj_func = lambda x: self.objective(x, p)
        try:
            r = root_scalar(
                f=obj_func,
                fprime=False,
                fprime2=False,
                bracket=bracket,
                x0=x0,
            )
            assert r.converged
            return r.root
        except ValueError:
            x_vals = [bracket[0], x0, bracket[1]]
            obj_func_vals = np.array([obj_func(bracket[0]), obj_func(x0), obj_func(bracket[1])])
            idx_min = np.argmin(np.abs(obj_func_vals))
            return x_vals[idx_min]



def regression_ece_for_gaussian_mixture(
    means: np.ndarray,
    scales: np.ndarray,
    targets: np.ndarray,
    probs: np.ndarray = np.linspace(0.01, 0.99, 20),
) -> float:
    """
    Computes the regression ECE for a gaussian mixture predictive distribution
    with uniform weights.

    Args:
        means: np.ndarray of shape S x N of predictive means.
        scales: np.ndarray of shape S x N of predictive standard deviations.
        targets: np.ndarray of shape (N,) with prediction targets.
        probs: np.ndarray with probabilities to compute percentiles for.

    Returns:
        The regression ECE as a float.
    """
    S, N = means.shape
    weights = np.ones(S)/S

    # Compute observed probabilities
    observed_probs = np.zeros_like(probs)
    for n in range(N):
        target = targets[n]
        m, s = means[:,n], scales[:,n]
        gmm = GaussianMixtureDistributionNumpy(means=m, scales=s, weights=weights)

        for prob_idx, prob in enumerate(probs):
            percentile = gmm.ppf(prob)
            if target <= percentile:
                observed_probs[prob_idx:] += 1
                break

    observed_probs /= N
    ece = np.abs(observed_probs - probs).mean()

    return ece


def expected_calibration_error_regression(means, scales, targets, num_bins=15):
    """
    Calculates the Expected Calibration Error (ECE) for regression tasks
    assuming a Gaussian distribution.

    Args:
        means (np.array): Predicted means of shape (N,).
        scales (np.array): Predicted standard deviations (uncertainties) of shape (N,).
        targets (np.array): Ground truth target values of shape (N,).
        num_bins (int): Number of bins to use for the calibration curve.

    Returns:
        float: The calculated ECE score.
    """
    from scipy.stats import norm

    # Ensure inputs are numpy arrays
    means = np.asarray(means)
    scales = np.asarray(scales)
    targets = np.asarray(targets)

    # 1. Calculate the probability integral transform (PIT) values
    #    This tells us where the true target falls in the predicted CDF.
    #    If perfectly calibrated, these should be Uniformly distributed [0, 1].
    pit_values = norm.cdf(targets, loc=means, scale=scales)

    # 2. define confidence levels (p)
    #    We check calibration for range of quantile levels (e.g. 0.1, 0.2, ... 1.0)
    p_levels = np.linspace(0, 1, num_bins + 1)[1:]  # Exclude 0, keep 1.0

    ece = 0.0

    # 3. Compute the empirical frequency for each confidence level
    #    In a perfectly calibrated model, the fraction of PIT values <= p should be exactly p.
    for p in p_levels:
        # Theoretical fraction (expected coverage)
        expected_coverage = p

        # Empirical fraction (observed coverage)
        # Count how many targets fell within the 'p' quantile of their predicted distributions
        observed_coverage = np.mean(pit_values <= p)

        # Accumulate the absolute difference
        ece += np.abs(observed_coverage - expected_coverage)

    # 4. Average over the number of levels (bins)
    return ece / num_bins
