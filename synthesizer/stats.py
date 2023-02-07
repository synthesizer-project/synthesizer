import numpy as np


def weighted_mean(data, weights):
    """
    Calculate the weighted mean

    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights

    Author: Stephen Wilkins

    """
    return np.sum(data * weights) / np.sum(weights)


def weighted_median(data, weights):
    """
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights
    """
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx : idx + 2])
        else:
            w_median = s_data[idx + 1]
    return w_median


# Weighted quantiles
def weighted_quantile(
    values, quantiles, sample_weight=None, values_sorted=False, old_style=False
):
    """
    Taken from From https://stackoverflow.com/a/29677616/1718096

    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """

    # do some housekeeping
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(
        quantiles <= 1
    ), "quantiles should be in [0, 1]"

    # if not sorted, sort values array
    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def binned_weighted_quantile(x, y, weights, bins, quantiles):

    # if ~isinstance(quantiles,list):
    #     quantiles = [quantiles]

    out = np.full((len(bins) - 1, len(quantiles)), np.nan)
    for i, (b1, b2) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (x >= b1) & (x < b2)
        if np.sum(mask) > 0:
            out[i, :] = weighted_quantile(
                y[mask], quantiles, sample_weight=weights[mask]
            )

    return np.squeeze(out)


def n_weighted_moment(values, weights, n):

    assert n > 0 & (values.shape == weights.shape)
    w_avg = np.average(values, weights=weights)
    w_var = np.sum(weights * (values - w_avg) ** 2) / np.sum(weights)

    if n == 1:
        return w_avg
    elif n == 2:
        return w_var
    else:
        w_std = np.sqrt(w_var)
        return np.sum(weights * ((values - w_avg) / w_std) ** n) / np.sum(
            weights
        )
        # Same as np.average(((values - w_avg)/w_std)**n, weights=weights)
