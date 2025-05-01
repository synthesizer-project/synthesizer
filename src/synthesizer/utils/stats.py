"""Statistical functions for weighted means, medians, and quantiles.

This module provides functions to calculate weighted means, medians, and
quantiles. All of these are helper wrappers around existing numpy
functionality.

Example usage:
    from synthesizer.utils.stats import (
        weighted_mean,
        weighted_median,
        weighted_quantile,
        binned_weighted_quantile,
    )

    data = [1, 2, 3, 4, 5]
    weights = [0.1, 0.2, 0.3, 0.4, 0.5]
    mean = weighted_mean(data, weights)
    median = weighted_median(data, weights)
    quantiles = weighted_quantile(
        data, [0.25, 0.5, 0.75],
        sample_weight=weights,
    )
    binned_quantiles = binned_weighted_quantile(
        data, data, weights, bins=[0, 2, 4, 6], quantiles=[0.25, 0.5]
    )
"""

import numpy as np


def weighted_mean(
    data: np.ndarray | list,
    weights: np.ndarray | list,
) -> float:
    """Calculate the weighted mean.

    This is just a helpful alias around np.average which provides a weighted
    mean more efficient than using a combination of np.sum and np.mean.

    Args:
      data (list or np.ndarray):
          The data to calculate the mean of.
      weights (list or np.ndarray):
          The weights to apply to the data.

    Returns:
        float: The weighted mean.
    """
    return np.average(data, weights=weights)


def weighted_median(
    data: np.ndarray | list,
    weights: np.ndarray | list,
):
    """Calculate the weighted median.

    Args:
        data (list or numpy.array):
            The data to calculate the median of.
        weights (list or numpy.array):
            The weights to apply to the data.
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


def weighted_quantile(
    values: np.ndarray | list,
    quantiles: np.ndarray | list,
    sample_weight: np.ndarray | list = None,
    values_sorted: bool = False,
    old_style: bool = False,
) -> np.ndarray:
    """Calculate a weighted quantile.

    Taken from From https://stackoverflow.com/a/29677616/1718096.

    Very close to numpy.percentile, but supports weights.

    Args:
        values (np.ndarray or list):
            The values to compute the quantiles of.
        quantiles (np.ndarray or list):
            The quantiles to compute. Must be in [0, 1].
        sample_weight (np.ndarray or list):
            The weights to apply to the values.
        values_sorted (bool):
            If True, then values will not be sorted before the calculation.
        old_style (bool):
            If True, then the computed quantiles will be returned in the same
            style as numpy.percentile.

    Returns:
        np.ndarray: The computed quantiles.
    """
    # do some housekeeping
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), (
        "quantiles should be in [0, 1]"
    )

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


def binned_weighted_quantile(
    x: np.ndarray | list,
    y: np.ndarray | list,
    weights: np.ndarray | list,
    bins: np.ndarray | list,
    quantiles: np.ndarray | list,
) -> np.ndarray:
    """Calculate the weighted quantiles of y in bins of x.

    Args:
        x (np.ndarray or list):
            The x values to bin by.
        y (np.ndarray or list):
            The y values to calculate the quantiles of.
        weights (np.ndarray or list):
            The weights to apply to the y values.
        bins (np.ndarray or list):
            The bins to use for the x values.
        quantiles (np.ndarray or list):
            The quantiles to calculate.

    Returns:
        np.ndarray: The weighted quantiles of y in the bins of x.
    """
    out = np.full((len(bins) - 1, len(quantiles)), np.nan)
    for i, (b1, b2) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (x >= b1) & (x < b2)
        if np.sum(mask) > 0:
            out[i, :] = weighted_quantile(
                y[mask], quantiles, sample_weight=weights[mask]
            )

    return np.squeeze(out)


def n_weighted_moment(
    values: np.ndarray | list,
    weights: np.ndarray | list,
    n: int,
) -> float:
    """Calculate the n-th weighted moment of the values.

    Args:
        values (np.ndarray or list):
            The values to calculate the moment of.
        weights (np.ndarray or list):
            The weights to apply to the values.
        n (int):
            The order of the moment to calculate.

    Returns:
        float: The n-th weighted moment of the values.
    """
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
