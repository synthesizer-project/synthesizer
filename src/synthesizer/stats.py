"""A module containing statistical functions.

This module contains functions for calculating weighted means, medians,
quantiles, and moments.

Example Usage:
    weighted_mean(data, weights)
    weighted_median(data, weights)
    weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False)
    binned_weighted_quantile(x, y, weights, bins, quantiles)
    n_weighted_moment(values, weights, n)
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def weighted_mean(
    data: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> np.float64:
    """
    Calculate the weighted mean.

    Args:
        data: The data
        weights: The weights

    Returns:
        The weighted mean.
    """
    return np.sum(data * weights) / np.sum(weights)


def weighted_median(
    data: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> np.float64:
    """
    Calculate the weighted median.

    Args:
        data: The data
        weights: The weights

    Returns:
        The weighted median.
    """
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint: float = 0.5 * sum(s_weights)

    w_median: np.float64
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights: NDArray[np.float64] = np.cumsum(s_weights)
        idx: int = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx : idx + 2])
        else:
            w_median = s_data[idx + 1]
    return w_median


def weighted_quantile(
    values: NDArray[np.float64],
    quantiles: NDArray[np.float64],
    sample_weight: Optional[NDArray[np.float64]] = None,
    values_sorted: bool = False,
    old_style: bool = False,
) -> NDArray[np.float64]:
    """
    Compute quantiles of a weighted array.

    Taken from From https://stackoverflow.com/a/29677616/1718096

    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!

    Args:
        values: The values to weight
        quantiles: The array of quantiles needed
        sample_weight: The weights (same length as `array`)
        values_sorted: If True, then will avoid sorting of initial array
        old_style: If True, will correct output to be consistent
                   with numpy.percentile.
    Returns:
        The computed quantiles.
    """
    # Do some housekeeping
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(
        quantiles <= 1
    ), "quantiles should be in [0, 1]"

    # If not sorted, sort values array
    if not values_sorted:
        sorter: NDArray[np.int32] = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles: NDArray[np.float64] = (
        np.cumsum(sample_weight) - 0.5 * sample_weight
    )
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def binned_weighted_quantile(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    weights: NDArray[np.float64],
    bins: NDArray[np.float64],
    quantiles: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate the weighted quantiles of y in bins of x.

    Args:
        x: The x values
        y: The y values
        weights: The weights
        bins: The bins
        quantiles: The quantiles

    Returns:
        The weighted quantiles of y in bins of x.
    """
    out: NDArray[np.float64] = np.full((len(bins) - 1, len(quantiles)), np.nan)
    for i, (b1, b2) in enumerate(zip(bins[:-1], bins[1:])):
        mask: NDArray[np.bool_] = (x >= b1) & (x < b2)
        if np.sum(mask) > 0:
            out[i, :] = weighted_quantile(
                y[mask], quantiles, sample_weight=weights[mask]
            )

    return np.squeeze(out)


def n_weighted_moment(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    n: int,
) -> np.float64:
    """
    Calculate the weighted nth moment of the values.

    Args:
        values: The values
        weights: The weights
        n: The moment

    Returns:
        The weighted nth moment.
    """
    assert n > 0 & (values.shape == weights.shape)
    w_avg: np.float64 = np.average(values, weights=weights)
    w_var: np.float64 = np.sum(weights * (values - w_avg) ** 2) / np.sum(
        weights
    )

    if n == 1:
        return w_avg
    elif n == 2:
        return w_var
    else:
        w_std: np.float64 = np.sqrt(w_var)
        return np.sum(weights * ((values - w_avg) / w_std) ** n) / np.sum(
            weights
        )
