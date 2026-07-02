# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

"""
Lightweight, dependency-free array primitives.

These helpers depend only on :mod:`numpy` (no xarray, netCDF4 or shapely), so
they can be used by the CF layer (:mod:`ascat.cf_conversions`,
:mod:`ascat.ragged_array`) without pulling in heavier dependencies. The project
sentinel fill values live here and are recognised by
:func:`ascat.utils.mask_dtype_nans`.
"""

import numpy as np

int8_nan = np.iinfo(np.int8).min
uint8_nan = np.iinfo(np.uint8).max
int16_nan = np.iinfo(np.int16).min
uint16_nan = np.iinfo(np.uint16).max
int32_nan = np.iinfo(np.int32).min
uint32_nan = np.iinfo(np.uint32).max
int64_nan = np.iinfo(np.int64).min
uint64_nan = np.iinfo(np.uint64).max
float32_nan = -999999.
float64_nan = -999999.

dtype_to_nan = {
    np.dtype('int8'): int8_nan,
    np.dtype('uint8'): uint8_nan,
    np.dtype('int16'): int16_nan,
    np.dtype('uint16'): uint16_nan,
    np.dtype('int32'): int32_nan,
    np.dtype('uint32'): uint32_nan,
    np.dtype('int64'): int64_nan,
    np.dtype('uint64'): uint64_nan,
    np.dtype('float32'): float32_nan,
    np.dtype('float64'): float64_nan,
    np.dtype('<U1'): None,
    np.dtype('O'): None,
}


def fill_value(dtype):
    """
    Return the fill value used to pad missing elements of a given dtype.

    Uses the project-wide sentinel fills from :data:`dtype_to_nan` (so they are
    recognised by :func:`ascat.utils.mask_dtype_nans`), and ``NaT`` for
    datetimes.

    Parameters
    ----------
    dtype : numpy.dtype
        Data type.

    Returns
    -------
    fill : scalar
        Fill value for the given dtype.
    """
    if np.issubdtype(dtype, np.datetime64):
        return np.datetime64("NaT")
    return dtype_to_nan[dtype]


def vrange(starts, stops):
    """
    Create concatenated ranges of integers for multiple start/stop values.

    Parameters
    ----------
    starts : numpy.ndarray
        Starts for each range.
    stops : numpy.ndarray
        Stops for each range (same shape as starts).

    Returns
    -------
    ranges : numpy.ndarray
        Concatenated ranges.

    Example
    -------
        >>> vrange(np.array([1, 3, 4, 6]), np.array([1, 5, 7, 6]))
        array([3, 4, 4, 5, 6])
    """
    starts = np.asarray(starts)
    stops = np.asarray(stops)
    if starts.shape != stops.shape:
        raise ValueError("starts and stops must have the same shape")
    lengths = stops - starts  # lengths of each range
    return np.repeat(stops - lengths.cumsum(),
                     lengths) + np.arange(lengths.sum())


def pad_to_2d(var, x, y, shape):
    """
    Scatter a 1d array into a padded 2d array.

    Parameters
    ----------
    var : xarray.DataArray or numpy.ndarray
        1d array to be placed into a 2d array.
    x : numpy.ndarray
        Row indices.
    y : numpy.ndarray
        Column indices.
    shape : tuple
        Shape of the output array.

    Returns
    -------
    padded : numpy.ndarray
        2d array, with missing elements set to the dtype fill value.
    """
    values = getattr(var, "values", var)
    padded = np.full(shape, fill_value(values.dtype), dtype=values.dtype)
    padded[x, y] = values
    return padded
