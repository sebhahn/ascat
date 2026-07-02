# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

"""
Conversions between CF discrete sampling geometry representations.

These are plain functions operating on :class:`xarray.Dataset` objects with the
relevant dimension/variable names passed explicitly. They do not depend on the
wrapper classes in :mod:`ascat.cf_array` or :mod:`ascat.ragged_array`, so they
can be used directly on any dataset that follows the CF conventions.

Representations
---------------
- point            : one sample dimension, no grouping.
- indexed ragged   : an index variable maps each sample to an instance
  (CF 9.3.4).
- contiguous ragged: a count variable gives each instance's number of samples
  (CF 9.3.3).
- orthogonal multidim : a dense (instance x element) array sharing one element
  axis, complete/unpadded (CF 9.3.1).
- incomplete multidim : a dense (instance x element) array with a positional
  element axis, padded with fill values (CF 9.3.2).
"""

from __future__ import annotations

from typing import Union, Sequence

import numpy as np
import xarray as xr

from ascat.array_utils import fill_value, vrange


_GEOMETRY_STANDARD_NAMES = {"longitude", "latitude", "time", "height",
                            "altitude"}
_GEOMETRY_NAMES = {"lon", "longitude", "lat", "latitude", "time",
                   "alt", "altitude", "height"}


def _geometry_coordinates(ds: xr.Dataset) -> list:
    """Names of the spatiotemporal coordinate variables (lon/lat/time/alt)."""
    coords = []
    for v in ds.variables:
        name = str(v).lower()
        standard_name = ds[v].attrs.get("standard_name", "")
        if standard_name in _GEOMETRY_STANDARD_NAMES or name in _GEOMETRY_NAMES:
            coords.append(str(v))
    return coords


def finalize_cf(
    ds: xr.Dataset,
    feature_type: Union[str, None] = "timeSeries",
    instance_id_var: Union[str, None] = None,
    coordinates: Union[Sequence[str], None] = None,
) -> xr.Dataset:
    """
    Fill in CF discrete-sampling-geometry metadata on a dataset.

    Sets the global ``featureType`` attribute, marks the instance identifier
    with ``cf_role="timeseries_id"`` (for non-point feature types), and gives
    each measurement variable a ``coordinates`` attribute listing the
    spatiotemporal coordinate variables. Existing ``cf_role`` / ``coordinates``
    attributes are kept. The input dataset is not modified.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in any CF discrete sampling geometry representation.
    feature_type : str or None, optional
        Value for the global ``featureType`` attribute (default "timeSeries";
        None to leave it unchanged).
    instance_id_var : str, optional
        Name of the instance identifier variable to mark with
        ``cf_role="timeseries_id"``.
    coordinates : sequence of str, optional
        Coordinate variable names for the ``coordinates`` attribute. If None,
        they are auto-detected (lon/lat/time/alt by name or standard_name).

    Returns
    -------
    ds : xarray.Dataset
        A copy with the CF metadata filled in.
    """
    ds = ds.copy()
    ds.attrs = dict(ds.attrs)
    if feature_type is not None:
        ds.attrs["featureType"] = feature_type

    if (instance_id_var is not None and instance_id_var in ds.variables
            and feature_type != "point"):
        attrs = dict(ds[instance_id_var].attrs)
        attrs.setdefault("cf_role", "timeseries_id")
        ds[instance_id_var].attrs = attrs

    if coordinates is None:
        coordinates = _geometry_coordinates(ds)
    if coordinates:
        coord_str = " ".join(coordinates)
        # dimensions along which observations vary (the time axis); used to give
        # the coordinates attribute to measurement variables, not per-instance
        # metadata (which does not vary with time)
        obs_dims = set()
        for c in coordinates:
            if (str(c).lower() == "time"
                    or ds[c].attrs.get("standard_name") == "time"):
                obs_dims.update(ds[c].dims)
        skip = set(coordinates)
        if instance_id_var is not None:
            skip.add(instance_id_var)
        for v in ds.data_vars:
            attrs = ds[v].attrs
            # skip coordinate, identifier and structural (count/index) variables,
            # and variables that already declare coordinates (in attrs or, as set
            # by a NetCDF backend, in encoding)
            if (v in skip or "sample_dimension" in attrs
                    or "instance_dimension" in attrs or "cf_role" in attrs
                    or "coordinates" in attrs
                    or "coordinates" in ds[v].encoding):
                continue
            if obs_dims and not (set(ds[v].dims) & obs_dims):
                continue
            new_attrs = dict(attrs)
            new_attrs["coordinates"] = coord_str
            ds[v].attrs = new_attrs
    return ds


def detect_cf_representation(ds: xr.Dataset) -> str:
    """
    Detect the CF discrete sampling geometry representation of a dataset.

    Returns one of "contiguous", "indexed", "orthogonal", "incomplete" or
    "point". Ragged arrays are identified by their marker attributes
    (``sample_dimension`` / ``instance_dimension``); a multidimensional array is
    recognised by a data variable with two or more dimensions and classed as
    orthogonal when one of those dimensions is a shared coordinate axis, else
    incomplete; anything else is treated as point data.
    """
    for v in ds.variables:
        if "sample_dimension" in ds[v].attrs:
            return "contiguous"
    for v in ds.variables:
        if "instance_dimension" in ds[v].attrs:
            return "indexed"
    if ds.attrs.get("featureType") == "point":
        return "point"

    multidim = [v for v in ds.data_vars if ds[v].ndim >= 2]
    if multidim:
        # orthogonal has a shared 1-D element axis: a geometry coordinate
        # (e.g. time) that is a dimension coordinate. The instance dimension may
        # also carry a coordinate, but that holds ids, not a geometry axis.
        dims = set().union(*(ds[v].dims for v in multidim))
        geometry = set(_geometry_coordinates(ds))
        for d in dims:
            if (str(d) in geometry and d in ds.coords
                    and ds[str(d)].ndim == 1):
                return "orthogonal"
        return "incomplete"
    return "point"


def point_to_indexed(
    ds: xr.Dataset,
    sample_dim: str,
    instance_dim: str,
    timeseries_id: str,
    index_var: str = "locationIndex",
    instance_vars: Union[Sequence[str], None] = None,
    coord_vars: Union[Sequence[str], None] = None,
) -> xr.Dataset:
    """Convert a point dataset to an indexed ragged array dataset."""
    coord_vars = coord_vars or []
    instance_vars = instance_vars or []
    instance_vars = [timeseries_id] + list(instance_vars)

    _, unique_index_1d, instanceIndex = np.unique(
        ds[timeseries_id], return_index=True, return_inverse=True
    )
    # use assign (not ds[index_var] = ...) so the caller's dataset is not mutated
    ds = ds.assign(
        {index_var: (sample_dim, instanceIndex,
                     {"instance_dimension": instance_dim})}
    )

    for var in instance_vars:
        if var in ds:
            ds = ds.assign(
                {var: (instance_dim, ds[var][unique_index_1d].data, ds[var].attrs)}
            )
            if var in coord_vars:
                ds = ds.set_coords(var)
    ds = ds.assign_attrs({"featureType": "timeSeries"})
    return ds


def point_to_contiguous(
    ds: xr.Dataset,
    sample_dim: str,
    instance_dim: str,
    timeseries_id: str,
    count_var: str = "row_size",
    instance_vars: Union[Sequence[str], None] = None,
    coord_vars: Union[Sequence[str], None] = None,
    sort_vars: Union[Sequence[str], None] = None,
) -> xr.Dataset:
    """Convert a point dataset to a contiguous ragged array dataset."""
    coord_vars = coord_vars or []
    sort_vars = sort_vars or []
    instance_vars = instance_vars or []
    instance_vars = [timeseries_id] + list(instance_vars)

    ds = ds.sortby([timeseries_id, *sort_vars])
    _, unique_index_1d, row_size = np.unique(
        ds[timeseries_id], return_index=True, return_counts=True
    )

    ds = ds.assign(
        {count_var: (instance_dim, row_size, {"sample_dimension": sample_dim})}
    )

    for var in instance_vars:
        if var in ds:
            encoding = ds[var].encoding
            ds[var] = (instance_dim, ds[var][unique_index_1d].data, ds[var].attrs)
            ds[var].encoding = encoding
            if var in coord_vars:
                ds = ds.set_coords(var)
    ds = ds.assign_attrs({"featureType": "timeSeries"})
    return ds


def contiguous_to_indexed(
    ds: xr.Dataset,
    sample_dim: str,
    instance_dim: str,
    count_var: str,
    index_var: str,
) -> xr.Dataset:
    """Convert a contiguous ragged array dataset to an indexed ragged one."""
    row_size = np.where(ds[count_var].data > 0, ds[count_var].data, 0)

    locationIndex = np.repeat(np.arange(row_size.size), row_size)

    ds = ds.assign(
        {
            index_var: (
                sample_dim,
                locationIndex,
                {"instance_dimension": instance_dim},
            )
        }
    ).drop_vars([count_var])

    # put the index variable first
    ds = ds[[index_var] + [var for var in ds.variables if var != index_var]]

    return ds


def indexed_to_contiguous(
    ds: xr.Dataset,
    sample_dim: str,
    instance_dim: str,
    count_var: str,
    index_var: str,
    sort_vars: Union[Sequence[str], None] = None,
) -> xr.Dataset:
    """Convert an indexed ragged array dataset to a contiguous ragged one."""
    sort_vars = sort_vars or []

    ds = ds.sortby([index_var, *sort_vars])
    idxs, sizes = np.unique(ds[index_var], return_counts=True)

    row_size = np.zeros_like(ds[instance_dim].data)
    row_size[idxs] = sizes
    ds = ds.assign(
        {count_var: (instance_dim, row_size, {"sample_dimension": sample_dim})}
    ).drop_vars([index_var])

    return ds


def indexed_to_point(
    ds: xr.Dataset, sample_dim: str, instance_dim: str, index_var: str
) -> xr.Dataset:
    """Convert an indexed ragged array dataset to a point dataset."""
    instance_vars = [var for var in ds.variables if instance_dim in ds[var].dims]
    for instance_var in instance_vars:
        ds = ds.assign(
            {
                instance_var: (
                    sample_dim,
                    ds[instance_var][ds[index_var]].data,
                    ds[instance_var].attrs,
                )
            }
        )
    ds = ds.drop_vars([index_var]).assign_attrs({"featureType": "point"})
    return ds


def contiguous_to_point(
    ds: xr.Dataset,
    sample_dim: str,
    instance_dim: str,
    count_var: str,
) -> xr.Dataset:
    """Convert a contiguous ragged array dataset to a point dataset."""
    row_size = ds[count_var].values
    ds = ds.drop_vars(count_var)
    instance_vars = [var for var in ds.variables if instance_dim in ds[var].dims]
    for instance_var in instance_vars:
        ds = ds.assign(
            {
                instance_var: (
                    sample_dim,
                    np.repeat(ds[instance_var].values, row_size),
                    ds[instance_var].attrs,
                )
            }
        )
    ds = ds.assign_attrs({"featureType": "point"})
    return ds


def _scatter_along_sample(values, sample_axis, rows, cols, n_rows, n_cols,
                          fill):
    """
    Scatter a sample-indexed array into a dense grid.

    The ``sample_axis`` of ``values`` is replaced by two axes of length
    ``n_rows`` and ``n_cols``; ``rows``/``cols`` give the grid position of each
    sample. Cells that receive no sample keep ``fill``.
    """
    out_shape = (values.shape[:sample_axis]
                 + (n_rows, n_cols)
                 + values.shape[sample_axis + 1:])
    out = np.full(out_shape, fill, dtype=values.dtype)
    index = ((slice(None),) * sample_axis + (rows, cols)
             + (slice(None),) * (values.ndim - sample_axis - 1))
    out[index] = values
    return out


def _build_grid_dataset(ds, sample_dim, instance_dim, element_dim, rows, cols,
                        n_inst, n_elem, instance_ids, skip_vars):
    """
    Scatter every sample-indexed variable of ``ds`` into an
    ``(instance, element)`` grid and carry the instance-level coordinates.
    """
    reshaped = xr.Dataset()
    carry_coords = []
    for v in ds.variables:
        if v in skip_vars or v == sample_dim or v == instance_dim:
            continue
        dims = ds[v].dims
        if sample_dim in dims:
            # scatter a sample-indexed variable into the (instance, element) grid
            ax = dims.index(sample_dim)
            arr = _scatter_along_sample(
                ds[v].values, ax, rows, cols, n_inst, n_elem,
                fill_value(ds[v].dtype))
            new_dims = dims[:ax] + (instance_dim, element_dim) + dims[ax + 1:]
            reshaped[v] = (new_dims, arr, dict(ds[v].attrs))
            reshaped[v].encoding["_FillValue"] = fill_value(reshaped[v].dtype)
            if v in ds.coords:
                carry_coords.append(v)
        elif dims == (instance_dim,):
            # carry an instance-level variable (coord or data var) unchanged
            reshaped[v] = ((instance_dim,), ds[v].values, dict(ds[v].attrs))
            if v in ds.coords:
                carry_coords.append(v)

    reshaped = reshaped.assign_coords({instance_dim: instance_ids})
    if carry_coords:
        reshaped = reshaped.set_coords(carry_coords)
    reshaped.attrs = dict(ds.attrs)
    return reshaped


def _grid_valid(ds, ref_var, instance_dim, element_dim):
    """Boolean (instance, element) mask of non-fill cells from a reference var."""
    ref = ds[ref_var]
    extra = [d for d in ref.dims if d not in (instance_dim, element_dim)]
    ref2d = ref.isel({d: 0 for d in extra}).transpose(
        instance_dim, element_dim).values
    if np.issubdtype(ref2d.dtype, np.floating):
        return ~(np.isnan(ref2d) | (ref2d == fill_value(ref2d.dtype)))
    return ref2d != fill_value(ref2d.dtype)


def _flatten_grid(ds, grid_vars, instance_dim, element_dim, valid, row_size,
                  count_var, sample_dim):
    """Collect the ``valid`` grid cells of every grid variable, contiguously."""
    new_vars = {}
    for v in grid_vars:
        # place instance_dim/element_dim adjacent, keeping the other dims put
        new_order = []
        for d in ds[v].dims:
            if d == instance_dim:
                new_order += [instance_dim, element_dim]
            elif d != element_dim:
                new_order.append(d)
        da = ds[v].transpose(*new_order)
        row_axis = new_order.index(instance_dim)
        gathered = da.values[
            (slice(None),) * row_axis + (valid,)
            + (slice(None),) * (da.ndim - row_axis - 2)
        ]
        out_dims = tuple(
            sample_dim if d == instance_dim else d
            for d in new_order if d != element_dim
        )
        new_vars[v] = (out_dims, gathered, dict(ds[v].attrs))

    new_vars[count_var] = (
        (instance_dim,), row_size, {"sample_dimension": sample_dim}
    )
    # carry instance-level variables (coords and data vars) with their attrs
    coords = {}
    for v in ds.variables:
        if v in new_vars or v == instance_dim:
            continue
        if ds[v].dims == (instance_dim,):
            entry = ((instance_dim,), ds[v].values, dict(ds[v].attrs))
            if v in ds.coords:
                coords[v] = entry
            else:
                new_vars[v] = entry
    result = xr.Dataset(new_vars, coords=coords)
    result.attrs = dict(ds.attrs)
    return result


def contiguous_to_incomplete(
    ds: xr.Dataset,
    sample_dim: str,
    instance_dim: str,
    count_var: str,
    element_dim: Union[str, None] = None,
    instance_id_var: Union[str, None] = None,
) -> xr.Dataset:
    """
    Convert a contiguous ragged array to an incomplete multidimensional array
    (CF 9.3.2).

    Each instance's samples are placed in the leading columns of a dense
    (instance x element) array; the trailing columns are padded with the dtype
    fill value. Variables with extra (non-sample) dimensions keep them, and
    instance-level coordinates are preserved.
    """
    element_dim = element_dim or sample_dim
    row_size = ds[count_var].values
    n_inst = row_size.size

    if instance_id_var is not None:
        instance_ids = ds[instance_id_var].values
    elif instance_dim in ds:
        instance_ids = ds[instance_dim].values
    else:
        instance_ids = np.arange(n_inst)

    n_elem = int(row_size.max()) if row_size.size else 0
    rows = np.arange(n_inst).repeat(row_size)
    cols = vrange(np.zeros_like(row_size), row_size)

    reshaped = _build_grid_dataset(
        ds, sample_dim, instance_dim, element_dim, rows, cols,
        n_inst, n_elem, instance_ids, skip_vars={count_var})
    return finalize_cf(reshaped, "timeSeries", instance_id_var=instance_id_var)


def incomplete_to_contiguous(
    ds: xr.Dataset,
    instance_dim: str,
    element_dim: str,
    count_var: str = "row_size",
    sample_dim: str = "obs",
) -> xr.Dataset:
    """
    Convert an incomplete multidimensional array (CF 9.3.2) to a contiguous
    ragged array.

    Padded (fill-valued) elements are dropped, so each instance's valid samples
    are packed contiguously. A cell is considered padding if it equals the dtype
    fill value (or is NaN for floats).

    Notes
    -----
    Because the incomplete representation marks padding with fill values, this
    conversion cannot distinguish a genuine observation whose value equals the
    fill sentinel (or NaN) from padding — such observations are dropped. It also
    assumes the padding pattern is uniform across variables and takes the valid
    mask from the first ``(instance, element)`` variable. Use the contiguous or
    indexed representations to preserve fill-valued observations.
    """
    grid_vars = [
        v for v in ds.variables
        if instance_dim in ds[v].dims and element_dim in ds[v].dims
    ]
    if not grid_vars:
        raise ValueError("No (instance, element) variables to convert.")

    valid = _grid_valid(ds, grid_vars[0], instance_dim, element_dim)
    row_size = valid.sum(axis=1).astype(np.int32)
    result = _flatten_grid(ds, grid_vars, instance_dim, element_dim, valid,
                           row_size, count_var, sample_dim)
    return finalize_cf(result, "timeSeries")


def contiguous_to_orthogonal(
    ds: xr.Dataset,
    sample_dim: str,
    instance_dim: str,
    count_var: str,
    element_coord: str,
    element_dim: Union[str, None] = None,
    instance_id_var: Union[str, None] = None,
    strict: bool = True,
) -> xr.Dataset:
    """
    Convert a contiguous ragged array to an orthogonal multidimensional array
    (CF 9.3.1).

    All instances must share the same set of ``element_coord`` values (e.g. the
    same time axis); the samples are pivoted onto that shared 1-D coordinate.

    Parameters
    ----------
    element_coord : str
        Name of the per-sample coordinate variable defining the shared element
        axis (e.g. ``"time"``).
    strict : bool, optional
        If True (default), raise if the instances do not form a complete
        rectangular grid on ``element_coord`` (i.e. it is not truly orthogonal).

    Raises
    ------
    ValueError
        If ``strict`` and the instances do not share a common, complete element
        axis.
    """
    element_dim = element_dim or element_coord
    row_size = ds[count_var].values
    n_inst = row_size.size

    if instance_id_var is not None:
        instance_ids = ds[instance_id_var].values
    elif instance_dim in ds:
        instance_ids = ds[instance_dim].values
    else:
        instance_ids = np.arange(n_inst)

    rows = np.repeat(np.arange(n_inst), row_size)
    elem_vals = ds[element_coord].values
    unique_elems = np.unique(elem_vals)
    n_elem = unique_elems.size
    cols = np.searchsorted(unique_elems, elem_vals)

    if strict:
        flat = rows.astype(np.int64) * n_elem + cols
        if elem_vals.size != n_inst * n_elem or np.unique(flat).size != flat.size:
            raise ValueError(
                "Cannot build an orthogonal multidimensional array: the "
                f"instances do not share a complete '{element_coord}' axis. "
                "Use the incomplete representation instead."
            )

    reshaped = _build_grid_dataset(
        ds, sample_dim, instance_dim, element_dim, rows, cols,
        n_inst, n_elem, instance_ids, skip_vars={count_var, element_coord})
    # the shared element coordinate (1-D over the element dimension)
    reshaped = reshaped.assign_coords(
        {element_coord: ([element_dim], unique_elems)})
    return finalize_cf(reshaped, "timeSeries", instance_id_var=instance_id_var)


def orthogonal_to_contiguous(
    ds: xr.Dataset,
    instance_dim: str,
    element_dim: str,
    element_coord: Union[str, None] = None,
    count_var: str = "row_size",
    sample_dim: str = "obs",
) -> xr.Dataset:
    """
    Convert an orthogonal multidimensional array (CF 9.3.1) to a contiguous
    ragged array.

    Every (instance, element) cell is a valid observation, so each instance has
    ``element`` samples and the shared element coordinate is broadcast to every
    observation.
    """
    n_inst = ds.sizes[instance_dim]
    n_elem = ds.sizes[element_dim]

    grid_vars = [
        v for v in ds.variables
        if instance_dim in ds[v].dims and element_dim in ds[v].dims
    ]
    if not grid_vars:
        raise ValueError("No (instance, element) variables to convert.")

    valid = np.ones((n_inst, n_elem), dtype=bool)  # complete grid
    row_size = np.full(n_inst, n_elem, dtype=np.int32)
    result = _flatten_grid(ds, grid_vars, instance_dim, element_dim, valid,
                           row_size, count_var, sample_dim)

    if element_coord is not None and element_coord in ds:
        result = result.assign_coords(
            {element_coord: ((sample_dim,),
                             np.tile(ds[element_coord].values, n_inst))})
    return finalize_cf(result, "timeSeries")
