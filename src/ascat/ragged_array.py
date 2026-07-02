# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

from pathlib import Path

import numpy as np
import xarray as xr

from ascat.cf_conversions import point_to_indexed
from ascat.cf_conversions import point_to_contiguous
from ascat.cf_conversions import indexed_to_contiguous
from ascat.cf_conversions import contiguous_to_indexed
from ascat.cf_conversions import contiguous_to_point
from ascat.cf_conversions import indexed_to_point
from ascat.cf_conversions import contiguous_to_incomplete
from ascat.cf_conversions import incomplete_to_contiguous
from ascat.cf_conversions import contiguous_to_orthogonal
from ascat.cf_conversions import orthogonal_to_contiguous
from ascat.cf_conversions import detect_cf_representation
from ascat.cf_conversions import finalize_cf


def _save_cf(ds: xr.Dataset, filename: str, feature_type: str,
             instance_id_var: str = None):
    """Finalize CF metadata and write a dataset to NetCDF or Zarr."""
    ds = finalize_cf(ds, feature_type, instance_id_var=instance_id_var)
    suffix = Path(filename).suffix
    if suffix == ".nc":
        ds.to_netcdf(filename)
    elif suffix == ".zarr":
        ds.to_zarr(filename)
    else:
        raise ValueError(f"Unknown file suffix '{suffix}' "
                         "(.nc and .zarr supported)")


class _InstanceLookup:
    """
    Map instance ids to their positions.

    Memory scales with the number of instances, not the largest id value. When
    the ids are exactly ``0, 1, ..., N-1`` (a positional index) the lookup is a
    direct O(1) index; otherwise it is an O(log N) binary search over the sorted
    ids. ``positions`` returns ``-1`` for ids that are not present.
    """

    def __init__(self, instance_ids):
        instance_ids = np.asarray(instance_ids)
        self.size = instance_ids.size
        # fast path: ids are already the positional index 0..N-1
        self._identity = (
            self.size > 0
            and np.issubdtype(instance_ids.dtype, np.integer)
            and instance_ids[0] == 0
            and instance_ids[-1] == self.size - 1
            and np.array_equal(instance_ids, np.arange(self.size))
        )
        if not self._identity:
            order = np.argsort(instance_ids, kind="stable")
            self._sorted_ids = instance_ids[order]
            self._positions = order

    def positions(self, ids):
        """Return the position of each id, or -1 if absent."""
        ids = np.atleast_1d(np.asarray(ids))
        if self._identity:
            pos = ids.astype(np.int64, copy=True)
            pos[(ids < 0) | (ids >= self.size)] = -1
            return pos
        if self.size == 0:
            return np.full(ids.shape, -1, dtype=np.int64)
        rank = np.clip(
            np.searchsorted(self._sorted_ids, ids), 0, self.size - 1)
        found = self._sorted_ids[rank] == ids
        return np.where(found, self._positions[rank], -1)


def verify_multidim(ds: xr.Dataset, instance_dim: str,
                    element_dim: str) -> None:
    """
    Verify a dataset follows the CF multidimensional array definition
    (orthogonal or incomplete).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to be verified.
    instance_dim : str
        Name of the instance dimension.
    element_dim : str
        Name of the element dimension.

    Raises
    ------
    RuntimeError if verification fails.
    """
    # check that instance dimension exists
    if instance_dim not in ds.dims:
        raise RuntimeError(f"Instance dimension is missing '{instance_dim}'")

    # check that element dimension exists
    if element_dim not in ds.dims:
        raise RuntimeError(f"Element dimension is missing '{element_dim}'")


def verify_contiguous_ragged(ds: xr.Dataset, count_var: str,
                             instance_dim: str) -> None:
    """
    Verify dataset follows contiguous ragged array CF definition.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to be verified.
    count_var : str
        Name of the count variable.
        Count variable contains the length of each time series feature. It is
        identified by having an attribute with name 'sample_dimension' whose
        value is name of the sample dimension. The count variable implicitly
        partitions into individual instances all variables that have the
        sample dimension.

    Raises
    ------
    RuntimeError if verification fails.
    """
    # check that count variable exists
    if count_var not in ds:
        raise RuntimeError(f"Count variable is missing: {count_var}")

    # check that count variable contains sample_dimension attribute
    if "sample_dimension" not in ds[count_var].attrs:
        raise RuntimeError(f"Count variable '{count_var}' has no "
                           "sample_dimension attribute")

    # check that count variable has instance_dimension as single dimension
    if ds[count_var].dims != (instance_dim,):
        raise RuntimeError(f"Count variable '{count_var}' must have the "
                           f"instance dimension '{instance_dim}' as its "
                           "single dimension")


def verify_indexed_ragged(ds: xr.Dataset, index_var: str,
                          sample_dim: str) -> None:
    """
    Verify dataset follows indexed ragged array CF definition.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset.
    index_var : str
        The index variable can be identified by having an attribute with
        name of instance_dimension whose value is the instance dimension.
    sample_dim : str
        Name of the sample dimension.

    Raises
    ------
    RuntimeError if verification fails.
    """
    # check that index variable exists
    if index_var not in ds:
        raise RuntimeError(f"Index variable is missing: {index_var}")

    # check that index variable must have sample dimension as single dimension
    if ds[index_var].dims != (sample_dim,):
        raise RuntimeError(f"Index variable '{index_var}' must have the "
                           f"sample dimension '{sample_dim}' as its "
                           "single dimension")

    # check that index variable has instance_dimension attribute
    if "instance_dimension" not in ds[index_var].attrs:
        raise RuntimeError(f"Index variable '{index_var}' has no "
                           "instance_dimension attribute")


def verify_point_array(ds: xr.Dataset, sample_dim: str) -> None:
    """
    Verify dataset follows the CF point data array convention.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to be verified.
    sample_dim : str
        Name of the sample dimension.

    Raises
    ------
    RuntimeError if verification fails.
    """
    # check that the sample_dim exists
    if sample_dim not in ds.dims:
        raise RuntimeError(f"Sample dimension '{sample_dim}' is missing.")

    # check all data and coordinate variables have only the sample_dim
    for var in ds.variables:
        dims = ds[var].dims
        if ds[var].ndim > 0 and dims != (sample_dim,):
            raise RuntimeError(
                f"Variable '{var}' does not conform to point structure "
                f"(dims: {dims}). All variables must use only the sample_dim ('{sample_dim}')."
            )


class PointData:
    """
    Point data represent scattered locations and times with no implied
    relationship among of coordinate positions, both data and coordinates must
    share the same (sample) instance dimension.
    """

    def __init__(self, ds: xr.Dataset, sample_dim: str):
        """
        Initialize.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset to be verified.
        sample_dim : str
            Name of the sample dimension. The sample dimension indicates the
            number of instances (e.g. stations, locations).
        """
        self.sample_dim = sample_dim
        self._data = ds
        self.validate()

    def validate(self):
        """Validate format."""
        verify_point_array(self.ds, self.sample_dim)

    @classmethod
    def from_file(cls, filename: str, sample_dim: str = None, **kwargs):
        """
        Load point data from a file.

        Parameters
        ----------
        filename : str
            Filename.
        sample_dim : str, optional
            Sample dimension name (inferred from the file if not given).

        Returns
        -------
        data : PointData
            Point data loaded from a file.
        """
        ds = xr.open_dataset(filename, **kwargs)
        return cls(ds, sample_dim or _infer_sample_dim(ds))

    @property
    def ds(self):
        return self._data

    def save(self, filename: str):
        """Write point data to file (".nc" or ".zarr") with CF metadata."""
        _save_cf(self.ds, filename, "point")

    def append(self, other):
        """
        Append point data along the sample dimension (in place).

        Parameters
        ----------
        other : PointData or xarray.Dataset
            Point data to append. It must follow the same point structure (all
            variables over the sample dimension).
        """
        ds = other.ds if isinstance(other, PointData) else other
        verify_point_array(ds, self.sample_dim)
        self._data = xr.concat([self._data, ds], dim=self.sample_dim)

    def to_indexed(self, index_var: str = "locationIndex",
                   instance_dim: str = "locations"):
        """
        Convert point data to indexed ragged array.

        Parameters
        ----------
        index_var : str
            Name of the new index variable to be added.
        instance_dim : str
            Name of the instance dimension.

        Returns
        -------
        indexed : IndexedRaggedArray
            Indexed ragged array object.
        """
        if instance_dim not in self.ds:
            raise ValueError(
                f"'{instance_dim}' must be a coordinate in the dataset")

        new_ds = point_to_indexed(
            self.ds, self.sample_dim, instance_dim, instance_dim,
            index_var=index_var, coord_vars=[instance_dim])
        return IndexedRaggedArray(
            new_ds, index_var=index_var, sample_dim=self.sample_dim)

    def to_contiguous(self,
                      count_var: str = "row_size",
                      instance_dim: str = "locations"):
        """
        Convert point data to contiguous ragged array.

        Parameters
        ----------
        count_var : str
            Name of the new count variable to be added (default: 'row_size').
        instance_dim : str
            Name of the instance dimension (default: 'locations').

        Returns
        -------
        contiguous : ContiguousRaggedArray
            Contiguous ragged array object.
        """
        if instance_dim not in self.ds:
            raise ValueError(
                f"'{instance_dim}' must be a coordinate in the dataset")

        new_ds = point_to_contiguous(
            self.ds, self.sample_dim, instance_dim, instance_dim,
            count_var=count_var, coord_vars=[instance_dim])
        return ContiguousRaggedArray(
            new_ds, count_var=count_var, instance_dim=instance_dim)

    def to_incomplete(self,
                      count_var: str = "row_size",
                      instance_dim: str = "locations"):
        """
        Convert point data to an incomplete multidimensional array (CF 9.3.2).

        Parameters
        ----------
        count_var : str, optional
            Name of the intermediate count variable (default: "row_size").
        instance_dim : str, optional
            Name of the instance dimension (default: "locations").

        Returns
        -------
        data : IncompleteMultidimArray
            Incomplete multidimensional array time series.
        """
        return self.to_contiguous(
            count_var=count_var, instance_dim=instance_dim).to_incomplete()

    def to_orthogonal(self,
                      element_coord: str,
                      count_var: str = "row_size",
                      instance_dim: str = "locations",
                      element_dim: str = None,
                      strict: bool = True):
        """
        Convert point data to an orthogonal multidimensional array (CF 9.3.1).

        Parameters
        ----------
        element_coord : str
            Name of the per-sample coordinate defining the shared element axis.
        count_var : str, optional
            Name of the intermediate count variable (default: "row_size").
        instance_dim : str, optional
            Name of the instance dimension (default: "locations").
        element_dim : str, optional
            Name of the resulting element dimension (default: ``element_coord``).
        strict : bool, optional
            If True (default), raise if the instances do not form a complete
            grid on ``element_coord``.

        Returns
        -------
        data : OrthogonalMultidimArray
            Orthogonal multidimensional array time series.
        """
        return self.to_contiguous(
            count_var=count_var, instance_dim=instance_dim).to_orthogonal(
                element_coord, element_dim=element_dim, strict=strict)


class MultidimArray:
    """
    Base class for CF multidimensional array representations.

    Holds a dense ``(instance, element)`` dataset and the behaviour shared by
    the orthogonal (CF 9.3.1) and incomplete (CF 9.3.2) representations. The
    conversions back to ragged/point form are defined by the subclasses, which
    differ in whether the element dimension is a shared coordinate axis
    (orthogonal) or a padded positional index (incomplete).

    Attributes
    ----------
    instance_dim : str
        Name of the instance dimension.
    element_dim : str
        Name of the element dimension.
    ds : xarray.Dataset
        Multidimensional array dataset.
    """

    def __init__(self,
                 ds: xr.Dataset,
                 instance_dim: str = "locations",
                 element_dim: str = "time",
                 instance_id_var: str = None):
        """
        Initialize.

        Parameters
        ----------
        ds : xr.Dataset
            Data stored in multidimensional array format.
        instance_dim : str
            Instance dimension name.
        element_dim : str
            Element dimension name.
        instance_id_var : str, optional
            Variable holding the instance identifiers (e.g. "location_id") when
            they are not stored on the instance dimension. If None, the instance
            dimension coordinate is used, or a positional 0..N-1 index.
        """
        self.instance_dim = instance_dim
        self.element_dim = element_dim
        self.instance_id_var = instance_id_var
        self._data = ds
        self.validate()
        self._set_instance_lut()

    def validate(self):
        """Validate format."""
        verify_multidim(self.ds, self.instance_dim, self.element_dim)

    def _set_instance_lut(self):
        """Set instance lookup-table mapping instance ids to positions."""
        if self.instance_id_var is not None:
            instance_ids = self.ds[self.instance_id_var].to_numpy()
        elif self.instance_dim in self.ds:
            instance_ids = self.ds[self.instance_dim].to_numpy()
        else:
            instance_ids = np.arange(self.ds.sizes[self.instance_dim])
        self._lookup = _InstanceLookup(instance_ids)

    @property
    def ds(self):
        return self._data

    def save(self, filename: str):
        """Write to file (".nc" or ".zarr") with CF metadata finalized."""
        _save_cf(self.ds, filename, "timeSeries",
                 instance_id_var=self.instance_id_var)

    @property
    def size(self) -> int:
        """Number of instances."""
        return self.ds.sizes[self.instance_dim]

    @property
    def instance_ids(self):
        """Instance ids."""
        if self.instance_id_var is not None:
            return self.ds[self.instance_id_var].values
        if self.instance_dim in self.ds:
            return self.ds[self.instance_dim].values
        return np.arange(self.ds.sizes[self.instance_dim])

    @property
    def instance_variables(self) -> list:
        """
        Instance variables (dimensioned by the instance dimension only, i.e.
        the per-instance metadata such as lon/lat/location_id).

        Returns
        -------
        instance_variables : list of str
            Instance variable names.
        """
        return [
            var for var in self.ds.variables
            if (self.ds[var].dims == (self.instance_dim,)) and
            (var != self.instance_dim) and
            ("sample_dimension" not in self.ds[var].attrs)
        ]

    def _wrap(self, ds: xr.Dataset):
        """Wrap a subset dataset in a new array of the same type."""
        return type(self)(ds, self.instance_dim, self.element_dim,
                          instance_id_var=self.instance_id_var)

    def sel_instance(self, i: int):
        """Read time series for a given instance id."""
        pos = int(self._lookup.positions(i)[0])
        if pos == -1:
            return None
        return self.ds.isel({self.instance_dim: pos})

    def sel_instances(self, i: np.ndarray):
        """
        Select several instances, preserving request order.

        Returns
        -------
        data : MultidimArray or None
            A multidimensional array with the selected instances, or None if
            none of the ids are present.
        """
        i = np.atleast_1d(np.asarray(i))
        pos = self._lookup.positions(i)
        pos = pos[pos != -1]
        if pos.size == 0:
            return None
        return self._wrap(self.ds.isel({self.instance_dim: pos}))

    def __iter__(self):
        """Iterator over time series"""
        for i in self.instance_ids:
            yield self.sel_instance(i)

    def iter(self):
        """Explicit iterator method"""
        return self.__iter__()

    def apply(self, func):
        """Apply a function to each instance time series."""
        return [func(ts) for ts in self]

    def append(self, other):
        """
        Append another multidimensional array (in place).

        The two collections are merged by instance via a contiguous ragged
        array (observations of shared instances are combined, new instances
        added), then converted back to this representation.

        Parameters
        ----------
        other : MultidimArray or xarray.Dataset
            Multidimensional array to append (same structure).
        """
        other_ds = other.ds if isinstance(other, MultidimArray) else other
        cra = self.to_contiguous()
        cra.append(self._wrap(other_ds).to_contiguous())
        self._data = self._from_contiguous(cra).ds
        self._set_instance_lut()

    def _from_contiguous(self, cra):
        """Subclass hook: build this representation from a contiguous array."""
        raise NotImplementedError

    def _to_contiguous_ds(self, count_var, sample_dim):
        """Subclass hook: convert to a contiguous ragged array dataset."""
        raise NotImplementedError

    def to_contiguous(self, count_var: str = "row_size",
                      sample_dim: str = "obs"):
        """
        Convert to a contiguous ragged array.

        Returns
        -------
        data : ContiguousRaggedArray
            Contiguous ragged array time series.
        """
        new_ds = self._to_contiguous_ds(count_var, sample_dim)
        instance_id_var = (
            self.instance_id_var
            if self.instance_id_var in new_ds else None)
        return ContiguousRaggedArray(new_ds, count_var, self.instance_dim,
                                     instance_id_var=instance_id_var)

    def to_indexed(self, count_var: str = "row_size",
                   sample_dim: str = "obs", index_var: str = "locationIndex"):
        """
        Convert to an indexed ragged array (via a contiguous ragged array).

        Returns
        -------
        data : IndexedRaggedArray
            Indexed ragged array time series.
        """
        return self.to_contiguous(
            count_var=count_var, sample_dim=sample_dim).to_indexed(index_var)

    def to_point_data(self, count_var: str = "row_size",
                      sample_dim: str = "obs"):
        """
        Convert to point data (via a contiguous ragged array).

        Returns
        -------
        data : PointData
            Point data.
        """
        return self.to_contiguous(
            count_var=count_var, sample_dim=sample_dim).to_point_data()


class IncompleteMultidimArray(MultidimArray):
    """
    Incomplete multidimensional array representation (CF 9.3.2).

    Instances may have different numbers of elements (and different element
    coordinates); the dense ``(instance, element)`` array is padded with the
    dtype fill value. The element dimension is a positional index.
    """

    @classmethod
    def from_file(cls,
                  filename: str,
                  instance_dim: str = "locations",
                  element_dim: str = "time",
                  instance_id_var: str = None,
                  **kwargs):
        """
        Load an incomplete multidimensional array from a file.

        Parameters
        ----------
        filename : str
            Filename.
        instance_dim : str, optional
            Instance dimension name (default: "locations").
        element_dim : str, optional
            Element dimension name (default: "time").
        instance_id_var : str, optional
            Variable used as instance identifier (default: None).

        Returns
        -------
        data : IncompleteMultidimArray
            Incomplete multidimensional array loaded from a file.
        """
        ds = xr.open_dataset(filename, **kwargs)
        return cls(ds, instance_dim, element_dim,
                   instance_id_var=instance_id_var)

    def _from_contiguous(self, cra):
        return cra.to_incomplete()

    def _to_contiguous_ds(self, count_var, sample_dim):
        return incomplete_to_contiguous(
            self.ds,
            self.instance_dim,
            self.element_dim,
            count_var=count_var,
            sample_dim=sample_dim,
        )


class OrthogonalMultidimArray(MultidimArray):
    """
    Orthogonal multidimensional array representation (CF 9.3.1).

    All instances share the same element coordinate axis (e.g. one time axis),
    stored as a single shared 1-D coordinate; the array is complete (no
    padding).

    Parameters
    ----------
    element_coord : str, optional
        Name of the shared element coordinate variable (default: the element
        dimension name).
    """

    def __init__(self,
                 ds: xr.Dataset,
                 instance_dim: str = "locations",
                 element_dim: str = "time",
                 element_coord: str = None,
                 instance_id_var: str = None):
        self.element_coord = element_coord or element_dim
        super().__init__(ds, instance_dim, element_dim,
                         instance_id_var=instance_id_var)

    @classmethod
    def from_file(cls,
                  filename: str,
                  instance_dim: str = "locations",
                  element_dim: str = "time",
                  element_coord: str = None,
                  instance_id_var: str = None,
                  **kwargs):
        """
        Load an orthogonal multidimensional array from a file.

        Parameters
        ----------
        filename : str
            Filename.
        instance_dim : str, optional
            Instance dimension name (default: "locations").
        element_dim : str, optional
            Element dimension name (default: "time").
        element_coord : str, optional
            Shared element coordinate variable name (default: element_dim).
        instance_id_var : str, optional
            Variable used as instance identifier (default: None).

        Returns
        -------
        data : OrthogonalMultidimArray
            Orthogonal multidimensional array loaded from a file.
        """
        ds = xr.open_dataset(filename, **kwargs)
        return cls(ds, instance_dim, element_dim, element_coord=element_coord,
                   instance_id_var=instance_id_var)

    def _wrap(self, ds: xr.Dataset) -> "OrthogonalMultidimArray":
        """Wrap a subset dataset, preserving the shared element coordinate."""
        return OrthogonalMultidimArray(
            ds, self.instance_dim, self.element_dim,
            element_coord=self.element_coord,
            instance_id_var=self.instance_id_var)

    def _from_contiguous(self, cra):
        return cra.to_orthogonal(self.element_coord,
                                 element_dim=self.element_dim, strict=False)

    def _to_contiguous_ds(self, count_var, sample_dim):
        return orthogonal_to_contiguous(
            self.ds,
            self.instance_dim,
            self.element_dim,
            element_coord=self.element_coord,
            count_var=count_var,
            sample_dim=sample_dim,
        )


class ContiguousRaggedArray:
    """
    Contiguous ragged array representation (CF convention).

    In an contiguous ragged array representation, the dataset for all time
    series are stored in a single 1D array. Additional variables or
    dimensions provide the metadata needed to map these values back to their
    respective time series.

    The contiguous ragged array representation can be used only if the size
    of each instance is known at the time that it is created. In this
    representation the data for each instance will be contiguous on disk.

    If the instance dimension exists as a variable, it is assumed that the
    values represent the identifiers for each instance otherwise they are
    count upwards from 0.

    Attributes
    ----------
    instance_dim : str
        Name of the instance dimension.
    sample_dim : str
        Name of the sample dimension. The variable bearing the
        sample_dimension attribute (i.e. count_var) must have the instance
        dimension as its single dimension, and must have an integer type.
    count_var : str
        Name of the count variable. The count variable must be an integer
        type and must have the instance dimension as its sole dimension.
        The count variable are identifiable by the presence of an attribute,
        sample_dimension, found on the count variable, which names the sample
        dimension being counted.
    ds : xarray.Dataset
        Contiguous ragged array dataset.
    instance_variables : list
        Per-instance metadata variables (over the instance dimension).
    observation_variables : list
        Measurement variables (over the sample dimension).
    instance_ids : list
        List of instance ids.

    Methods
    -------
    sel_instance(i)
        Read time series for given instance.
    iter()
        Yield time series for each instance.
    """

    def __init__(self,
                 ds: xr.Dataset,
                 count_var: str,
                 instance_dim: str,
                 instance_id_var: str = None):
        """
        Initialize.

        Parameters
        ----------
        ds : xr.Dataset
            Data stored in contiguous ragged array format.
        count_var : str
            Count variable name.
        instance_dim : str
            Instance dimension name.
        instance_id_var: str, optional
            Variable used as instance identifier (default: None).
        """
        self.count_var = count_var
        self.instance_dim = instance_dim
        self._data = ds
        self.validate()

        self.sample_dim = ds[count_var].attrs["sample_dimension"]
        self.instance_id_var = instance_id_var

        # cache row_size and instance_ids data
        self._row_size = self.ds[self.count_var].to_numpy()

        if self.instance_id_var is None:
            self._instance_ids = self.ds[self.instance_dim].to_numpy()
        else:
            self._instance_ids = self.ds[self.instance_id_var].to_numpy()

        self._set_instance_lut()

    def validate(self):
        """Validate format."""
        verify_contiguous_ragged(self.ds, self.count_var, self.instance_dim)

    def _set_instance_lut(self):
        """
        Set instance lookup-table.

        Memory scales with the number of instances, not the largest id value.
        """
        self._lookup = _InstanceLookup(self._instance_ids)
        # exclusive prefix sum: first sample offset of each instance
        self._row_start = np.cumsum(self._row_size) - self._row_size

    @classmethod
    def from_file(cls,
                  filename: str,
                  count_var: str,
                  instance_dim: str,
                  instance_id_var: str = None,
                  trim: bool = False,
                  **kwargs):
        """
        Load time series from file.

        Parameters
        ----------
        filename : str
            Filename.
        count_var : str
            Count variable name.
        instance_dim : str
            Instance dimension name.
        instance_id_var: str, optional
            Variable used as instance identifier (default: None).
        trim : bool, optional
            If True, drop fill/padding locations after loading (default: False).

        Returns
        -------
        data : ContiguousRaggedArray
            ContiguousRaggedArray object loaded from a file.
        """
        ds = xr.open_dataset(filename, **kwargs)
        verify_contiguous_ragged(ds, count_var, instance_dim)

        data = cls(ds, count_var, instance_dim, instance_id_var)
        return data.trim() if trim else data

    @property
    def ds(self):
        """
        Dataset.

        Returns
        -------
        ds : xr.Dataset
            Contiguous ragged array dataset.
        """
        return self._data

    def save(self, filename: str):
        """Write to file (".nc" or ".zarr") with CF metadata finalized."""
        _save_cf(self.ds, filename, "timeSeries",
                 instance_id_var=self.instance_id_var)

    @property
    def size(self) -> int:
        """
        Number of instances.

        Returns
        -------
        instance_ids : int
            Number of instance.
        """
        return self._instance_ids.size

    @property
    def instance_ids(self) -> list:
        """
        Instance ids

        Returns
        -------
        instance_ids : list of int
            Instance ids.
        """
        return self._instance_ids

    @property
    def instance_variables(self) -> list:
        """
        Instance variables (dimensioned by the instance dimension only, i.e.
        the per-instance metadata such as lon/lat/location_id).

        Returns
        -------
        instance_variables : list of str
            Instance variable names.
        """
        return [
            var for var in self.ds.variables
            if (self.ds[var].dims == (self.instance_dim,)) and
            (var != self.instance_dim) and
            ("sample_dimension" not in self.ds[var].attrs)
        ]

    @property
    def observation_variables(self) -> list:
        """
        Observation variables (dimensioned by the sample dimension, i.e. the
        measurements that vary per observation).

        Returns
        -------
        observation_variables : list of str
            Observation variable names.
        """
        return [
            var for var in self.ds.variables
            if (self.ds[var].dims == (self.sample_dim,)) and
            (var != self.sample_dim)
        ]

    def get_observation_variables(self, include_dtype: bool = False) -> list:
        """
        Observation variables, optionally with their dtypes.

        Parameters
        ----------
        include_dtype : bool, optional
            If True, return ``(name, dtype)`` tuples (default: False).

        Returns
        -------
        observation_variables : list of str or list of (str, dtype)
            Observation variable names (dimensioned by the sample dimension).
        """
        if include_dtype:
            return [
                (var, self.ds[var].dtype)
                for var in self.observation_variables
            ]
        return self.observation_variables

    def sel_instance(self, i: int):
        """Read time series"""
        idx = int(self._lookup.positions(i)[0])
        if idx == -1:
            return None
        start = int(self._row_start[idx])
        end = start + int(self._row_size[idx])
        return self.ds.isel({
            self.sample_dim: slice(start, end),
            self.instance_dim: idx,
        })

    def _wrap(self, ds: xr.Dataset) -> "ContiguousRaggedArray":
        """Wrap a subset dataset in a new ContiguousRaggedArray."""
        return ContiguousRaggedArray(ds, self.count_var, self.instance_dim,
                                     instance_id_var=self.instance_id_var)

    def sel_instances(self, i: np.ndarray) -> "ContiguousRaggedArray":
        """
        Select several instances, preserving request order.

        Parameters
        ----------
        i : np.ndarray
            Array of instance IDs.

        Returns
        -------
        data : ContiguousRaggedArray or None
            A contiguous ragged array with the selected instances (in the order
            of ``i``), or None if none of the ids are present.
        """
        i = np.atleast_1d(np.asarray(i))
        pos = self._lookup.positions(i)
        pos = pos[pos != -1]
        if pos.size == 0:
            return None

        starts = self._row_start[pos]
        ends = starts + self._row_size[pos]
        sample_idx = np.concatenate(
            [np.arange(s, e) for s, e in zip(starts, ends)])
        return self._wrap(self.ds.isel({
            self.sample_dim: sample_idx,
            self.instance_dim: pos,
        }))

    def trim(self):
        """
        Drop fill/padding locations.

        Cell files often over-allocate the instance dimension and pad the unused
        locations with fill values (a negative fill for the integer count
        variable). This removes those locations and keeps only the observations
        belonging to real ones. Locations with a valid count of zero are kept.

        Returns
        -------
        data : ContiguousRaggedArray
            Trimmed contiguous ragged array (self if nothing to trim).
        """
        valid = self._row_size >= 0
        if bool(valid.all()):
            return self

        clamped = np.where(valid, self._row_size, 0)
        starts = np.cumsum(clamped) - clamped
        sample_idx = np.concatenate(
            [np.arange(s, s + c)
             for s, c, v in zip(starts, clamped, valid) if v]
        ) if valid.any() else np.array([], dtype=np.int64)

        ds = self.ds.isel({
            self.sample_dim: sample_idx,
            self.instance_dim: valid,
        })
        return ContiguousRaggedArray(
            ds, self.count_var, self.instance_dim,
            instance_id_var=self.instance_id_var)

    def __iter__(self):
        """
        Iterator over instances.

        Returns
        -------
        ds : xr.Dataset
            Time series for instance.
        """
        for i in self.instance_ids:
            yield self.sel_instance(i)

    def iter(self):
        """
        Explicit iterator method.

        Returns
        -------
        ds : xr.Dataset
            Time series for instance.
        """
        return self.__iter__()

    def to_indexed(self, index_var: str = "locationIndex"):
        """
        Convert to indexed ragged array.

        Parameters
        ----------
        index_var : str, optional
            Name of the index variable to create (default: "locationIndex").

        Returns
        -------
        data : IndexedRaggedArray
            Indexed ragged array time series.
        """
        ds = contiguous_to_indexed(self.ds, self.sample_dim, self.instance_dim,
                                   self.count_var, index_var)

        return IndexedRaggedArray(ds, index_var, self.sample_dim,
                                  instance_id_var=self.instance_id_var)

    def to_incomplete(self):
        """
        Convert to an incomplete multidimensional array (CF 9.3.2).

        Each instance's samples are packed into the leading columns of a dense
        (instance x element) array and the rest padded with fill values.

        Returns
        -------
        data : IncompleteMultidimArray
            Incomplete multidimensional array time series.
        """
        instance_id_var = (
            self.instance_id_var if self.instance_id_var is not None
            else self.instance_dim
        )
        reshaped_ds = contiguous_to_incomplete(
            self.ds,
            self.sample_dim,
            self.instance_dim,
            self.count_var,
            element_dim=self.sample_dim,
            instance_id_var=instance_id_var,
        )
        return IncompleteMultidimArray(
            reshaped_ds, self.instance_dim, self.sample_dim,
            instance_id_var=self.instance_id_var)

    def to_orthogonal(self, element_coord: str, element_dim: str = None,
                      strict: bool = True):
        """
        Convert to an orthogonal multidimensional array (CF 9.3.1).

        All instances must share the same set of ``element_coord`` values (e.g.
        the same time axis); the samples are pivoted onto that shared
        coordinate.

        Parameters
        ----------
        element_coord : str
            Name of the per-sample coordinate defining the shared element axis.
        element_dim : str, optional
            Name of the resulting element dimension (default: ``element_coord``).
        strict : bool, optional
            If True (default), raise if the instances do not form a complete
            grid on ``element_coord``.

        Returns
        -------
        data : OrthogonalMultidimArray
            Orthogonal multidimensional array time series.
        """
        element_dim = element_dim or element_coord
        instance_id_var = (
            self.instance_id_var if self.instance_id_var is not None
            else self.instance_dim
        )
        reshaped_ds = contiguous_to_orthogonal(
            self.ds,
            self.sample_dim,
            self.instance_dim,
            self.count_var,
            element_coord,
            element_dim=element_dim,
            instance_id_var=instance_id_var,
            strict=strict,
        )
        return OrthogonalMultidimArray(
            reshaped_ds, self.instance_dim, element_dim,
            element_coord=element_coord,
            instance_id_var=self.instance_id_var)

    def to_point_data(self):
        """
        Convert to point data.

        Instance-level variables are broadcast to the sample dimension so that
        every observation carries its instance's coordinates.

        Returns
        -------
        data : PointData
            Point data.
        """
        ds = contiguous_to_point(self.ds, self.sample_dim, self.instance_dim,
                                 self.count_var)
        return PointData(ds, self.sample_dim)

    def apply(self, func):
        """Apply a function to each instance's time series."""
        return [func(ts) for ts in self]

    def append(self, other):
        """
        Append another contiguous ragged array (in place).

        The two collections are merged by instance: observations of shared
        instances are combined, and instances present in only one are added.

        Parameters
        ----------
        other : ContiguousRaggedArray or xarray.Dataset
            Contiguous ragged array to append (same structure).
        """
        other_ds = other.ds if isinstance(other, ContiguousRaggedArray) \
            else other
        other_cra = ContiguousRaggedArray(
            other_ds, self.count_var, self.instance_dim,
            instance_id_var=self.instance_id_var)

        id_var = self.instance_id_var or self.instance_dim
        # per-instance variables to collapse back after regrouping (not the id)
        instance_vars = [v for v in self.instance_variables if v != id_var]
        coord_vars = [v for v in instance_vars if v in self.ds.coords]

        # broadcast both to point form (every obs carries its instance vars),
        # concatenate along the sample dimension, then regroup by instance id
        combined = xr.concat(
            [self.to_point_data().ds, other_cra.to_point_data().ds],
            dim=self.sample_dim)
        new_ds = point_to_contiguous(
            combined, self.sample_dim, self.instance_dim, id_var,
            count_var=self.count_var, instance_vars=instance_vars,
            coord_vars=coord_vars)
        self.__init__(new_ds, self.count_var, self.instance_dim,
                      instance_id_var=self.instance_id_var)


class IndexedRaggedArray:
    """
    Indexed ragged array representation (CF convention).

    In an indexed ragged array representation, the dataset is structured
    to store variable-length data (e.g., time series with varying lengths)
    compactly. To achieve this, auxiliary indexing variables that map the
    flat array storage to meaningful groups (e.g. locations).

    If the instance dimension exists as a variable, it is assumed that the
    values represent the identfiers for each instance otherwise they counting
    upwards from 0.

    Attributes
    ----------
    index_var : str
        The indexed ragged array representation must contain an index
        variable, which must be an integer type, and must have the sample
        dimension as its single dimension.
        The index variable can be identified by having an attribute
        'instance_dimension' whose value is the instance dimension.
    sample_dim : str
        Name of the sample dimension. The sample dimension indicates the
        number of instances (e.g. stations, locations).
    instance_dim : str
        The name of the instance dimension. The value is defined by
        the 'instance_dimension' attribute, which must be present on the
        index variable. All variables having the instance dimension are
        instance variables, i.e. variables holding time series data.
    ds : xarray.Dataset
        Indexed ragged array dataset.
    instance_variables : list
        Per-instance metadata variables (over the instance dimension).
    observation_variables : list
        Measurement variables (over the sample dimension).
    instance_ids : list
        List of instance ids.

    Methods
    -------
    sel_instance(i)
        Read time series for given instance.
    iter()
        Yield time series for each instance.
    """

    def __init__(self, ds: xr.Dataset, index_var: str, sample_dim: str,
                 instance_id_var: str = None):
        """
        Initialize.

        Parameters
        ----------
        ds : xr.Dataset
            Data in indexed ragged array structure.
        index_var : str
            Index variable name.
        sample_dim : str
            Sample dimension name.
        instance_id_var : str, optional
            Variable holding the instance identifiers (e.g. "location_id") when
            they are not stored on the instance dimension. If None, the instance
            dimension coordinate is used, or a positional 0..N-1 index.
        """
        self.index_var = index_var
        self.sample_dim = sample_dim
        self.instance_id_var = instance_id_var
        # index_var is a (non-dimension) coordinate, but we deliberately do not
        # build an xarray/pandas index on it: sel_instance uses a precomputed
        # sort-by-index offset table instead (see _set_instance_lut), which is
        # faster and lighter than a label lookup over the duplicated index.
        self._data = ds.set_coords(self.index_var)
        self.validate()

        self.instance_dim = ds[index_var].attrs["instance_dimension"]
        self._set_instance_lut()

    def validate(self):
        """Validate format."""
        verify_indexed_ragged(self.ds, self.index_var, self.sample_dim)

    def __repr__(self):
        """"""
        return self.ds.__repr__()

    def _set_instance_lut(self):
        """
        Set instance lookup-table.

        Uses a binary-search lookup over the sorted instance ids, so memory
        scales with the number of instances rather than the largest id value.
        """
        if self.instance_id_var is not None:
            instance_ids = self.ds[self.instance_id_var].to_numpy()
        elif self.instance_dim in self.ds:
            instance_ids = self.ds[self.instance_dim].to_numpy()
        else:
            instance_ids = np.unique(self.ds[self.index_var])
        self._lookup = _InstanceLookup(instance_ids)

        # precomputed sort-by-index access: obs indices grouped by instance
        # position, so an instance's samples are a contiguous slice of _order
        # (O(1) offset lookup, like the contiguous ragged array)
        n_inst = instance_ids.size
        index = self.ds[self.index_var].to_numpy()
        self._order = np.argsort(index, kind="stable")
        self._counts = np.bincount(index, minlength=n_inst)
        self._starts = np.cumsum(self._counts) - self._counts
        # when the samples are already grouped by instance (common: produced by
        # to_indexed, and typical for cell files), each instance is a contiguous
        # slice and we can avoid the fancy-index gather entirely
        self._grouped = index.size == 0 or bool(np.all(index[:-1] <= index[1:]))

    @classmethod
    def from_file(cls, filename: str, index_var: str, sample_dim: str,
                  instance_id_var: str = None):
        """
        Read data from file.

        Parameters
        ----------
        filename : str
            Filename.
        index_var : str
            Index variable name.
        sample_dim : str
            Sample dimension name.
        instance_id_var : str, optional
            Variable holding the instance identifiers (default: None).

        Returns
        -------
        data : IndexRaggedArray
            IndexRaggedArray object loaded from a file.
        """
        ds = xr.open_dataset(filename)
        verify_indexed_ragged(ds, index_var, sample_dim)
        return cls(ds, index_var, sample_dim, instance_id_var=instance_id_var)

    def save(self, filename: str):
        """
        Write data to file with CF metadata finalized.

        Parameters
        ----------
        filename : str
            Filename (".nc" or ".zarr").
        """
        _save_cf(self.ds, filename, "timeSeries",
                 instance_id_var=self.instance_id_var)

    @property
    def ds(self) -> xr.Dataset:
        """
        Dataset.

        Returns
        -------
        ds : xr.Dataset
            Indexed ragged array dataset.
        """
        return self._data

    @property
    def size(self) -> int:
        """
        Number of instances.

        Returns
        -------
        instance_ids : int
            Number of instance.
        """
        return self.instance_ids.size

    @property
    def instance_ids(self) -> list:
        """
        Instance ids.

        Returns
        -------
        instance_ids : list of int
            Instance ids.
        """
        if self.instance_id_var is not None:
            return self.ds[self.instance_id_var].values
        return self.ds[self.instance_dim].values

    @property
    def instance_variables(self) -> list:
        """
        Instance variables (dimensioned by the instance dimension only, i.e.
        the per-instance metadata such as lon/lat/location_id).

        Returns
        -------
        instance_variables : list of str
            Instance variable names.
        """
        return [
            var for var in self.ds.variables
            if (self.ds[var].dims == (self.instance_dim,)) and
            (var != self.instance_dim) and
            ("sample_dimension" not in self.ds[var].attrs)
        ]

    @property
    def observation_variables(self) -> list:
        """
        Observation variables (dimensioned by the sample dimension, i.e. the
        measurements that vary per observation).

        Returns
        -------
        observation_variables : list of str
            Observation variable names.
        """
        return [
            var for var in self.ds.variables
            if (self.ds[var].dims == (self.sample_dim,)) and
            (var != self.sample_dim)
        ]

    def sel_instance(self, i: int) -> xr.Dataset:
        """
        Read time series.

        Parameters
        ----------
        i : int
            Instance identifier.

        Returns
        -------
        ds : xr.Dataset or None
            Time series for instance, or None if the instance is not present.
        """
        pos = int(self._lookup.positions(i)[0])
        if pos == -1:
            return None
        # the instance's samples are a contiguous slice of the precomputed
        # sort-by-index order; select them and the instance-level data by
        # position (no label lookup / index needed)
        start = int(self._starts[pos])
        end = start + int(self._counts[pos])
        sample_idx = slice(start, end) if self._grouped \
            else self._order[start:end]
        data = self.ds.isel({self.sample_dim: sample_idx,
                             self.instance_dim: pos})

        # reset index variable or drop index variable (my preference)?
        data[self.index_var] = (self.sample_dim,
                                np.zeros(data[self.index_var].size, dtype=int))

        return data

    def _wrap(self, ds: xr.Dataset) -> "IndexedRaggedArray":
        """Wrap a subset dataset in a new IndexedRaggedArray."""
        return IndexedRaggedArray(ds, self.index_var, self.sample_dim,
                                  instance_id_var=self.instance_id_var)

    def sel_instances(self,
                      i: np.array,
                      ignore_missing: bool = True) -> "IndexedRaggedArray":
        """
        Select several instances, preserving request order.

        Parameters
        ----------
        i : numpy.array
            Instance identifier(s).

        Returns
        -------
        data : IndexedRaggedArray
            An indexed ragged array with the selected instances.
        """
        i = np.atleast_1d(np.asarray(i))
        positions = self._lookup.positions(i)

        if ignore_missing:
            keep = positions != -1
            if not keep.any():
                return None
        else:
            if np.any(positions == -1):
                raise ValueError("Missing instances selected")
            keep = np.ones(i.size, dtype=bool)

        i = i[keep]
        positions = positions[keep]

        # keep samples whose index-variable position is among the selected ones
        sample_mask = np.isin(self.ds[self.index_var].values, positions)
        data = self.ds.isel({self.sample_dim: sample_mask})

        # remap each old position to its new index (request order of i)
        new_index = _InstanceLookup(positions).positions(
            data[self.index_var].values)
        data[self.index_var] = (self.sample_dim, new_index)
        # select the instance-level data by position (request order)
        data = data.isel({self.instance_dim: positions})

        # copy attributes
        data[self.index_var].attrs = self.ds[self.index_var].attrs

        return self._wrap(data)

    def __iter__(self) -> xr.Dataset:
        """
        Iterator over instances.

        Returns
        -------
        ds : xr.Dataset
            Time series for instance.
        """
        for i in self.instance_ids:
            yield self.sel_instance(i)

    def iter(self) -> xr.Dataset:
        """
        Explicit iterator method.

        Returns
        -------
        ds : xr.Dataset
            Time series for instance.
        """
        return self.__iter__()

    def to_contiguous(self,
                      count_var: str = "row_size") -> ContiguousRaggedArray:
        """
        Convert to contiguous ragged array.

        Parameters
        ----------
        count_var : str, optional
            Count variable (default: "row_size").

        Returns
        -------
        data : ContiguousRaggedArray
            Contiguous ragged array time series.
        """
        ds = indexed_to_contiguous(self.ds, self.sample_dim, self.instance_dim,
                                   count_var, self.index_var)

        return ContiguousRaggedArray(ds, count_var, self.instance_dim,
                                     instance_id_var=self.instance_id_var)

    def to_incomplete(self) -> "IncompleteMultidimArray":
        """
        Convert to an incomplete multidimensional array (via a contiguous array).

        Returns
        -------
        data : IncompleteMultidimArray
            Incomplete multidimensional array time series.
        """
        return self.to_contiguous().to_incomplete()

    def to_orthogonal(self, element_coord: str, element_dim: str = None,
                      strict: bool = True) -> "OrthogonalMultidimArray":
        """
        Convert to an orthogonal multidimensional array (via a contiguous array).

        Returns
        -------
        data : OrthogonalMultidimArray
            Orthogonal multidimensional array time series.
        """
        return self.to_contiguous().to_orthogonal(
            element_coord, element_dim=element_dim, strict=strict)

    def to_point_data(self):
        """
        Convert to point data.

        Instance-level variables are broadcast to the sample dimension so that
        every observation carries its instance's coordinates.

        Returns
        -------
        data : PointData
            Point data.
        """
        ds = indexed_to_point(self.ds, self.sample_dim, self.instance_dim,
                              self.index_var)
        return PointData(ds, self.sample_dim)

    def apply(self, func):
        """Apply a function to each instance's time series."""
        return [func(ts) for ts in self]

    def append(self, other):
        """
        Append another indexed ragged array (in place).

        The two collections are merged by instance (via a contiguous ragged
        array): observations of shared instances are combined, and instances
        present in only one are added.

        Parameters
        ----------
        other : IndexedRaggedArray or xarray.Dataset
            Indexed ragged array to append (same structure).
        """
        other_ds = other.ds if isinstance(other, IndexedRaggedArray) else other
        other_ira = IndexedRaggedArray(
            other_ds, self.index_var, self.sample_dim,
            instance_id_var=self.instance_id_var)

        cra = self.to_contiguous()
        cra.append(other_ira.to_contiguous())
        new_ds = cra.to_indexed(index_var=self.index_var).ds
        self.__init__(new_ds, self.index_var, self.sample_dim,
                      instance_id_var=self.instance_id_var)


def _find_marker_var(ds: xr.Dataset, attr: str) -> str:
    """Return the name of the variable carrying a CF marker attribute."""
    for v in ds.variables:
        if attr in ds[v].attrs:
            return str(v)
    raise ValueError(f"No variable with a '{attr}' attribute found.")


def _infer_sample_dim(ds: xr.Dataset) -> str:
    """Infer the sample dimension of a point dataset."""
    if len(ds.dims) == 1:
        return str(list(ds.dims)[0])
    for v in ds.data_vars:
        if ds[v].ndim == 1:
            return str(ds[v].dims[0])
    raise ValueError("Could not infer the sample dimension; pass 'sample_dim'.")


def open_cf(source, instance_id_var: str = None, sample_dim: str = None,
            instance_dim: str = None, element_dim: str = None,
            element_coord: str = None, **kwargs):
    """
    Open a CF discrete sampling geometry dataset or file and return the matching
    wrapper.

    The representation is auto-detected with
    :func:`ascat.cf_conversions.detect_cf_representation`. Contiguous ragged,
    indexed ragged and point datasets are configured automatically from their CF
    marker attributes; multidimensional arrays require ``instance_dim`` and
    ``element_dim``.

    Parameters
    ----------
    source : str, pathlib.Path or xarray.Dataset
        A file path, or an already-open dataset.
    instance_id_var : str, optional
        Variable holding the instance identifiers (e.g. "location_id").
    sample_dim : str, optional
        Sample dimension name (point data; inferred if not given).
    instance_dim, element_dim : str, optional
        Instance / element dimension names (required for multidimensional
        arrays).
    element_coord : str, optional
        Shared element coordinate name (orthogonal arrays; default element_dim).
    **kwargs
        Passed to :func:`xarray.open_dataset` when ``source`` is a path.

    Returns
    -------
    data : PointData, ContiguousRaggedArray, IndexedRaggedArray, \
OrthogonalMultidimArray or IncompleteMultidimArray
        The wrapper matching the detected representation.
    """
    ds = source if isinstance(source, xr.Dataset) \
        else xr.open_dataset(source, **kwargs)

    kind = detect_cf_representation(ds)

    if kind == "contiguous":
        count_var = _find_marker_var(ds, "sample_dimension")
        instance_dim = instance_dim or ds[count_var].dims[0]
        return ContiguousRaggedArray(ds, count_var, instance_dim,
                                     instance_id_var=instance_id_var)
    if kind == "indexed":
        index_var = _find_marker_var(ds, "instance_dimension")
        sample_dim = sample_dim or ds[index_var].dims[0]
        return IndexedRaggedArray(ds, index_var, sample_dim,
                                  instance_id_var=instance_id_var)
    if kind == "point":
        return PointData(ds, sample_dim or _infer_sample_dim(ds))

    # multidimensional array (orthogonal / incomplete)
    if instance_dim is None or element_dim is None:
        raise ValueError(
            f"'{kind}' multidimensional array requires 'instance_dim' and "
            "'element_dim' to be specified.")
    if kind == "orthogonal":
        return OrthogonalMultidimArray(ds, instance_dim, element_dim,
                                       element_coord=element_coord,
                                       instance_id_var=instance_id_var)
    return IncompleteMultidimArray(ds, instance_dim, element_dim,
                                   instance_id_var=instance_id_var)

