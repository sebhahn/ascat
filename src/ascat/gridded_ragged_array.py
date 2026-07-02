# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

"""
Grid-aware reading of ragged-array cell files.

A :mod:`pygeogrids` ``CellGrid`` links grid point indices (gpi) and coordinates
to cell numbers, and each cell is stored in its own ragged-array file. A gpi (or
the nearest gpi to a lon/lat) is translated to a cell, the cell file is located
and read with the matching :mod:`ascat.ragged_array` class, and the single grid
point's time series is returned.

Use :class:`GriddedContiguousRaggedArray` for contiguous ragged cell files and
:class:`GriddedIndexedRaggedArray` for indexed ragged cell files.
"""

from pathlib import Path
from typing import Union

import numpy as np
from pygeogrids.grids import CellGrid

from ascat.grids.grid_registry import GridRegistry
from ascat.ragged_array import (
    ContiguousRaggedArray,
    IndexedRaggedArray,
    OrthogonalMultidimArray,
)


class GriddedRaggedArray:
    """
    Base class: read gridded ragged-array cell files by grid point or location.

    A ``CellGrid`` maps grid point indices (gpi) and coordinates to cell numbers;
    each cell is stored in its own ragged-array file. A gpi (or the nearest gpi
    to a lon/lat) is translated to a cell via ``grid.gpi2cell``, the cell file is
    located with ``fn_format`` and read, and the grid point's time series is
    selected by its identifier.

    Two caching strategies control how cell files are kept in memory:

    - ``cache=False`` (default): only the most recently read cell is kept, like
      pynetcf's ``GriddedNcTs``. Efficient for reads that stay within one cell;
      re-reads when hopping between cells.
    - ``cache=True``: every cell that is read is retained in memory. Efficient
      when reading many grid points spread across (and revisiting) cells, at the
      cost of memory.

    This base class is not used directly; use
    :class:`GriddedContiguousRaggedArray` or :class:`GriddedIndexedRaggedArray`.

    Parameters
    ----------
    root_path : str or pathlib.Path
        Directory containing the cell files (searched recursively).
    grid : pygeogrids.grids.CellGrid or str
        Grid object, or a name resolvable by
        :class:`~ascat.grids.grid_registry.GridRegistry` (e.g. "fibgrid_12.5").
    fn_format : str, optional
        Format string for cell file names, formatted with the cell number, e.g.
        "{:04d}.nc" (default) or "H120_{:04d}.nc".
    cache : bool, optional
        Keep every read cell in memory instead of only the last one
        (default: False).
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        grid: Union[CellGrid, str],
        fn_format: str = "{:04d}.nc",
        cache: bool = False,
    ):
        self.root_path = Path(root_path)
        self.grid = GridRegistry().get(grid) if isinstance(grid, str) else grid
        self.fn_format = fn_format
        self.cache = cache

        # cache=True: {cell: ragged array}
        self._cells = {}
        # cache=False: only the most recently opened cell is kept
        self._active_cell = None
        self._active_reader = None

    def _open_cell(self, cell: int):
        """Open the file for ``cell`` and return a ragged-array reader."""
        raise NotImplementedError

    def _cell_filename(self, cell: int) -> Path:
        """Locate the file for a cell under ``root_path``."""
        name = self.fn_format.format(cell)
        matches = sorted(self.root_path.glob("**/" + name))
        if not matches:
            raise FileNotFoundError(
                f"No cell file for cell {cell} ('{name}') under "
                f"'{self.root_path}'.")
        return matches[0]

    def read_cell(self, cell: int):
        """
        Read a whole cell.

        Parameters
        ----------
        cell : int
            Cell number.

        Returns
        -------
        data : ContiguousRaggedArray or IndexedRaggedArray
            Ragged array for the cell (cached per the caching strategy).
        """
        cell = int(cell)
        if self.cache:
            if cell not in self._cells:
                self._cells[cell] = self._open_cell(cell)
            return self._cells[cell]

        if cell != self._active_cell:
            self._active_reader = self._open_cell(cell)
            self._active_cell = cell
        return self._active_reader

    def gpi_from_coords(
        self, lon: float, lat: float, max_dist: float = np.inf
    ) -> int:
        """
        Nearest grid point index to a lon/lat.

        Parameters
        ----------
        lon, lat : float
            Coordinates.
        max_dist : float, optional
            Maximum allowed distance in meters (default: no limit).

        Returns
        -------
        gpi : int
            Grid point index of the nearest grid point.

        Raises
        ------
        ValueError
            If the nearest grid point is farther than ``max_dist``.
        """
        gpi, dist = self.grid.find_nearest_gpi(lon, lat)
        if dist > max_dist:
            raise ValueError(
                f"No grid point within {max_dist} m of ({lon}, {lat}); "
                f"nearest is {dist:.0f} m away.")
        return int(gpi)

    def read(
        self,
        gpi=None,
        lon=None,
        lat=None,
        max_dist: float = np.inf,
    ):
        """
        Read the time series of one or more grid points.

        Either ``gpi`` or both ``lon`` and ``lat`` must be given. Each may be a
        scalar (single grid point) or array-like (many grid points at once).

        Parameters
        ----------
        gpi : int or array-like of int, optional
            Grid point index/indices.
        lon, lat : float or array-like of float, optional
            Coordinates; the nearest grid point to each is used.
        max_dist : float, optional
            Maximum allowed distance in meters for a lon/lat lookup. For a
            single lon/lat this raises if exceeded; for arrays, coordinates
            beyond it are dropped.

        Returns
        -------
        ds : xarray.Dataset or None, or dict of {gpi: (xarray.Dataset or None)}
            For a scalar request, the time series (or None if the grid point has
            no data in its cell file). For an array request, a dict mapping each
            grid point index to its time series (or None). Cell files are read
            once per cell, so grouping many grid points is efficient.
        """
        if gpi is None:
            if lon is None or lat is None:
                raise ValueError("Provide either 'gpi' or both 'lon' and 'lat'.")
            if np.ndim(lon) == 0:
                gpi = self.gpi_from_coords(lon, lat, max_dist)
            else:
                near, dist = self.grid.find_nearest_gpi(lon, lat)
                gpi = np.asarray(near)[np.asarray(dist) <= max_dist]

        if np.ndim(gpi) == 0:
            cell = int(self.grid.gpi2cell(gpi))
            return self.read_cell(cell).sel_instance(int(gpi))

        return self._read_gpis(np.asarray(gpi))

    def _read_gpis(self, gpis: np.ndarray) -> dict:
        """Read many grid points, reading each cell file only once."""
        cells = np.atleast_1d(self.grid.gpi2cell(gpis))
        per_gpi = {}
        for cell in np.unique(cells):
            reader = self.read_cell(int(cell))
            for g in gpis[cells == cell]:
                per_gpi[int(g)] = reader.sel_instance(int(g))
        # return in the order of the requested gpis
        return {int(g): per_gpi[int(g)] for g in gpis}

    @staticmethod
    def _close_reader(reader):
        """Close a reader's underlying dataset, if any."""
        if reader is not None:
            reader.ds.close()

    def close(self):
        """Close all open cell files and drop cached cells."""
        for reader in self._cells.values():
            self._close_reader(reader)
        self._close_reader(self._active_reader)
        self._cells.clear()
        self._active_cell = None
        self._active_reader = None

    def clear_cache(self):
        """Close all open cell files and drop cached cells (alias of close)."""
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class GriddedContiguousRaggedArray(GriddedRaggedArray):
    """
    Read gridded contiguous ragged-array cell files.

    See :class:`GriddedRaggedArray` for the gridded reading and caching
    behavior. Each cell is read with
    :class:`~ascat.ragged_array.ContiguousRaggedArray`.

    Parameters
    ----------
    root_path : str or pathlib.Path
        Directory containing the cell files.
    grid : pygeogrids.grids.CellGrid or str
        Grid object or registry name.
    fn_format : str, optional
        Cell file name format (default: "{:04d}.nc").
    count_var : str, optional
        Count variable name (default: "row_size").
    instance_dim : str, optional
        Instance dimension name (default: "locations").
    instance_id_var : str, optional
        Variable holding the instance identifiers, matching the grid gpi
        (default: "location_id").
    trim : bool, optional
        Drop fill/padding locations when reading a cell (default: True).
    cache : bool, optional
        Keep every read cell in memory (default: False).
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        grid: Union[CellGrid, str],
        fn_format: str = "{:04d}.nc",
        count_var: str = "row_size",
        instance_dim: str = "locations",
        instance_id_var: str = "location_id",
        trim: bool = True,
        cache: bool = False,
    ):
        super().__init__(root_path, grid, fn_format=fn_format, cache=cache)
        self.count_var = count_var
        self.instance_dim = instance_dim
        self.instance_id_var = instance_id_var
        self.trim = trim

    def _open_cell(self, cell: int) -> ContiguousRaggedArray:
        return ContiguousRaggedArray.from_file(
            self._cell_filename(cell),
            count_var=self.count_var,
            instance_dim=self.instance_dim,
            instance_id_var=self.instance_id_var,
            trim=self.trim,
        )


class GriddedIndexedRaggedArray(GriddedRaggedArray):
    """
    Read gridded indexed ragged-array cell files.

    See :class:`GriddedRaggedArray` for the gridded reading and caching
    behavior. Each cell is read with
    :class:`~ascat.ragged_array.IndexedRaggedArray`.

    Parameters
    ----------
    root_path : str or pathlib.Path
        Directory containing the cell files.
    grid : pygeogrids.grids.CellGrid or str
        Grid object or registry name.
    fn_format : str, optional
        Cell file name format (default: "{:04d}.nc").
    index_var : str, optional
        Index variable name (default: "locationIndex").
    sample_dim : str, optional
        Sample dimension name (default: "obs").
    instance_id_var : str, optional
        Variable holding the instance identifiers, matching the grid gpi
        (default: "location_id").
    cache : bool, optional
        Keep every read cell in memory (default: False).
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        grid: Union[CellGrid, str],
        fn_format: str = "{:04d}.nc",
        index_var: str = "locationIndex",
        sample_dim: str = "obs",
        instance_id_var: str = "location_id",
        cache: bool = False,
    ):
        super().__init__(root_path, grid, fn_format=fn_format, cache=cache)
        self.index_var = index_var
        self.sample_dim = sample_dim
        self.instance_id_var = instance_id_var

    def _open_cell(self, cell: int) -> IndexedRaggedArray:
        return IndexedRaggedArray.from_file(
            self._cell_filename(cell),
            index_var=self.index_var,
            sample_dim=self.sample_dim,
            instance_id_var=self.instance_id_var,
        )


class GriddedOrthoMultiArray(GriddedRaggedArray):
    """
    Read gridded orthogonal multidimensional array cell files (CF 9.3.1).

    See :class:`GriddedRaggedArray` for the gridded reading and caching
    behavior. Each cell is read with
    :class:`~ascat.ragged_array.OrthogonalMultidimArray`; all locations in a
    cell share the same element (e.g. time) axis.

    Parameters
    ----------
    root_path : str or pathlib.Path
        Directory containing the cell files.
    grid : pygeogrids.grids.CellGrid or str
        Grid object or registry name.
    fn_format : str, optional
        Cell file name format (default: "{:04d}.nc").
    instance_dim : str, optional
        Instance dimension name (default: "locations").
    element_dim : str, optional
        Element dimension name (default: "time").
    element_coord : str, optional
        Shared element coordinate variable name (default: ``element_dim``).
    instance_id_var : str, optional
        Variable holding the instance identifiers, matching the grid gpi
        (default: "location_id").
    cache : bool, optional
        Keep every read cell in memory (default: False).
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        grid: Union[CellGrid, str],
        fn_format: str = "{:04d}.nc",
        instance_dim: str = "locations",
        element_dim: str = "time",
        element_coord: str = None,
        instance_id_var: str = "location_id",
        cache: bool = False,
    ):
        super().__init__(root_path, grid, fn_format=fn_format, cache=cache)
        self.instance_dim = instance_dim
        self.element_dim = element_dim
        self.element_coord = element_coord
        self.instance_id_var = instance_id_var

    def _open_cell(self, cell: int) -> OrthogonalMultidimArray:
        return OrthogonalMultidimArray.from_file(
            self._cell_filename(cell),
            instance_dim=self.instance_dim,
            element_dim=self.element_dim,
            element_coord=self.element_coord,
            instance_id_var=self.instance_id_var,
        )
