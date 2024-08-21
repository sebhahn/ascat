# Copyright (c) 2024, TU Wien, Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of TU Wien, Department of Geodesy and Geoinformation
#      nor the names of its contributors may be used to endorse or promote
#      products derived from this software without specific prior written
#      permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL TU WIEN DEPARTMENT OF GEODESY AND
# GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from ascat.file_handling import Filenames
from ascat.utils import get_toi_subset, get_roi_subset
from ascat.utils import mask_dtype_nans


class AscatFile(Filenames):
    """
    Class reading ASCAT files.
    """
    def __init__(self, filename, read_generic=False):
        """
        Initialize AscatFile.

        Parameters
        ----------
        filename : str
            Filename.
        read_generic : boolean, optional
            Convert original data field names to generic field names by default when
            reading (default: False).
        """
        super().__init__(filename)
        self.read_generic = read_generic

    def read(self, toi=None, roi=None, generic=None, to_xarray=False, **kwargs):
        """
        Read ASCAT Level 1b data.

        Parameters
        ----------
        toi : tuple of datetime, optional
            Filter data for given time of interest (default: None).
            e.g. (datetime(2020, 1, 1, 12), datetime(2020, 1, 2))
        roi : tuple of 4 float, optional
            Filter data for region of interest (default: None).
            e.g. latmin, lonmin, latmax, lonmax
        generic : boolean, optional
            Convert original data field names to generic field names. Defaults
            to the value of self.read_generic.
        to_xarray : boolean, optional
            Convert data to xarray.Dataset otherwise numpy.ndarray will be
            returned (default: False).

        Returns
        -------
        data : xarray.Dataset or numpy.ndarray
            ASCAT data.
        metadata : dict
            Metadata.

        Notes
        -----
        TODO Figure out if subsetting should be done before or after merging,
        and implement if necessary.
        """
        if generic is None:
            if to_xarray:
                generic = True
            else:
                generic = self.read_generic

        data, metadata = super().read(generic=generic, to_xarray=to_xarray, **kwargs)

        if to_xarray and generic:
            data = mask_dtype_nans(data)

        if toi:
            data = get_toi_subset(data, toi)

        if roi:
            data = get_roi_subset(data, roi)

        return data, metadata

    def read_period(self, dt_start, dt_end, **kwargs):
        """
        Read interval.

        Parameters
        ----------
        dt_start : datetime
            Start datetime.
        dt_end : datetime
            End datetime.

        Returns
        -------
        data : xarray.Dataset or numpy.ndarray
            ASCAT data.
        metadata : dict
            Metadata.
        """
        return self.read(toi=(dt_start, dt_end), **kwargs)
