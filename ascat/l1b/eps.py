"""
Module for reading L1b and L2 ASCAT EPS native format.
"""

import os
import numpy as np
import datetime as dt
from tempfile import NamedTemporaryFile
from gzip import GzipFile


from bs4 import BeautifulSoup, NavigableString, Comment

import matplotlib.dates as mpl_dates

# import pygenio.genio as genio

long_nan = -2 ** 31
ulong_nan = 2 ** 32 - 1
int_nan = -2 ** 15
uint_nan = 2 ** 16 - 1
byte_nan = -2 ** 7


class EPSProductTemplate(object):

    """
    Product template class
    """

    def short_cds_time(self):
        """
        short cds time
        """
        struct = np.dtype([('day', np.uint16), ('time', np.uint32)])

        return struct

    def long_cds_time(self):
        """
        long cds time
        """
        struct = np.dtype([('day', np.uint16),
                           ('ms', np.uint32),
                           ('mms', np.uint16)])

        return struct

    def general_time(self):
        """
        general time
        """
        struct = np.dtype([('yyyy', np.int8, 4),
                           ('mm', np.int8, 2),
                           ('dd', np.int8, 2),
                           ('hh', np.int8, 2),
                           ('mm', np.int8, 2),
                           ('ss', np.int8, 2),
                           ('z', np.int8, 1)])

        return struct

    def grh(self):
        """
        General header template.
        """
        struct = np.dtype([('record_class', np.ubyte),
                           ('instrument_group', np.ubyte),
                           ('record_subclass', np.ubyte),
                           ('record_subclass_version', np.ubyte),
                           ('record_size', np.uint32),
                           ('record_start_time', self.short_cds_time()),
                           ('record_stop_time', self.short_cds_time())])
        return struct

    def mphr(self):
        """
        Header template.
        """
        struct = np.dtype([('grh', self.grh()),
                           ('product_name', np.dtype('S67')),
                           ('parent_product_name_1', np.dtype('S67')),
                           ('parent_product_name_2', np.dtype('S67')),
                           ('parent_product_name_3', np.dtype('S67')),
                           ('parent_product_name_4', np.dtype('S67')),
                           ('instrument_id', np.dtype('S4')),
                           ('instrument_model', np.dtype('S3')),
                           ('product_type', np.dtype('S3')),
                           ('processing_level', np.dtype('S2')),
                           ('spacecraft_id', np.dtype('S3')),
                           ('sensing_start', np.dtype('S15')),
                           ('sensing_end', np.dtype('S15')),
                           ('sensing_start_theoretical', np.dtype('S15')),
                           ('sensing_start_dump_theorectical',
                            np.dtype('S15')),
                           ('processing_centre', np.dtype('S4')),
                           ('processor_major_version', np.dtype('S5')),
                           ('processor_minor_version', np.dtype('S5')),
                           ('format_major_version', np.dtype('S5')),
                           ('format_minor_version', np.dtype('S5')),
                           ('processing_time_start', np.dtype('S15')),
                           ('processing_time_end', np.dtype('S15')),
                           ('processing_mode', np.dtype('S1')),
                           ('disposition_mode', np.dtype('S1')),
                           ('receiving_ground_station', np.dtype('S3')),
                           ('receive_time_start', np.dtype('S15')),
                           ('receive_time_end', np.dtype('S15')),
                           ('orbit_start', np.dtype('S5')),
                           ('orbit_end', np.dtype('S5')),
                           ('actual_product_size', np.dtype('S11')),
                           ('state_vector_time', np.dtype('S18')),
                           ('semi_major_axis', np.dtype('S11')),
                           ('eccentricity', np.dtype('S11')),
                           ('inclination', np.dtype('S11')),
                           ('perigee_argument', np.dtype('S11')),
                           ('right_ascension', np.dtype('S11')),
                           ('mean_anomaly', np.dtype('S11')),
                           ('x_position', np.dtype('S11')),
                           ('y_position', np.dtype('S11')),
                           ('z_position', np.dtype('S11')),
                           ('x_velocity', np.dtype('S11')),
                           ('y_velocity', np.dtype('S11')),
                           ('z_velocity', np.dtype('S11')),
                           ('earth_sun_distance_ratio', np.dtype('S11')),
                           ('location_tolerance_radial', np.dtype('S11')),
                           ('location_tolerance_alongtrack', np.dtype('S11')),
                           ('location_tolerance_crosstrack', np.dtype('S11')),
                           ('yaw_error', np.dtype('S11')),
                           ('roll_error', np.dtype('S11')),
                           ('pitch_error', np.dtype('S11')),
                           ('subsat_latitude_start', np.dtype('S11')),
                           ('subsat_longitude_start', np.dtype('S11')),
                           ('subsat_latitude_end', np.dtype('S11')),
                           ('subsat_longitude_end', np.dtype('S11')),
                           ('leap_second', np.dtype('S2')),
                           ('leap_second_utc', np.dtype('S15')),
                           ('total_records', np.dtype('S6')),
                           ('total_mphr', np.dtype('S6')),
                           ('total_sphr', np.dtype('S6')),
                           ('total_ipr', np.dtype('S6')),
                           ('total_geadr', np.dtype('S6')),
                           ('total_giadr', np.dtype('S6')),
                           ('total_veadr', np.dtype('S6')),
                           ('total_viadr', np.dtype('S6')),
                           ('total_mdr', np.dtype('S6')),
                           ('count_degraded_inst_mdr', np.dtype('S6')),
                           ('count_degraded_proc_mdr', np.dtype('S6')),
                           ('count_degraded_inst_mdr_blocks', np.dtype('S6')),
                           ('count_degraded_proc_mdr_blocks', np.dtype('S6')),
                           ('duration_of_product', np.dtype('S8')),
                           ('milliseconds_of_data_present', np.dtype('S8')),
                           ('milliseconds_of_data_missing', np.dtype('S8')),
                           ('subsetted_product', np.dtype('S1'))])

        return struct

    def ipr(self):
        """
        ipr template.
        """
        struct = np.dtype([('grh', self.grh()),
                           ('target_record_class', np.ubyte),
                           ('target_instrument_group', np.ubyte),
                           ('target_record_subclass', np.ubyte),
                           ('target_record_offset', np.uint32)])
        return struct

    def geadr(self):
        """
        geadr template.
        """
        struct = np.dtype([('grh', self.grh()),
                           ('aux_data_pointer', np.ubyte, 100)])
        return struct

    def giadr_archive(self):
        """
        giadr template.
        """
        struct = np.dtype([('grh', self.grh()),
                           ('archive_facility', np.ubyte, 100),
                           ('space_time_subset', np.ubyte),
                           ('band_subset', np.ubyte),
                           ('full_product_record_start_time', np.ubyte, 6),
                           ('full_product_record_stop_time', np.ubyte, 6),
                           ('subsetted_product_record_start_time',
                            np.ubyte, 6),
                           ('subsetted_product_record_stop_time', np.ubyte, 6),
                           ('number_of_original_bands', np.int16),
                           ('number_of_bands_in_subset', np.int16),
                           ('bands_present_bitmap', np.ubyte, 20)])

        return struct

    def veadr(self):
        """
        veadr template.
        """
        struct = np.dtype([('grh', self.grh()),
                           ('aux_data_pointer', np.ubyte, 100)])
        return struct

    def record_dummy_mdr(self):
        """
        dummy mdr template.
        """
        struct = np.dtype([('grh', self.grh()),
                           ('spare_flag', np.ubyte)])
        return struct

    def sphr_asca_fmv_12_sc_1(self):
        """
        read_ascat_sphr_fmv_12
        """
        struct = np.dtype([('grh', self.grh()),
                           ('n_l1a_mdr', np.dtype('S8')),
                           ('n_l1a_mdr_b0', np.dtype('S8')),
                           ('n_l1a_mdr_b1', np.dtype('S8')),
                           ('n_l1a_mdr_b2', np.dtype('S8')),
                           ('n_l1a_mdr_b3', np.dtype('S8')),
                           ('n_l1a_mdr_b4', np.dtype('S8')),
                           ('n_l1a_mdr_b5', np.dtype('S8')),
                           ('n_gaps', np.dtype('S8')),
                           ('total_gaps_size', np.dtype('S8')),
                           ('n_hktm_packets_received', np.dtype('S8')),
                           ('n_f_noise', np.dtype('S8')),
                           ('n_f_pg', np.dtype('S8')),
                           ('n_v_pg', np.dtype('S8')),
                           ('n_f_filter', np.dtype('S8')),
                           ('n_v_filter', np.dtype('S8')),
                           ('n_f_pgp', np.dtype('S8')),
                           ('n_f_np', np.dtype('S8')),
                           ('n_f_orbit', np.dtype('S8')),
                           ('n_f_attitude', np.dtype('S8')),
                           ('n_f_omega', np.dtype('S8')),
                           ('n_f_man', np.dtype('S8')),
                           ('n_f_osv', np.dtype('S8')),
                           ('n_f_e_tel_pres', np.dtype('S8')),
                           ('n_f_e_tel_ir', np.dtype('S8')),
                           ('n_f_ce', np.dtype('S8')),
                           ('n_v_ce', np.dtype('S8')),
                           ('n_f_oa', np.dtype('S8')),
                           ('n_f_tel', np.dtype('S8')),
                           ('n_f_ref', np.dtype('S8')),
                           ('n_f_sa', np.dtype('S8')),
                           ('n_f_land', np.dtype('S8')),
                           ('n_f_geo', np.dtype('S8')),
                           ('n_f_sign', np.dtype('S8')),
                           ('n_l1b_mdr', np.dtype('S8')),
                           ('n_empty_s0_trip', np.dtype('S8')),
                           ('n_l1b_mdr_f', np.dtype('S8')),
                           ('n_empty_s0_trip_f', np.dtype('S8')),
                           ('n_l1b_mdr_m', np.dtype('S8')),
                           ('n_empty_s0_trip_m', np.dtype('S8')),
                           ('n_l1b_mdr_a', np.dtype('S8')),
                           ('n_empty_s0_trip_a', np.dtype('S8')),
                           ('n_f_kp_f', np.dtype('S8')),
                           ('n_f_usable_f', np.dtype('S8')),
                           ('n_f_f_f', np.dtype('S8')),
                           ('n_f_v_f', np.dtype('S8')),
                           ('n_f_oa_f', np.dtype('S8')),
                           ('n_f_sa_f', np.dtype('S8')),
                           ('n_f_tel_f', np.dtype('S8')),
                           ('n_f_ref_f', np.dtype('S8')),
                           ('n_f_land_f', np.dtype('S8')),
                           ('n_f_kp_m', np.dtype('S8')),
                           ('n_f_usable_m', np.dtype('S8')),
                           ('n_f_f_m', np.dtype('S8')),
                           ('n_f_v_m', np.dtype('S8')),
                           ('n_f_oa_m', np.dtype('S8')),
                           ('n_f_sa_m', np.dtype('S8')),
                           ('n_f_tel_m', np.dtype('S8')),
                           ('n_f_ref_m', np.dtype('S8')),
                           ('n_f_land_m', np.dtype('S8')),
                           ('n_f_kp_a', np.dtype('S8')),
                           ('n_f_usable_a', np.dtype('S8')),
                           ('n_f_f_a', np.dtype('S8')),
                           ('n_f_v_a', np.dtype('S8')),
                           ('n_f_oa_a', np.dtype('S8')),
                           ('n_f_sa_a', np.dtype('S8')),
                           ('n_f_tel_a', np.dtype('S8')),
                           ('n_f_ref_a', np.dtype('S8')),
                           ('n_f_land_a', np.dtype('S8')),
                           ('processing_message_1', np.dtype('S50')),
                           ('processing_message_2', np.dtype('S50'))])

        return struct

    def sphr_asca_fmv_11_sc_1(self):
        """
        read_ascat_sphr_fmv_12
        """
        struct = np.dtype([('grh', self.grh()),
                           ('n_l1a_mdr', np.dtype('S8')),
                           ('n_l1a_mdr_b0', np.dtype('S8')),
                           ('n_l1a_mdr_b1', np.dtype('S8')),
                           ('n_l1a_mdr_b2', np.dtype('S8')),
                           ('n_l1a_mdr_b3', np.dtype('S8')),
                           ('n_l1a_mdr_b4', np.dtype('S8')),
                           ('n_l1a_mdr_b5', np.dtype('S8')),
                           ('n_gaps', np.dtype('S8')),
                           ('total_gaps_size', np.dtype('S8')),
                           ('n_hktm_packets_received', np.dtype('S8')),
                           ('n_f_echo', np.dtype('S8')),
                           ('n_m_echo', np.dtype('S8')),
                           ('n_c_echo', np.dtype('S8')),
                           ('n_i_echo', np.dtype('S8')),
                           ('n_f_noise', np.dtype('S8')),
                           ('n_m_noise', np.dtype('S8')),
                           ('n_c_noise', np.dtype('S8')),
                           ('n_i_noise', np.dtype('S8')),
                           ('n_f_pg', np.dtype('S8')),
                           ('n_v_pg', np.dtype('S8')),
                           ('n_f_ext_pg', np.dtype('S8')),
                           ('n_f_filter', np.dtype('S8')),
                           ('n_v_filter', np.dtype('S8')),
                           ('n_f_ext_filter', np.dtype('S8')),
                           ('n_f_tel_filter', np.dtype('S8')),
                           ('n_f_orbit', np.dtype('S8')),
                           ('n_f_attitude', np.dtype('S8')),
                           ('n_f_omega', np.dtype('S8')),
                           ('n_f_man', np.dtype('S8')),
                           ('n_f_dsl', np.dtype('S8')),
                           ('n_f_e_tel_pres', np.dtype('S8')),
                           ('n_f_e_tel_ir', np.dtype('S8')),
                           ('n_f_ce', np.dtype('S8')),
                           ('n_v_ce', np.dtype('S8')),
                           ('n_f_oa', np.dtype('S8')),
                           ('n_f_tel', np.dtype('S8')),
                           ('n_f_sa', np.dtype('S8')),
                           ('n_f_land', np.dtype('S8')),
                           ('n_l1b_mdr', np.dtype('S8')),
                           ('n_empty_s0_trip', np.dtype('S8')),
                           ('n_l1b_mdr_f', np.dtype('S8')),
                           ('n_empty_s0_trip_f', np.dtype('S8')),
                           ('n_l1b_mdr_m', np.dtype('S8')),
                           ('n_empty_s0_trip_m', np.dtype('S8')),
                           ('n_l1b_mdr_a', np.dtype('S8')),
                           ('n_empty_s0_trip_a', np.dtype('S8')),
                           ('n_f_kp_f', np.dtype('S8')),
                           ('n_f_usable_f', np.dtype('S8')),
                           ('avg_f_f_f', np.dtype('S8')),
                           ('avg_f_v_f', np.dtype('S8')),
                           ('avg_f_oa_f', np.dtype('S8')),
                           ('avg_f_sa_f', np.dtype('S8')),
                           ('avg_f_tel_f', np.dtype('S8')),
                           ('avg_f_ext_fil_f', np.dtype('S8')),
                           ('avg_f_land_f', np.dtype('S8')),
                           ('n_f_kp_m', np.dtype('S8')),
                           ('n_f_usable_m', np.dtype('S8')),
                           ('avg_f_f_m', np.dtype('S8')),
                           ('avg_f_v_m', np.dtype('S8')),
                           ('avg_f_oa_m', np.dtype('S8')),
                           ('avg_f_sa_m', np.dtype('S8')),
                           ('avg_f_tel_m', np.dtype('S8')),
                           ('avg_f_ext_fil_m', np.dtype('S8')),
                           ('avg_f_land_m', np.dtype('S8')),
                           ('n_f_kp_a', np.dtype('S8')),
                           ('n_f_usable_a', np.dtype('S8')),
                           ('avg_f_f_a', np.dtype('S8')),
                           ('avg_f_v_a', np.dtype('S8')),
                           ('avg_f_oa_a', np.dtype('S8')),
                           ('avg_f_sa_a', np.dtype('S8')),
                           ('avg_f_tel_a', np.dtype('S8')),
                           ('avg_f_ext_fil_a', np.dtype('S8')),
                           ('avg_f_land_a', np.dtype('S8')),
                           ('processing_message_1', np.dtype('S50')),
                           ('processing_message_2', np.dtype('S50'))])

        return struct

    def viadr_asca_fmv_11_sc_4(self):
        """
        VIADR-OA
        """
        struct = np.dtype([('grh', self.grh()),
                           ('ac_utc_time', self.long_cds_time()),
                           ('ac_sv_position', np.int64, 3),
                           ('ac_sv_velocity', np.int64, 3),
                           ('att_ys_law', np.int32, 3),
                           ('att_dist_law', np.int32, (3, 3, 4))])

        return struct

    def viadr_asca_fmv_11_sc_6(self):
        """
        VIADR SUBCLASS 7
        """
        struct = np.dtype([('grh', self.grh()),
                           ('processor_version1', np.uint8),
                           ('processor_version2', np.uint8),
                           ('processor_version3', np.uint8),
                           ('prc_version1', np.uint8),
                           ('prc_version2', np.uint8),
                           ('ins_version1', np.uint8),
                           ('ins_version2', np.uint8),
                           ('ntb_version1', np.uint8),
                           ('ntb_version2', np.uint8),
                           ('deb_version1', np.uint8),
                           ('deb_version2', np.uint8)])

        return struct

    def viadr_asca_fmv_12_sc_4(self):
        """
        VIADR-OA
        """
        struct = np.dtype([('grh', self.grh()),
                           ('ac_utc_time', self.long_cds_time()),
                           ('ac_sv_position', np.int64, 3),
                           ('ac_sv_velocity', np.int64, 3),
                           ('att_ys_law', np.int32, 3),
                           ('att_dist_law', np.int32, (3, 3, 4))])

        return struct

    def viadr_asca_fmv_12_sc_6(self):
        """
        VIADR-VER
        """
        struct = np.dtype([('grh', self.grh()),
                           ('processor_version1', np.int8),
                           ('processor_version2', np.int8),
                           ('processor_version3', np.int8),
                           ('prc_version1', np.int8),
                           ('prc_version2', np.int8),
                           ('ins_version1', np.int8),
                           ('ins_version2', np.int8),
                           ('ntb_version1', np.int8),
                           ('ntb_version2', np.int8),
                           ('xcl_version1', np.int8),
                           ('xcl_version2', np.int8)])

        return struct

    def viadr_asca_fmv_12_sc_7(self):
        """
        VIADR SUBCLASS 7 version 2
        """
        struct = np.dtype([('grh', self.grh()),
                           ('processor_version1', np.uint8),
                           ('processor_version2', np.uint8),
                           ('processor_version3', np.uint8),
                           ('prc_version1', np.uint8),
                           ('prc_version2', np.uint8),
                           ('ins_version1', np.uint8),
                           ('ins_version2', np.uint8),
                           ('ntb_version1', np.uint8),
                           ('ntb_version2', np.uint8),
                           ('xcl_version1', np.uint8),
                           ('xcl_version2', np.uint8),
                           ('somo_processor_version1', np.uint8),
                           ('somo_processor_version2', np.uint8),
                           ('somo_processor_version3', np.uint8),
                           ('smc_version1', np.uint8),
                           ('smc_version2', np.uint8),
                           ('curv_version', np.uint8),
                           ('curv_noise_version', np.uint8),
                           ('dry_version', np.uint8),
                           ('dry_noise_version', np.uint8),
                           ('ms_mean_version', np.uint8),
                           ('nonscat_version', np.uint8),
                           ('slop_version', np.uint8),
                           ('slop_noise_version', np.uint8),
                           ('wet_version', np.uint8),
                           ('wet_noise_version', np.uint8)])

        return struct

    def viadr_asca_fmv_12_sc_8(self):
        """
        VIADR-GRID
        """

        num_nodes = 81

        struct = np.dtype([('grh', self.grh()),
                           ('utc_line_nodes', self.short_cds_time()),
                           ('abs_line_number', np.int32),
                           ('latitude_left', np.int32, num_nodes),
                           ('longitude_left', np.int32, num_nodes),
                           ('latitude_right', np.int32, num_nodes),
                           ('longitude_right', np.int32, num_nodes)])

        return struct

    def mdr_asca_fmv_10_sc_4(self):
        """
        MDR L2 SM-25KM SUBCLASS 4
        """

        num_nodes = 82

        struct = np.dtype([('grh', self.grh()),
                           ('utc_line_nodes', self.short_cds_time()),
                           ('sat_track_azi', np.uint16),
                           ('node_num', np.int16, num_nodes),
                           ('swath_indicator', np.uint8, num_nodes),
                           ('latitude', np.int32, num_nodes),
                           ('longitude', np.int32, num_nodes),
                           ('sigma0_trip', np.int32, (num_nodes, 3)),
                           ('kp', np.uint16, (num_nodes, 3)),
                           ('inc_angle_trip', np.uint16, (num_nodes, 3)),
                           ('azi_angle_trip', np.int16, (num_nodes, 3)),
                           ('f_kp', np.uint8, (num_nodes, 3)),
                           ('f_usable', np.uint8, (num_nodes, 3)),
                           ('f_land', np.uint16, (num_nodes, 3)),
                           ('warp_nrt_version', np.uint16),
                           ('param_db_version', np.uint16),
                           ('soil_moisture', np.uint16, num_nodes),
                           ('soil_moisture_error', np.uint16, num_nodes),
                           ('sigma40', np.int32, num_nodes),
                           ('sigma40_error', np.int32, num_nodes),
                           ('slope40', np.int32, num_nodes),
                           ('slope40_error', np.int32, num_nodes),
                           ('soil_moisture_sensitivity', np.uint32, num_nodes),
                           ('dry_backscatter', np.int32, num_nodes),
                           ('wet_backscatter', np.int32, num_nodes),
                           ('mean_surf_soil_moisture', np.uint16, num_nodes),
                           ('rainfall_flag', np.uint8, num_nodes),
                           ('correction_flags', np.uint8, num_nodes),
                           ('processing_flags', np.uint16, num_nodes),
                           ('aggregated_quality_flag', np.uint8, num_nodes),
                           ('snow_cover_probability', np.uint8, num_nodes),
                           ('frozen_soil_probability', np.uint8, num_nodes),
                           ('inundation_or_wetland', np.uint8, num_nodes),
                           ('topographical_complexity', np.uint8, num_nodes)])

        return struct

    def mdr_asca_fmv_10_sc_5(self):
        """
        MDR L2 SM-50KM SUBCLASS 5
        """
        num_nodes = 42

        struct = np.dtype([('grh', self.grh()),
                           ('utc_line_nodes', self.short_cds_time()),
                           ('sat_track_azi', np.uint16),
                           ('node_num', np.int16, num_nodes),
                           ('swath_indicator', np.uint8, num_nodes),
                           ('latitude', np.int32, num_nodes),
                           ('longitude', np.int32, num_nodes),
                           ('sigma0_trip', np.int32, (num_nodes, 3)),
                           ('kp', np.uint16, (num_nodes, 3)),
                           ('inc_angle_trip', np.uint16, (num_nodes, 3)),
                           ('azi_angle_trip', np.int16, (num_nodes, 3)),
                           ('f_kp', np.uint8, (num_nodes, 3)),
                           ('f_usable', np.uint8, (num_nodes, 3)),
                           ('f_land', np.uint16, (num_nodes, 3)),
                           ('warp_nrt_version', np.uint16),
                           ('param_db_version', np.uint16),
                           ('soil_moisture', np.uint16, num_nodes),
                           ('soil_moisture_error', np.uint16, num_nodes),
                           ('sigma40', np.int32, num_nodes),
                           ('sigma40_error', np.int32, num_nodes),
                           ('slope40', np.int32, num_nodes),
                           ('slope40_error', np.int32, num_nodes),
                           ('soil_moisture_sensitivity', np.uint32, num_nodes),
                           ('dry_backscatter', np.int32, num_nodes),
                           ('wet_backscatter', np.int32, num_nodes),
                           ('mean_surf_soil_moisture', np.uint16, num_nodes),
                           ('rainfall_flag', np.uint8, num_nodes),
                           ('correction_flags', np.uint8, num_nodes),
                           ('processing_flags', np.uint16, num_nodes),
                           ('aggregated_quality_flag', np.uint8, num_nodes),
                           ('snow_cover_probability', np.uint8, num_nodes),
                           ('frozen_soil_probability', np.uint8, num_nodes),
                           ('inundation_or_wetland', np.uint8, num_nodes),
                           ('topographical_complexity', np.uint8, num_nodes)])

        return struct

    def mdr_asca_fmv_11_sc_1(self):
        """
        MDR SUBCLASS 1
        """
        num_nodes = 82

        struct = np.dtype([('grh', self.grh()),
                           ('utc_line_nodes', self.short_cds_time()),
                           ('sat_track_azi', np.uint16),
                           ('node_num', np.int16, num_nodes),
                           ('swath_indicator', np.uint8, num_nodes),
                           ('latitude', np.int32, num_nodes),
                           ('longitude', np.int32, num_nodes),
                           ('atmospheric_height', np.uint16, num_nodes),
                           ('atmospheric_loss', np.uint32, num_nodes),
                           ('sigma0_trip', np.int32, (num_nodes, 3)),
                           ('kp', np.uint16, (num_nodes, 3)),
                           ('inc_angle_trip', np.uint16, (num_nodes, 3)),
                           ('azi_angle_trip', np.int16, (num_nodes, 3)),
                           ('f_kp', np.uint8, (num_nodes, 3)),
                           ('f_usable', np.uint8, (num_nodes, 3)),
                           ('f_f', np.uint16, (num_nodes, 3)),
                           ('f_v', np.uint16, (num_nodes, 3)),
                           ('f_oa', np.uint16, (num_nodes, 3)),
                           ('f_sa', np.uint16, (num_nodes, 3)),
                           ('f_tel', np.uint16, (num_nodes, 3)),
                           ('f_ext_fil', np.uint16, (num_nodes, 3)),
                           ('f_land', np.uint16, (num_nodes, 3))])

        return struct

    def mdr_asca_fmv_11_sc_2(self):
        """
        MDR SUBCLASS 2
        """
        num_nodes = 42

        struct = np.dtype([('grh', self.grh()),
                           ('utc_line_nodes', self.short_cds_time()),
                           ('sat_track_azi', np.uint16),
                           ('node_num', np.uint8, num_nodes),
                           ('swath_indicator', np.uint8, num_nodes),
                           ('latitude', np.int32, num_nodes),
                           ('longitude', np.int32, num_nodes),
                           ('atmospheric_height', np.int16, num_nodes),
                           ('atmospheric_loss', np.int32, num_nodes),
                           ('sigma0_trip', np.int32, (num_nodes, 3)),
                           ('kp', np.uint16, (num_nodes, 3)),
                           ('inc_angle_trip', np.uint16, (num_nodes, 3)),
                           ('azi_angle_trip', np.int16, (num_nodes, 3)),
                           ('f_kp', np.uint8, (num_nodes, 3)),
                           ('f_usable', np.uint8, (num_nodes, 3)),
                           ('f_f', np.uint16, (num_nodes, 3)),
                           ('f_v', np.uint16, (num_nodes, 3)),
                           ('f_oa', np.uint16, (num_nodes, 3)),
                           ('f_sa', np.uint16, (num_nodes, 3)),
                           ('f_tel', np.uint16, (num_nodes, 3)),
                           ('f_ext_fil', np.uint16, (num_nodes, 3)),
                           ('f_land', np.uint16, (num_nodes, 3))])

        return struct

    def mdr_asca_fmv_11_sc_4(self):
        """
        MDR L2 SM-25KM SUBCLASS 4
        """
        num_nodes = 82

        struct = np.dtype([('grh', self.grh()),
                           ('degraded_inst_mdr', np.uint8),
                           ('degraded_proc_mdr', np.uint8),
                           ('utc_line_nodes', self.short_cds_time()),
                           ('sat_track_azi', np.uint16),
                           ('node_num', np.int16, num_nodes),
                           ('swath_indicator', np.uint8, num_nodes),
                           ('latitude', np.int32, num_nodes),
                           ('longitude', np.int32, num_nodes),
                           ('atmospheric_height', np.uint16, num_nodes),
                           ('atmospheric_loss', np.uint32, num_nodes),
                           ('sigma0_trip', np.int32, (num_nodes, 3)),
                           ('kp', np.uint16, (num_nodes, 3)),
                           ('inc_angle_trip', np.uint16, (num_nodes, 3)),
                           ('azi_angle_trip', np.int16, (num_nodes, 3)),
                           ('f_kp', np.uint8, (num_nodes, 3)),
                           ('f_usable', np.uint8, (num_nodes, 3)),
                           ('f_f', np.uint16, (num_nodes, 3)),
                           ('f_v', np.uint16, (num_nodes, 3)),
                           ('f_oa', np.uint16, (num_nodes, 3)),
                           ('f_sa', np.uint16, (num_nodes, 3)),
                           ('f_tel', np.uint16, (num_nodes, 3)),
                           ('f_ext_fil', np.uint16, (num_nodes, 3)),
                           ('f_land', np.uint16, (num_nodes, 3)),
                           ('warp_nrt_version', np.uint16),
                           ('param_db_version', np.uint16),
                           ('soil_moisture', np.uint16, num_nodes),
                           ('soil_moisture_error', np.uint16, num_nodes),
                           ('sigma40', np.int32, num_nodes),
                           ('sigma40_error', np.int32, num_nodes),
                           ('slope40', np.int32, num_nodes),
                           ('slope40_error', np.int32, num_nodes),
                           ('soil_moisture_sensitivity', np.uint32, num_nodes),
                           ('dry_backscatter', np.int32, num_nodes),
                           ('wet_backscatter', np.int32, num_nodes),
                           ('mean_surf_soil_moisture', np.uint16, num_nodes),
                           ('rainfall_flag', np.uint8, num_nodes),
                           ('correction_flags', np.uint8, num_nodes),
                           ('processing_flags', np.uint16, num_nodes),
                           ('aggregated_quality_flag', np.uint8, num_nodes),
                           ('snow_cover_probability', np.uint8, num_nodes),
                           ('frozen_soil_probability', np.uint8, num_nodes),
                           ('inundation_or_wetland', np.uint8, num_nodes),
                           ('topographical_complexity', np.uint8, num_nodes)])

        return struct

    def mdr_asca_fmv_11_sc_5(self):
        """
        MDR L2 SM-25KM SUBCLASS 5
        """
        num_nodes = 42

        struct = np.dtype([('grh', self.grh()),
                           ('degraded_inst_mdr', np.uint8),
                           ('degraded_proc_mdr', np.uint8),
                           ('utc_line_nodes', self.short_cds_time()),
                           ('sat_track_azi', np.uint16),
                           ('node_num', np.int16, num_nodes),
                           ('swath_indicator', np.uint8, num_nodes),
                           ('latitude', np.int32, num_nodes),
                           ('longitude', np.int32, num_nodes),
                           ('atmospheric_height', np.uint16, num_nodes),
                           ('atmospheric_loss', np.uint32, num_nodes),
                           ('sigma0_trip', np.int32, (num_nodes, 3)),
                           ('kp', np.uint16, (num_nodes, 3)),
                           ('inc_angle_trip', np.uint16, (num_nodes, 3)),
                           ('azi_angle_trip', np.int16, (num_nodes, 3)),
                           ('f_kp', np.uint8, (num_nodes, 3)),
                           ('f_usable', np.uint8, (num_nodes, 3)),
                           ('f_f', np.uint16, (num_nodes, 3)),
                           ('f_v', np.uint16, (num_nodes, 3)),
                           ('f_oa', np.uint16, (num_nodes, 3)),
                           ('f_sa', np.uint16, (num_nodes, 3)),
                           ('f_tel', np.uint16, (num_nodes, 3)),
                           ('f_ext_fil', np.uint16, (num_nodes, 3)),
                           ('f_land', np.uint16, (num_nodes, 3)),
                           ('warp_nrt_version', np.uint16),
                           ('param_db_version', np.uint16),
                           ('soil_moisture', np.uint16, num_nodes),
                           ('soil_moisture_error', np.uint16, num_nodes),
                           ('sigma40', np.int32, num_nodes),
                           ('sigma40_error', np.int32, num_nodes),
                           ('slope40', np.int32, num_nodes),
                           ('slope40_error', np.int32, num_nodes),
                           ('soil_moisture_sensitivity', np.uint32, num_nodes),
                           ('dry_backscatter', np.int32, num_nodes),
                           ('wet_backscatter', np.int32, num_nodes),
                           ('mean_surf_soil_moisture', np.uint16, num_nodes),
                           ('rainfall_flag', np.uint8, num_nodes),
                           ('correction_flags', np.uint8, num_nodes),
                           ('processing_flags', np.uint16, num_nodes),
                           ('aggregated_quality_flag', np.uint8, num_nodes),
                           ('snow_cover_probability', np.uint8, num_nodes),
                           ('frozen_soil_probability', np.uint8, num_nodes),
                           ('inundation_or_wetland', np.uint8, num_nodes),
                           ('topographical_complexity', np.uint8, num_nodes)])

        return struct

    def mdr_asca_fmv_12_sc_1(self):
        """
        MDR SUBCLASS 1
        """
        num_nodes = 82

        struct = np.dtype([('grh', self.grh()),
                           ('degraded_inst_mdr', np.uint8),
                           ('degraded_proc_mdr', np.uint8),
                           ('utc_line_nodes', self.short_cds_time()),
                           ('abs_line_number', np.int32),
                           ('sat_track_azi', np.uint16),
                           ('as_des_pass', np.int8),
                           ('swath_indicator', np.uint8, num_nodes),
                           ('latitude', np.int32, num_nodes),
                           ('longitude', np.int32, num_nodes),
                           ('sigma0_trip', np.int32, (num_nodes, 3)),
                           ('kp', np.uint16, (num_nodes, 3)),
                           ('inc_angle_trip', np.uint16, (num_nodes, 3)),
                           ('azi_angle_trip', np.int16, (num_nodes, 3)),
                           ('num_val_trip', np.uint32, (num_nodes, 3)),
                           ('f_kp', np.uint8, (num_nodes, 3)),
                           ('f_usable', np.uint8, (num_nodes, 3)),
                           ('f_f', np.uint16, (num_nodes, 3)),
                           ('f_v', np.uint16, (num_nodes, 3)),
                           ('f_oa', np.uint16, (num_nodes, 3)),
                           ('f_sa', np.uint16, (num_nodes, 3)),
                           ('f_tel', np.uint16, (num_nodes, 3)),
                           ('f_ref', np.uint16, (num_nodes, 3)),
                           ('f_land', np.uint16, (num_nodes, 3))])

        return struct

    def mdr_asca_fmv_12_sc_2(self):
        """
        MDR SUBCLASS 2
        """
        num_nodes = 42

        struct = np.dtype([('grh', self.grh()),
                           ('degraded_inst_mdr', np.uint8),
                           ('degraded_proc_mdr', np.uint8),
                           ('utc_line_nodes', self.short_cds_time()),
                           ('abs_line_number', np.int32),
                           ('sat_track_azi', np.uint16),
                           ('as_des_pass', np.int8),
                           ('swath_indicator', np.uint8, num_nodes),
                           ('latitude', np.int32, num_nodes),
                           ('longitude', np.int32, num_nodes),
                           ('sigma0_trip', np.int32, (num_nodes, 3)),
                           ('kp', np.uint16, (num_nodes, 3)),
                           ('inc_angle_trip', np.uint16, (num_nodes, 3)),
                           ('azi_angle_trip', np.int16, (num_nodes, 3)),
                           ('num_val_trip', np.uint32, (num_nodes, 3)),
                           ('f_kp', np.uint8, (num_nodes, 3)),
                           ('f_usable', np.uint8, (num_nodes, 3)),
                           ('f_f', np.uint16, (num_nodes, 3)),
                           ('f_v', np.uint16, (num_nodes, 3)),
                           ('f_oa', np.uint16, (num_nodes, 3)),
                           ('f_sa', np.uint16, (num_nodes, 3)),
                           ('f_tel', np.uint16, (num_nodes, 3)),
                           ('f_ref', np.uint16, (num_nodes, 3)),
                           ('f_land', np.uint16, (num_nodes, 3))])

        return struct

    def mdr_asca_fmv_12_sc_3(self):
        """
        MDR SUBCLASS 2
        """
        num_meas = 192

        struct = np.dtype([('grh', self.grh()),
                           ('degraded_inst_mdr', np.uint8),
                           ('degraded_proc_mdr', np.uint8),
                           ('utc_localisation', self.short_cds_time()),
                           ('sat_track_azi', np.uint16),
                           ('as_des_pass', np.uint8),
                           ('beam_number', np.uint8),
                           ('sigma0_full', np.int32, num_meas),
                           ('inc_angle_full', np.uint16, num_meas),
                           ('azi_angle_full', np.int16, num_meas),
                           ('latitude_full', np.int32, num_meas),
                           ('longitude_full', np.int32, num_meas),
                           ('land_frac', np.uint16, num_meas),
                           ('flagfield_rf1', np.uint8),
                           ('flagfield_rf2', np.uint8),
                           ('flagfield_pl', np.uint8),
                           ('flagfield_gen1', np.uint8),
                           ('flagfield_gen2', np.uint8, num_meas)])

        return struct

    def mdr_asca_fmv_12_sc_4(self):
        """
        MDR L2 SM-25KM SUBCLASS 4
        """
        num_nodes = 82

        struct = np.dtype([('grh', self.grh()),
                           ('degraded_inst_mdr', np.uint8),
                           ('degraded_proc_mdr', np.uint8),
                           ('utc_line_nodes', self.short_cds_time()),
                           ('abs_line_number', np.int32),
                           ('sat_track_azi', np.uint16),
                           ('as_des_pass', np.uint8),
                           ('swath_indicator', np.uint8, num_nodes),
                           ('latitude', np.int32, num_nodes),
                           ('longitude', np.int32, num_nodes),
                           ('sigma0_trip', np.int32, (num_nodes, 3)),
                           ('kp', np.uint16, (num_nodes, 3)),
                           ('inc_angle_trip', np.uint16, (num_nodes, 3)),
                           ('azi_angle_trip', np.int16, (num_nodes, 3)),
                           ('num_val_trip', np.uint32, (num_nodes, 3)),
                           ('f_kp', np.uint8, (num_nodes, 3)),
                           ('f_usable', np.uint8, (num_nodes, 3)),
                           ('f_f', np.uint16, (num_nodes, 3)),
                           ('f_v', np.uint16, (num_nodes, 3)),
                           ('f_oa', np.uint16, (num_nodes, 3)),
                           ('f_sa', np.uint16, (num_nodes, 3)),
                           ('f_tel', np.uint16, (num_nodes, 3)),
                           ('f_ref', np.uint16, (num_nodes, 3)),
                           ('f_land', np.uint16, (num_nodes, 3)),
                           ('warp_nrt_version', np.uint16),
                           ('param_db_version', np.uint16),
                           ('soil_moisture', np.uint16, num_nodes),
                           ('soil_moisture_error', np.uint16, num_nodes),
                           ('sigma40', np.int32, num_nodes),
                           ('sigma40_error', np.int32, num_nodes),
                           ('slope40', np.int32, num_nodes),
                           ('slope40_error', np.int32, num_nodes),
                           ('soil_moisture_sensitivity', np.uint32, num_nodes),
                           ('dry_backscatter', np.int32, num_nodes),
                           ('wet_backscatter', np.int32, num_nodes),
                           ('mean_surf_soil_moisture', np.uint16, num_nodes),
                           ('rainfall_flag', np.uint8, num_nodes),
                           ('correction_flags', np.uint8, num_nodes),
                           ('processing_flags', np.uint16, num_nodes),
                           ('aggregated_quality_flag', np.uint8, num_nodes),
                           ('snow_cover_probability', np.uint8, num_nodes),
                           ('frozen_soil_probability', np.uint8, num_nodes),
                           ('inundation_or_wetland', np.uint8, num_nodes),
                           ('topographical_complexity', np.uint8, num_nodes)])

        return struct

    def mdr_asca_fmv_12_sc_5(self):
        """
        MDR L2 SM-25KM SUBCLASS 5
        """
        num_nodes = 42

        struct = np.dtype([('grh', self.grh()),
                           ('degraded_inst_mdr', np.uint8),
                           ('degraded_proc_mdr', np.uint8),
                           ('utc_line_nodes', self.short_cds_time()),
                           ('abs_line_number', np.int32),
                           ('sat_track_azi', np.uint16),
                           ('as_des_pass', np.uint8),
                           ('swath_indicator', np.uint8, num_nodes),
                           ('latitude', np.int32, num_nodes),
                           ('longitude', np.int32, num_nodes),
                           ('sigma0_trip', np.int32, (num_nodes, 3)),
                           ('kp', np.uint16, (num_nodes, 3)),
                           ('inc_angle_trip', np.uint16, (num_nodes, 3)),
                           ('azi_angle_trip', np.int16, (num_nodes, 3)),
                           ('num_val_trip', np.uint32, (num_nodes, 3)),
                           ('f_kp', np.uint8, (num_nodes, 3)),
                           ('f_usable', np.uint8, (num_nodes, 3)),
                           ('f_f', np.uint16, (num_nodes, 3)),
                           ('f_v', np.uint16, (num_nodes, 3)),
                           ('f_oa', np.uint16, (num_nodes, 3)),
                           ('f_sa', np.uint16, (num_nodes, 3)),
                           ('f_tel', np.uint16, (num_nodes, 3)),
                           ('f_ref', np.uint16, (num_nodes, 3)),
                           ('f_land', np.uint16, (num_nodes, 3)),
                           ('warp_nrt_version', np.uint16),
                           ('param_db_version', np.uint16),
                           ('soil_moisture', np.uint16, num_nodes),
                           ('soil_moisture_error', np.uint16, num_nodes),
                           ('sigma40', np.int32, num_nodes),
                           ('sigma40_error', np.int32, num_nodes),
                           ('slope40', np.int32, num_nodes),
                           ('slope40_error', np.int32, num_nodes),
                           ('soil_moisture_sensitivity', np.uint32, num_nodes),
                           ('dry_backscatter', np.int32, num_nodes),
                           ('wet_backscatter', np.int32, num_nodes),
                           ('mean_surf_soil_moisture', np.uint16, num_nodes),
                           ('rainfall_flag', np.uint8, num_nodes),
                           ('correction_flags', np.uint8, num_nodes),
                           ('processing_flags', np.uint16, num_nodes),
                           ('aggregated_quality_flag', np.uint8, num_nodes),
                           ('snow_cover_probability', np.uint8, num_nodes),
                           ('frozen_soil_probability', np.uint8, num_nodes),
                           ('inundation_or_wetland', np.uint8, num_nodes),
                           ('topographical_complexity', np.uint8, num_nodes)])

        return struct


class EPSProduct(object):

    """
    Class for reading EPS products.
    """

    def __init__(self, filename):

        self.is_zipped = False
        if os.path.splitext(filename)[1] == '.gz':
            self.is_zipped = True

        self.filename = filename
        self.fid = None
        self.filesize = 0

        self.grh = None
        self.mphr = None
        self.sphr = None
        self.ipr = []
        self.geadr = []
        self.giadr_archive = None
        self.veadr = []
        self.viadr = []
        self.viadr_grid = []
        self.dummy_mdr = []
        self.mdr = []
        self.eor = 0
        self.bor = 0
        self.mdr_counter = 0

    def read(self):
        """
        Read an EPS file.
        """
        if self.is_zipped:
            with NamedTemporaryFile(delete=False) as tmp_fid:
                with GzipFile(self.filename) as gz_fid:
                    tmp_fid.write(gz_fid.read())
                filename = tmp_fid.name
        else:
            filename = self.filename

        self.fid = open(filename, 'rb')
        self.filesize = os.path.getsize(filename)
        self.eor = self.fid.tell()

        while self.eor < self.filesize:

            # remember beginning of the record
            self.bor = self.fid.tell()

            # just read grh of current dataset
            self._read_grh()

            # return pointer to the beginning of the record
            self.fid.seek(self.bor)

            if self.grh['record_class'] == 1:
                self._read_mphr()

            elif self.grh['record_class'] == 2:
                self._read_sphr()

            elif self.grh['record_class'] == 3:
                dtype = EPSProductTemplate().ipr()
                self.ipr.append(self._read_record(dtype))

            elif self.grh['record_class'] == 4:
                dtype = EPSProductTemplate().geadr()
                self.geadr.append(self._read_record(dtype))

            elif self.grh['record_class'] == 5:
                if self.grh['record_subclass'] == 99:
                    dtype = EPSProductTemplate().giadr_archive()
                    self.giadr_archive = self._read_record(dtype)
                else:
                    raise RuntimeError("Record class not found!")

            elif self.grh['record_class'] == 6:
                dtype = EPSProductTemplate().veadr()
                self.veadr.append(self._read_record(dtype))

            elif self.grh['record_class'] == 7:
                if self.grh['record_subclass'] == 8:
                    dtype = self._get_dtype('viadr')
                    self.viadr_grid.append(self._read_record(dtype))
                else:
                    dtype = self._get_dtype('viadr')
                    self.viadr.append(self._read_record(dtype))

            elif self.grh['record_class'] == 8:
                if self.grh['instrument_group'] == 13:
                    dtype = EPSProductTemplate().record_dummy_mdr()
                    self.dummy_mdr.append(self._read_record(dtype))
                else:
                    dtype = self._get_dtype('mdr')
                    self.mdr.append(self._read_record(dtype))
                    self.mdr_counter += 1

            else:
                raise RuntimeError("Record class not found!")

            # Determine number of bytes read
            self.eor = self.fid.tell()

        self.fid.close()

        if self.is_zipped:
            os.remove(filename)

    def _skip(self, recordsize):
        self.fid.seek(self.fid.tell() + recordsize)

    def _read_grh(self):
        dtype = EPSProductTemplate().grh()
        self.grh = self._read_record(dtype)

    def _read_mphr(self):
        self.mphr = np.zeros((1,), dtype=EPSProductTemplate().mphr())
        self._read_grh()
        for name in self.mphr.dtype.names:
            if name == "grh":
                self.mphr[name] = self.grh.copy()
                continue
            self.mphr[name] = self.fid.readline()[32:32 +
                                                  self.mphr[name].dtype.itemsize]

    def _read_sphr(self):
        self.sphr = np.zeros((1,), dtype=self._get_dtype("sphr"))
        self._read_grh()
        self.sphr['grh'] = self.grh.copy()

        for name in self.sphr.dtype.names:
            if name == "grh":
                self.sphr[name] = self.grh.copy()
                continue
            self.sphr[name] = self.fid.readline()[32:32 +
                                                  self.sphr[name].dtype.itemsize]

    def _read_record(self, dtype, count=1):
        record = np.fromfile(self.fid, dtype=dtype, count=count)
        return record.newbyteorder('B')

    def _get_dtype(self, name):
        """
        Get a generic product template.
        """
        func_name = ("".join((name, "_",
                              self.mphr['instrument_id'][0].strip(" "),
                              "_fmv_",
                              self.mphr['format_major_version'][0].strip(" "),
                              "_sc_",
                              str(self.grh['record_subclass'][0])))).lower()
        return eval("".join(("EPSProductTemplate().", func_name, "()")))


class NewEPSProduct(object):

    """
    Class for reading EPS products.
    """

    def __init__(self, filename):

        self.is_zipped = False
        if os.path.splitext(filename)[1] == '.gz':
            self.is_zipped = True

        self.filename = filename
        self.fid = None
        self.filesize = 0

        self.grh = None
        self.mphr = None
        self.sphr = None
        self.ipr = []
        self.geadr = []
        self.giadr_archive = None
        self.veadr = []
        self.viadr = []
        self.viadr_grid = []
        self.dummy_mdr = []
        self.mdr = []
        self.eor = 0
        self.bor = 0
        self.mdr_counter = 0

    def read(self):
        """
        Read an EPS file.
        """
        if self.is_zipped:
            with NamedTemporaryFile(delete=False) as tmp_fid:
                with GzipFile(self.filename) as gz_fid:
                    tmp_fid.write(gz_fid.read())
                filename = tmp_fid.name
        else:
            filename = self.filename

        self.fid = open(filename, 'rb')
        self.filesize = os.path.getsize(filename)
        self.eor = self.fid.tell()

        while self.eor < self.filesize:

            # remember beginning of the record
            self.bor = self.fid.tell()

            # just read grh of current dataset
            self._read_grh()

            # return pointer to the beginning of the record
            self.fid.seek(self.bor)

            if self.grh['record_class'] == 1:
                self.mphr = np.zeros((1,), dtype=EPSProductTemplate().mphr())
                self._read_grh()
                for name in self.mphr.dtype.names:
                    if name == "grh":
                        self.mphr[name] = self.grh.copy()
                        continue
                    self.mphr[name] = self.fid.readline()[32:32 +
                                                          self.mphr[name].dtype.itemsize]

            elif self.grh['record_class'] == 2:
                self.sphr = np.zeros((1,), dtype=self._get_dtype("sphr"))
                self._read_grh()
                self.sphr['grh'] = self.grh.copy()

                for name in self.sphr.dtype.names:
                    if name == "grh":
                        self.sphr[name] = self.grh.copy()
                        continue
                    self.sphr[name] = self.fid.readline()[32:32 +
                                                          self.sphr[name].dtype.itemsize]

            elif self.grh['record_class'] == 3:
                dtype = EPSProductTemplate().ipr()
                self.ipr.append(self._read_record(dtype))

            elif self.grh['record_class'] == 4:
                dtype = EPSProductTemplate().geadr()
                self.geadr.append(self._read_record(dtype))

            elif self.grh['record_class'] == 5:
                if self.grh['record_subclass'] == 99:
                    dtype = EPSProductTemplate().giadr_archive()
                    self.giadr_archive = self._read_record(dtype)
                else:
                    raise RuntimeError("Record class not found!")

            elif self.grh['record_class'] == 6:
                dtype = EPSProductTemplate().veadr()
                self.veadr.append(self._read_record(dtype))

            elif self.grh['record_class'] == 7:
                if self.grh['record_subclass'] == 8:
                    dtype = self._get_dtype('viadr')
                    self.viadr_grid.append(self._read_record(dtype))
                else:
                    dtype = self._get_dtype('viadr')
                    self.viadr.append(self._read_record(dtype))

            elif self.grh['record_class'] == 8:
                if self.grh['instrument_group'] == 13:
                    dtype = EPSProductTemplate().record_dummy_mdr()
                    self.dummy_mdr.append(self._read_record(dtype))
                else:
                    dtype = self._get_dtype('mdr')
                    self.mdr.append(self._read_record(dtype))
                    self.mdr_counter += 1

            else:
                raise RuntimeError("Record class not found!")

            # Determine number of bytes read
            self.eor = self.fid.tell()

        self.fid.close()

        if self.is_zipped:
            os.remove(filename)

    def _read_grh(self):
        dtype = EPSProductTemplate().grh()
        self.grh = self._read_record(dtype)

    def _read_record(self, dtype, count=1):
        record = np.fromfile(self.fid, dtype=dtype, count=count)
        return record.newbyteorder('B')

    def _get_dtype(self, name):
        """
        Get a generic product template.
        """
        func_name = ("".join((name, "_",
                              self.mphr['instrument_id'][0].strip(" "),
                              "_fmv_",
                              self.mphr['format_major_version'][0].strip(" "),
                              "_sc_",
                              str(self.grh['record_subclass'][0])))).lower()
        return eval("".join(("EPSProductTemplate().", func_name, "()")))


def shortcdstime2dtordinal(days, milliseconds):
    """
    Converting shortcdstime to datetime ordinal.

    Parameters
    ----------
    days : int
        Days.
    milliseconds : int
        Milliseconds

    Returns
    -------
    date : datetime.datetime
        Ordinal datetime.
    """
    epoch = dt.datetime.strptime('2000-01-01 00:00:00',
                                 '%Y-%m-%d %H:%M:%S').toordinal()
    offset = days + (milliseconds / 1000.) / (24. * 60. * 60.)
    return epoch + offset


def _read_szf_fmv_12(eps_file):
    """
    Read SZF format version 12.

    Parameters
    ----------
    eps_file : EPSProduct object
        EPS Product object.

    Returns
    -------
    data : numpy.ndarray
        SZF data.
    orbit_grid : numpy.ndarray
        6.25km orbit lat/lon grid.
    """
    raw_data = np.array(eps_file.mdr)[:, 0]
    template = genio.GenericIO.get_template("SZF__001")
    data = np.repeat(template, eps_file.mdr_counter * 192)
    idx_nodes = np.arange(eps_file.mdr_counter).repeat(192)

    data['jd'] = mpl_dates.num2julian(shortcdstime2dtordinal(
        raw_data['utc_localisation'].flatten()['day'],
        raw_data['utc_localisation'].flatten()['time']))[idx_nodes]

    data['spacecraft_id'] = np.int8(eps_file.mphr['spacecraft_id'][0][-1])

    fields = ['processor_major_version', 'processor_minor_version',
              'format_major_version', 'format_minor_version']
    for field in fields:
        data[field] = np.int16(eps_file.mphr[field][0])

    fields = [('degraded_inst_mdr', 1), ('degraded_proc_mdr', 1),
              ('sat_track_azi', 1e-2), ('as_des_pass', 1),
              ('beam_number', 1), ('flagfield_rf1', 1),
              ('flagfield_rf2', 1), ('flagfield_pl', 1),
              ('flagfield_gen1', 1)]
    for field in fields:
        data[field[0]] = raw_data[field[0]].flatten()[idx_nodes] * field[1]

    data['swath_indicator'] = np.int8(data['beam_number'].flatten() > 3)

    fields = [('lon', 'longitude_full', long_nan, 1e-6),
              ('lat', 'latitude_full', long_nan, 1e-6),
              ('sig', 'sigma0_full', long_nan, 1e-6),
              ('inc', 'inc_angle_full', uint_nan, 1e-2),
              ('azi', 'azi_angle_full', int_nan, 1e-2),
              ('land_frac', 'land_frac', uint_nan, 1e-2),
              ('flagfield_gen2', 'flagfield_gen2', byte_nan, 1)]
    for field in fields:
        data[field[0]] = raw_data[field[1]].flatten()
        valid = data[field[0]] != field[2]
        data[field[0]][valid] = data[field[0]][valid] * field[3]

    # modify longitudes from (0, 360) to (-180, 180)
    mask = (data['lon'] != long_nan) & (data['lon'] > 180)
    data['lon'][mask] += -360.

    # modify azimuth from (-180, 180) to (0, 360)
    mask = (data['azi'] != int_nan) & (data['azi'] < 0)
    data['azi'][mask] += 360

    viadr_grid = np.concatenate(eps_file.viadr_grid)
    orbit_grid = np.zeros(viadr_grid.size * 2 * 81,
                          dtype=np.dtype([('lon', np.float32),
                                          ('lat', np.float32),
                                          ('node_num', np.int16),
                                          ('line_num', np.int32)]))

    for pos_all in range(orbit_grid['lon'].size):
        line = pos_all / 162
        pos_small = pos_all % 81
        if (pos_all % 162 <= 80):
            # left swath
            orbit_grid['lon'][pos_all] = viadr_grid[
                'longitude_left'][line][80 - pos_small]
            orbit_grid['lat'][pos_all] = viadr_grid[
                'latitude_left'][line][80 - pos_small]
        else:
            # right swath
            orbit_grid['lon'][pos_all] = viadr_grid[
                'longitude_right'][line][pos_small]
            orbit_grid['lat'][pos_all] = viadr_grid[
                'latitude_right'][line][pos_small]

#     orbit_grid['lon'] = np.concatenate((viadr_grid['longitude_left'],
#                                     viadr_grid['longitude_right'])).flatten()
#     orbit_grid['lat'] = np.concatenate((viadr_grid['latitude_left'],
#                                     viadr_grid['latitude_right'])).flatten()

    n_node_per_line = 2 * 81
    orbit_grid['node_num'] = np.tile((np.arange(n_node_per_line) + 1),
                                     viadr_grid.size)

    lines = np.arange(0, viadr_grid.size * 2, 2)
    orbit_grid['line_num'] = np.repeat(lines, 2 * 81)

    fields = ['lon', 'lat']
    for field in fields:
        mask = orbit_grid[field] != long_nan
        orbit_grid[field] = orbit_grid[field] * 1e-6

    mask = (orbit_grid['lon'] != long_nan) & (orbit_grid['lon'] > 180)
    orbit_grid['lon'][mask] += -360.

    set_flags(data)

    data['as_des_pass'] = (data['sat_track_azi'] < 270).astype(np.uint8)

    return data, orbit_grid


def set_flags(data):
    """
    Compute summary flag for each measurement with a value of 0, 1 or 2
    indicating nominal, slightly degraded or severely degraded data.

    Parameters
    ----------
    data : numpy.ndarray
        SZF data.
    """

    # category:status = 'red': 2, 'amber': 1, 'warning': 0
    flag_status_bit = {'flagfield_rf1': {'2': [2, 4],
                                         '1': [0, 1, 3]},

                       'flagfield_rf2': {'2': [0, 1]},

                       'flagfield_pl': {'2': [0, 1, 2, 3],
                                        '0': [4]},

                       'flagfield_gen1': {'2': [1],
                                          '0': [0]},

                       'flagfield_gen2': {'2': [2],
                                          '1': [0],
                                          '0': [1]}
                       }

    for flagfield in flag_status_bit.keys():
        unpacked_bits = np.unpackbits(data[flagfield])

        set_bits = np.where(unpacked_bits == 1)[0]
        if (set_bits.size != 0):
            pos_8 = 7 - (set_bits % 8)

            for category in sorted(flag_status_bit[flagfield].keys()):
                if (int(category) == 0) and (flagfield != 'flagfield_gen2'):
                    continue

                for bit2check in flag_status_bit[flagfield][category]:
                    pos = np.where(pos_8 == bit2check)[0]
                    data['f_usable'][set_bits[pos] / 8] = int(category)

                    # land points
                    if (flagfield == 'flagfield_gen2') and (bit2check == 1):
                        data['f_land'][set_bits[pos] / 8] = 1


def _read_szx_fmv_12(eps_file):
    """
    Read SZO/SZR format version 12.

    Parameters
    ----------
    eps_file : EPSProduct object
        EPS Product object.

    Returns
    -------
    data : numpy.ndarray
        SZO/SZR data.
    """
    raw_data = np.array(eps_file.mdr)[:, 0]
    template = genio.GenericIO.get_template("SZX__002")
    n_node_per_line = eps_file.mdr[0]['longitude'].size
    n_records = eps_file.mdr_counter * n_node_per_line
    data = np.repeat(template, n_records)
    idx_nodes = np.arange(eps_file.mdr_counter).repeat(n_node_per_line)

    data['jd'] = mpl_dates.num2julian(shortcdstime2dtordinal(
        raw_data['utc_line_nodes'].flatten()['day'],
        raw_data['utc_line_nodes'].flatten()['time']))[idx_nodes]

    data['spacecraft_id'] = np.int8(eps_file.mphr['spacecraft_id'][0][-1])
    data['abs_orbit_nr'] = np.uint32(eps_file.mphr['orbit_start'])[0]

    fields = ['processor_major_version', 'processor_minor_version',
              'format_major_version', 'format_minor_version']
    for field in fields:
        data[field] = np.int16(eps_file.mphr[field][0])

    fields = [('degraded_inst_mdr', 1), ('degraded_proc_mdr', 1),
              ('sat_track_azi', 1e-2), ('as_des_pass', 1)]
    for field in fields:
        data[field[0]] = raw_data[field[0]].flatten()[idx_nodes] * field[1]

    fields = [('lon', 'longitude', long_nan, 1e-6),
              ('lat', 'latitude', long_nan, 1e-6),
              ('swath_indicator', 'swath_indicator', byte_nan, 1)]
    for field in fields:
        data[field[0]] = raw_data[field[1]].flatten()
        valid = data[field[0]] != field[2]
        data[field[0]][valid] = data[field[0]][valid] * field[3]

    fields = [('sig', 'sigma0_trip', long_nan, 1e-6),
              ('inc', 'inc_angle_trip', uint_nan, 1e-2),
              ('azi', 'azi_angle_trip', int_nan, 1e-2),
              ('kp', 'kp', uint_nan, 1e-2),
              ('num_val', 'num_val_trip', ulong_nan, 1),
              ('f_kp', 'f_kp', byte_nan, 1),
              ('f_usable', 'f_usable', byte_nan, 1),
              ('f_f', 'f_f', uint_nan, 1),
              ('f_v', 'f_v', uint_nan, 1),
              ('f_oa', 'f_oa', uint_nan, 1),
              ('f_sa', 'f_sa', uint_nan, 1),
              ('f_tel', 'f_tel', uint_nan, 1),
              ('f_ref', 'f_ref', uint_nan, 1),
              ('f_land', 'f_land', uint_nan, 1)]
    for field in fields:
        data[field[0]] = raw_data[field[1]].reshape(n_records, 3)
        valid = data[field[0]] != field[2]
        data[field[0]][valid] = data[field[0]][valid] * field[3]

    # modify longitudes from (0, 360) to (-180,180)
    mask = np.logical_and(data['lon'] != long_nan, data['lon'] > 180)
    data['lon'][mask] += -360.

    # modify azimuth from (-180, 180) to (0, 360)
    mask = (data['azi'] != int_nan) & (data['azi'] < 0)
    data['azi'][mask] += 360

    data['node_num'] = np.tile((np.arange(n_node_per_line) + 1),
                               eps_file.mdr_counter)

    data['line_num'] = idx_nodes

    data['as_des_pass'] = (data['sat_track_azi'] < 270).astype(np.uint8)

    return data


def _read_szx_fmv_11(eps_file):
    """
    Read SZO/SZR format version 11.

    Parameters
    ----------
    eps_file : EPSProduct object
        EPS Product object.

    Returns
    -------
    data : numpy.ndarray
        SZO/SZR data.
    """
    raw_data = np.array(eps_file.mdr)[:, 0]
    template = genio.GenericIO.get_template("SZX__002")
    n_node_per_line = eps_file.mdr[0]['longitude'].size
    n_records = eps_file.mdr_counter * n_node_per_line
    data = np.repeat(template, n_records)
    idx_nodes = np.arange(eps_file.mdr_counter).repeat(n_node_per_line)

    data['jd'] = mpl_dates.num2julian(shortcdstime2dtordinal(
        raw_data['utc_line_nodes'].flatten()['day'],
        raw_data['utc_line_nodes'].flatten()['time']))[idx_nodes]

    data['spacecraft_id'] = np.int8(eps_file.mphr['spacecraft_id'][0][-1])
    data['abs_orbit_nr'] = np.uint32(eps_file.mphr['orbit_start'])[0]

    fields = ['processor_major_version', 'processor_minor_version',
              'format_major_version', 'format_minor_version']
    for field in fields:
        data[field] = np.int16(eps_file.mphr[field][0])

    fields = [('sat_track_azi', 1e-2)]
    for field in fields:
        data[field[0]] = raw_data[field[0]].flatten()[idx_nodes] * field[1]

    fields = [('lon', 'longitude', long_nan, 1e-6),
              ('lat', 'latitude', long_nan, 1e-6),
              ('swath_indicator', 'swath_indicator', byte_nan, 1)]
    for field in fields:
        data[field[0]] = raw_data[field[1]].flatten()
        valid = data[field[0]] != field[2]
        data[field[0]][valid] = data[field[0]][valid] * field[3]

    fields = [('sig', 'sigma0_trip', long_nan, 1e-6),
              ('inc', 'inc_angle_trip', uint_nan, 1e-2),
              ('azi', 'azi_angle_trip', int_nan, 1e-2),
              ('kp', 'kp', uint_nan, 1e-2),
              ('f_kp', 'f_kp', byte_nan, 1),
              ('f_usable', 'f_usable', byte_nan, 1),
              ('f_f', 'f_f', uint_nan, 1),
              ('f_v', 'f_v', uint_nan, 1),
              ('f_oa', 'f_oa', uint_nan, 1),
              ('f_sa', 'f_sa', uint_nan, 1),
              ('f_tel', 'f_tel', uint_nan, 1),
              ('f_land', 'f_land', uint_nan, 1)]
    for field in fields:
        data[field[0]] = raw_data[field[1]].reshape(n_records, 3)
        valid = data[field[0]] != field[2]
        data[field[0]][valid] = data[field[0]][valid] * field[3]

    # modify longitudes from (0, 360) to (-180,180)
    mask = np.logical_and(data['lon'] != long_nan, data['lon'] > 180)
    data['lon'][mask] += -360.

    # modify azimuth from (-180, 180) to (0, 360)
    mask = (data['azi'] != int_nan) & (data['azi'] < 0)
    data['azi'][mask] += 360

    data['node_num'] = np.tile((np.arange(n_node_per_line) + 1),
                               eps_file.mdr_counter)

    data['line_num'] = idx_nodes

    data['as_des_pass'] = (data['sat_track_azi'] < 270).astype(np.uint8)

    return data


def read_ascat_l1b(filename):
    """
    Reader for SZF, SZO and SZR native EPS files.

    Parameters
    ----------
    filename : str
        EPS filename.

    Returns
    -------
    data : numpy.ndarry
        Data records.
    """
    eps_file = read_eps_nat(filename)
    ptype = eps_file.mphr['product_type'][0]
    fmv = int(eps_file.mphr['format_major_version'][0])

    if ptype == 'SZF':
        if fmv == 12:
            return _read_szf_fmv_12(eps_file)

    elif (ptype == 'SZR') or (ptype == 'SZO'):
        if fmv == 11:
            return _read_szx_fmv_11(eps_file)
        if fmv == 12:
            return _read_szx_fmv_12(eps_file)

    raise ValueError("Format not supported. Product type {:1}"
                     " Format major version: {:2}".format(ptype, fmv))


def read_ascat_l2(filename):
    """
    Reader for SMO and SMR native EPS files.

    Parameters
    ----------
    filename : str
        EPS filename.

    Returns
    -------
    data : numpy.ndarray
        Data records.
    """
    eps_file = read_eps_nat(filename)

    n_node_per_line = eps_file.mdr[0]['longitude'].size
    n_records = eps_file.mdr_counter * n_node_per_line
    idx_nodes = np.arange(eps_file.mdr_counter).repeat(n_node_per_line)

    template = genio.GenericIO.get_template("SMR__001")
    data = np.repeat(template, n_records)
    raw_data = np.array(eps_file.mdr)

    # utc_line_nodes (time)
    ascat_time = mpl_dates.num2julian(
        shortcdstime2dtordinal(raw_data['utc_line_nodes'].flatten()['day'],
                               raw_data['utc_line_nodes'].flatten()['time']))
    data['jd'] = ascat_time[idx_nodes]

    fields = [('sig', 'sigma0_trip', long_nan, 1e-6),
              ('inc', 'inc_angle_trip', uint_nan, 1e-2),
              ('azi', 'azi_angle_trip', int_nan, 1e-2),
              ('kp', 'kp', uint_nan, 1e-4),
              ('f_land', 'f_land', uint_nan, 1e-3)]

    for field in fields:
        data[field[0]] = raw_data[field[1]].reshape(n_records, 3)
        valid = data[field[0]] != field[2]
        data[field[0]][valid] = data[field[0]][valid] * field[3]

    fields = [('lon', 'longitude', long_nan, 1e-6),
              ('lat', 'latitude', long_nan, 1e-6),
              ('ssm', 'soil_moisture', uint_nan, 1e-2),
              ('ssm_noise', 'soil_moisture_error', uint_nan, 1e-2),
              ('norm_sigma', 'sigma40', long_nan, 1e-6),
              ('norm_sigma_noise', 'sigma40_error', long_nan, 1e-6),
              ('slope', 'slope40', long_nan, 1e-6),
              ('slope_noise', 'slope40_error', long_nan, 1e-6),
              ('dry_ref', 'dry_backscatter', long_nan, 1e-6),
              ('wet_ref', 'wet_backscatter', long_nan, 1e-6),
              ('mean_ssm', 'mean_surf_soil_moisture', uint_nan, 1e-2),
              ('ssm_sens', 'soil_moisture_sensitivity', ulong_nan, 1e-6),
              ('correction_flag', 'correction_flags', None, 1),
              ('processing_flag', 'processing_flags', None, 1),
              ('aggregated_flag', 'aggregated_quality_flag', None, 1),
              ('snow', 'snow_cover_probability', None, 1),
              ('frozen', 'frozen_soil_probability', None, 1),
              ('wetland', 'inundation_or_wetland', None, 1),
              ('topo', 'topographical_complexity', None, 1)]

    for field in fields:
        data[field[0]] = raw_data[field[1]].flatten()
        if field[2] is None:
            data[field[0]] = data[field[0]] * field[3]
        else:
            valid = data[field[0]] != field[2]
            data[field[0]][valid] = data[field[0]][valid] * field[3]

    # sat_track_azi (uint)
    data['as_des_pass'] = \
        np.array(raw_data['sat_track_azi'].flatten()[idx_nodes] < 27000)

    # modify longitudes from [0,360] to [-180,180]
    mask = np.logical_and(data['lon'] != long_nan, data['lon'] > 180)
    data['lon'][mask] += -360.

    # modify azimuth from (-180, 180) to (0, 360)
    mask = (data['azi'] != int_nan) & (data['azi'] < 0)
    data['azi'][mask] += 360

    data['param_db_version'] = \
        raw_data['param_db_version'].flatten()[idx_nodes]
    data['warp_nrt_version'] = \
        raw_data['warp_nrt_version'].flatten()[idx_nodes]

    data['spacecraft_id'] = int(eps_file.mphr['spacecraft_id'][0][2])

    lswath = raw_data['swath_indicator'].flatten() == 0
    rswath = raw_data['swath_indicator'].flatten() == 1

    if raw_data.dtype.fields.has_key('node_num') is False:
        if (n_node_per_line == 82):
            leftSw = np.arange(20, -21, -1)
            rightSw = np.arange(-20, 21, 1)

        if (n_node_per_line == 42):
            leftSw = np.arange(10, -11, -1)
            rightSw = np.arange(-10, 11, 1)

        lineNum = np.concatenate((leftSw, rightSw), axis=0).flatten()
        nodes = np.repeat(np.array(lineNum, ndmin=2),
                          raw_data['abs_line_number'].size, axis=0)

    if (n_node_per_line == 82):
        if raw_data.dtype.fields.has_key('node_num'):
            data['node_num'][lswath] = 21 + raw_data['node_num'].flat[lswath]
            data['node_num'][rswath] = 62 + raw_data['node_num'].flat[rswath]
        else:
            data['node_num'][lswath] = 21 + nodes.flat[lswath]
            data['node_num'][rswath] = 62 + nodes.flat[rswath]
    if (n_node_per_line == 42):
        if raw_data.dtype.fields.has_key('node_num'):
            data['node_num'][lswath] = 11 + raw_data['node_num'].flat[lswath]
            data['node_num'][rswath] = 32 + raw_data['node_num'].flat[rswath]
        else:
            data['node_num'][lswath] = 11 + nodes.flat[lswath]
            data['node_num'][rswath] = 32 + nodes.flat[rswath]

    return data


def test_xml():
    filename = '../../formats/eps_ascatl1bszr_9.0.xml'

    # from xml.dom import minidom
    # xmldoc = minidom.parse(filename)

    with open(filename) as fid:
        soup = BeautifulSoup(fid, 'lxml')

    # dtype_lut = {'enumerated': 'S',
    #              'string': 'S',
    #              'integer': np.int32,
    #              'uinteger': np.uint32,
    #              'time': 'S',
    #              'longtime': 'S',
    #              'bolean': np.bool}

    # dtype = []
    # fields = {}
    # for field in soup.product.mphr.find_all('field'):
    #     field_name = field.attrs['name']

    #     if 'scaling-factor' in field.attrs:
    #         scaling_factor = field.attrs['scaling-factor']
    #     else:
    #         scaling_factor = None

    #     if 'units' in field.attrs:
    #         units = field.attrs['units']
    #     else:
    #         units = None

    #     fields[field_name] = {'length': field.attrs['length'],
    #                           'type': field.attrs['type'],
    #                           'description': field.attrs['description'],
    #                           'scaling_factor': scaling_factor,
    #                           'units': units}

    #     dtype.append((field_name, 'S{:}'.format(field.attrs['length'])))

    # print(dtype)
    # x = np.dtype(dtype)

    # dtype = []

    # for field in soup.product.sphr.find_all('field'):
    #     field_name = field.attrs['name']

    #     if 'scaling-factor' in field.attrs:
    #         scaling_factor = field.attrs['scaling-factor']
    #     else:
    #         scaling_factor = None

    #     if 'units' in field.attrs:
    #         units = field.attrs['units']
    #     else:
    #         units = None

    #     fields[field_name] = {'length': field.attrs['length'],
    #                           'type': field.attrs['type'],
    #                           'description': field.attrs['description'],
    #                           'scaling_factor': scaling_factor,
    #                           'units': units}

    #     dtype.append((field_name, 'S{:}'.format(field.attrs['length'])))

    # print(dtype)

    # dtype = []

    # print(data)

    elements = {}
    for el in soup.product.children:

        if type(el) == NavigableString or type(el) == Comment:
            continue

        elements[el.attrs['name']] = {}

        attributes = ['subclass', 'version']
        for attr in attributes:
            if attr in el.attrs:
                attr_value = el.attrs[attr]
            else:
                attr_value = None

            elements[el.attrs['name']][attr] = attr_value

        data = get_fields_from_element(el)
        elements[el.attrs['name']]['data'] = data

    print(elements)
    import pdb
    pdb.set_trace()
    pass


def get_fields_from_element(el):

    fields = el.find_all('field')
    data = []

    for field in fields:

        shape = []
        for parent in field.find_parents('array'):
            shape.append(int(parent.attrs['length']))

        if 'length' in field.attrs:
            shape = [int(field.attrs['length'])]

        if len(shape) == 1:
            shape = shape[0]
        else:
            shape = tuple(shape)

        if not shape:
            shape = 1

        if 'name' in field.attrs:
            name = field.attrs['name']
        else:
            # last parent contains name
            name = parent.attrs['name']

        dtype = field.attrs['type']
        conv_dtype = convert_dtype(dtype)
        data.append((name, conv_dtype, shape))

    return data


def convert_dtype(dtype):

    dtype_lut = {'enumerated': np.str,
                 'string': np.str,
                 'uinteger': np.uint,
                 'uinteger1': np.uint8,
                 'uinteger2': np.uint16,
                 'uinteger4': np.uint32,
                 'integer': np.int,
                 'integer1': np.int8,
                 'integer2': np.int16,
                 'integer4': np.int32,
                 'integer8': np.int64,
                 'time': np.str,
                 'longtime': np.str,
                 'boolean': np.int8}

    if dtype not in dtype_lut:
        raise RuntimeError('dtype unkown: {:}'.format(dtype))

    conv_dtype = dtype_lut[dtype]

    return conv_dtype


def read_szx(filename):
    """
    Read SZO/SZR format.

    Parameters
    ----------
    filename : str
        File name.

    Returns
    -------
    data : numpy.ndarray
        SZO/SZR data.
    """

    eps_file = EPSProduct(filename)
    eps_file.read()
    ptype = eps_file.mphr['product_type'][0]
    fmv = int(eps_file.mphr['format_major_version'][0])

    print(ptype)
    print(fmv)

    raw_data = np.array(eps_file.mdr)[:, 0]

    return raw_data
