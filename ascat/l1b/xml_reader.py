import os
import lxml.etree as etree
from tempfile import NamedTemporaryFile
from gzip import GzipFile
from collections import OrderedDict

import numpy as np

short_cds_time = np.dtype([('day', np.uint16), ('time', np.uint32)])

long_cds_time = np.dtype([('day', np.uint16), ('ms', np.uint32),
                          ('mms', np.uint16)])


def read_xml_mdr(filename):
    """
    Read xml record.
    """
    doc = etree.parse(filename)
    elements = doc.xpath('//mdr')
    data = OrderedDict()
    length = []
    elem = elements[0]

    for child in elem.getchildren():

        if child.tag == 'delimiter':
            continue

        child_items = dict(child.items())
        name = child_items.pop('name')

        try:
            var_len = child_items.pop('length')
            length.append(np.int(var_len))
        except KeyError:
            pass

        data[name] = child_items

        if child.tag == 'array':
            for arr in child.iterdescendants():
                arr_items = dict(arr.items())
                if arr.tag == 'field':
                    data[name].update(arr_items)
                else:
                    try:
                        var_len = arr_items.pop('length')
                        length.append(np.int(var_len))
                    except KeyError:
                        pass

        if length:
            data[name].update({'length': length})
        else:
            data[name].update({'length': 1})

        length = []

    conv = {'longtime': long_cds_time, 'time': short_cds_time,
            'boolean': np.uint8, 'integer1': np.int8, 'uinteger1': np.uint8,
            'integer': np.int32, 'uinteger': np.uint32, 'integer2': np.int16,
            'uinteger2': np.uint16, 'integer4': np.int32,
            'uinteger4': np.uint32, 'integer8': np.int64,
            'enumerated': np.uint8, 'string': 'str'}

    scaling_factor = []
    scaled_dtype = []
    dtype = []

    for key, value in data.items():

        if 'scaling-factor' in value:
            sf_dtype = np.float32
            sf = eval(value['scaling-factor'].replace('^', '**'))
        else:
            sf_dtype = conv[value['type']]
            sf = 1

        scaling_factor.append(sf)
        scaled_dtype.append((key, sf_dtype, value['length']))
        dtype.append((key, conv[value['type']], value['length']))

    return np.dtype(dtype), np.dtype(scaled_dtype), np.array(scaling_factor,
                                                             dtype=np.float32)


def grh_record():
    """
    Generic record header.
    """
    record_dtype = np.dtype([('record_class', np.ubyte),
                             ('instrument_group', np.ubyte),
                             ('record_subclass', np.ubyte),
                             ('record_subclass_version', np.ubyte),
                             ('record_size', np.uint32),
                             ('record_start_time', short_cds_time),
                             ('record_stop_time', short_cds_time)])

    return record_dtype


def read_record(fid, dtype, count=1):
    """
    Read record
    """
    record = np.fromfile(fid, dtype=dtype, count=count)
    return record.newbyteorder('B')


def read_mphr(fid, grh):
    """
    Read Main Product Header (MPHR).
    """
    mphr = fid.read(grh['record_size'] - grh.itemsize)
    mphr_dict = OrderedDict(item.replace(' ', '').split('=')
                            for item in mphr.split('\n')[:-1])

    product_xml_table = {'ASCA_SZF_1B_9': 'eps_ascatl1bszf_9.0.xml',
                         'ASCA_SZR_1B_9': 'eps_ascatl1bszr_9.0.xml',
                         'ASCA_SZO_1B_9': 'eps_ascatl1bszo_9.0.xml',
                         'ASCA_SMR_02_4': 'eps_ascatl2smr_4.xml',
                         'ASCA_SMR_02_5': 'eps_ascatl2smr_4.xml'}

    key = '{:}_{:}_{:}_{:}'.format(mphr_dict['INSTRUMENT_ID'],
                                   mphr_dict['PRODUCT_TYPE'],
                                   mphr_dict['PROCESSING_LEVEL'],
                                   mphr_dict['PROCESSOR_MAJOR_VERSION'])

    # print('Product: {:}'.format(key))
    filename = product_xml_table[key]
    xml_file = os.path.join('..', '..', 'formats', filename)

    return xml_file, mphr_dict


def read_sphr(fid, grh):
    """
    Read Special Product Header (SPHR).
    """
    sphr = fid.read(grh['record_size'] - grh.itemsize)
    sphr_dict = OrderedDict(item.replace(' ', '').split('=')
                            for item in sphr.split('\n')[:-1])

    return sphr_dict


def read_eps(filename):
    """
    Read EPS file.
    """
    # record_class_dict = {1: 'MPHR', 2: 'SPHR', 3: 'IPR', 4: 'GEADR',
    #                      5: 'GIADR', 6: 'VEADR', 7: 'VIADR', 8: 'MDR'}

    mdr_template = None
    mdr_list = []

    zipped = False
    if os.path.splitext(filename)[1] == '.gz':
        zipped = True

    if zipped:
        with NamedTemporaryFile(delete=False) as tmp_fid:
            with GzipFile(filename) as gz_fid:
                tmp_fid.write(gz_fid.read())
            filename = tmp_fid.name

    fid = open(filename, 'rb')
    filesize = os.path.getsize(filename)
    eor = fid.tell()

    while eor < filesize:

        # remember beginning of the record
        bor = fid.tell()

        # read grh of current record
        grh = read_record(fid, grh_record())[0]
        record_size = grh['record_size']
        record_class = grh['record_class']

        if record_class == 1:
            xml_file, mphr = read_mphr(fid, grh)
            mdr_template, scaled_template, sfactor = read_xml_mdr(xml_file)
            continue

        if record_class == 2:
            read_sphr(fid, grh)
            continue

        # mdr
        if record_class == 8:
            import pdb
            pdb.set_trace()
            mdr = read_record(fid, mdr_template)
            mdr_list.append(mdr)

        # return pointer to the beginning of the record
        fid.seek(bor)
        fid.seek(record_size, 1)

        # determine number of bytes read
        # end of record
        eor = fid.tell()

    fid.close()

    if zipped:
        os.remove(filename)

    data = np.hstack(mdr_list)
    sc_data = np.zeros_like(data, dtype=scaled_template)

    for name, sf in zip(data.dtype.names, sfactor):
        if sf != 1:
            sc_data[name] = data[name] / sf
        else:
            sc_data[name] = data[name]

    return sc_data


def test_eps():
    """
    Test read EPS file.
    """
    root_path = os.path.join('/home', 'shahn', 'shahn', 'datapool', 'ascat',
                             'metop_a_szr', '2016')
    filename = os.path.join(
        root_path, 'ASCA_SZR_1B_M02_20160101005700Z_20160101023858Z_N_O_20160101023812Z.nat.gz')

    root_path = os.path.join('/home', 'shahn', 'shahn', 'datapool', 'ascat',
                             'ascat_test_data_for_readers', 'umarf',
                             'onlinedownload', 'shahn', '1196858')

    filename = os.path.join(
        root_path, 'ASCA_SZF_1B_M01_20161101014200Z_20161101032359Z_N_O_20161101023050Z.nat.gz')

    root_path = os.path.join('/home', 'shahn', 'shahn', 'datapool', 'ascat',
                             'ascat_test_data_for_readers', 'umarf',
                             'onlinedownload', 'shahn', '1196868')

    filename = os.path.join(
        root_path, 'ASCA_SMR_02_M01_20161101032400Z_20161101050558Z_N_O_20161101041358Z.nat.gz')

    data = read_eps(filename)

    import pdb
    pdb.set_trace()
    pass


if __name__ == '__main__':
    test_eps()
