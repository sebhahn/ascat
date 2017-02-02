import os
import lxml.etree as etree
from tempfile import NamedTemporaryFile
from gzip import GzipFile
from collections import OrderedDict
import numpy as np


def get_record(name):
    """
    Read xml record.
    """
    filename = '../../formats/eps_ascatl1bszr_9.0.xml'

    doc = etree.parse(filename)
    elem = doc.xpath('//*[@name="{:}"]'.format(name))

    elem[0].attrib.keys()
    x = elem[0]

    data = OrderedDict()
    length = []

    for child in x.getchildren():

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

        data[name].update({'length': length})
        length = []

    conv = {'longtime': 'i4', 'integer4': 'i4', 'uinteger2': 'i2',
            'integer8': 'i4', 'enumerated': 'enum', 'integer2': 'i2',
            'uinteger4': 'i4', 'uinteger1': 'i1',
            'string': 'str', 'time': 'time', 'uinteger': 'i4',
            'integer': 'i2', 'boolean': 'bool'}

    dtype = []
    for key, values in data.items():
        dtype.append((key, conv[values['type']], values['length']))

    return dtype


def grh_record():
    """
    Generic record header.
    """
    short_cds_time = np.dtype([('day', np.uint16),
                               ('time', np.uint32)])

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
                         'ASCA_SZO_1B_9': 'eps_ascatl1bszo_9.0.xml'}

    key = '{:}_{:}_{:}_{:}'.format(mphr_dict['INSTRUMENT_ID'],
                                   mphr_dict['PRODUCT_TYPE'],
                                   mphr_dict['PROCESSING_LEVEL'],
                                   mphr_dict['PROCESSOR_MAJOR_VERSION'])

    print('Product: {:}'.format(key))

    filename = product_xml_table[key]
    xml_file = os.path.join('..', '..', 'formats', filename)

    return xml_file


def read_sphr(fid, grh):
    """
    Read Special Product Header (SPHR).
    """
    sphr = fid.read(grh['record_size'] - grh.itemsize)
    sphr_dict = OrderedDict(item.replace(' ', '').split('=')
                            for item in sphr.split('\n')[:-1])
    print('SPHR')


def read_eps(filename):
    """
    Read EPS file.
    """
    # record_class_dict = {1: 'MPHR', 2: 'SPHR', 3: 'IPR', 4: 'GEADR',
    #                      5: 'GIADR', 6: 'VEADR', 7: 'VIADR', 8: 'MDR'}

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
        grh = read_record(fid, grh_record())
        record_size = grh['record_size'][0]
        record_class = grh['record_class'][0]

        # main product header
        if record_class == 1:
            xml_file = read_mphr(fid, grh)
            print(xml_file)

        # specific product header
        elif record_class == 2:
            read_sphr(fid, grh)

        # ipr
        # elif record_class == 3:
        #     pass

        # geadr
        # elif record_class == 4:
        #     pass

        # giadr
        # elif record_class == 5:
        #     pass

        # veadr
        # elif record_class == 6:
        #     pass

        # viadr
        # elif record_class == 7:
        #     pass

        else:
            # return pointer to the beginning of the record
            fid.seek(bor)
            fid.seek(record_size, 1)

        # Determine number of bytes read
        eor = fid.tell()

    fid.close()

    if zipped:
        os.remove(filename)


def test_xml():
    """
    Test xml file.
    """
    record_name = 'mphr'
    record_name = 'sphr'
    record_name = 'geadr-lsm'
    record_name = 'viadr-oa'
    record_name = 'viadr-ver'
    record_name = 'veadr - prc'
    record_name = 'mdr-1b-125'
    print(get_record(record_name))


def test_eps():
    """
    Test read EPS file.
    """
    root_path = os.path.join('/home', 'shahn', 'shahn', 'datapool', 'ascat',
                             'metop_a_szr', '2016')
    filename = os.path.join(
        root_path, 'ASCA_SZR_1B_M02_20160101005700Z_20160101023858Z_N_O_20160101023812Z.nat.gz')
    read_eps(filename)

if __name__ == '__main__':
    test_eps()
