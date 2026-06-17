#!/usr/bin/env python
# VENDORED into pscalib (US-006) from psana's pure-numpy geometry package:
#   lcls2/psana/psana/pscalib/geometry/SegGeometryStore.py
# Imports rewritten to the vendored relative path; no psana import remains.
# Trimmed to the jungfrau + epix10ka closure (see pscalib/_geometry/__init__.py).
"""
Class :py:class:`SegGeometryStore` is a factory class/method
============================================================

Switches between different device-dependent segments/sensors
to access their pixel geometry using :py:class:`SegGeometry` interface.

Usage::

    from .SegGeometryStore import sgs

    sg = sgs.Create(segname='SENS2X1:V1')
    sg = sgs.Create(segname='EPIX100:V1')
    sg = sgs.Create(segname='EPIX10KA:V1')
    sg = sgs.Create(segname='EPIXHR2X2:V1')
    sg = sgs.Create(segname='EPIXHR1X4:V1')
    sg = sgs.Create(segname='PNCCD:V1')
    sg = sgs.Create(segname='JUNGFRAU:V1')
    sg = sgs.Create(segname='JUNGFRAU:V2')
    sg = sgs.Create(segname='MTRX:512:512:54:54')
    sg = sgs.Create(segname='MTRX:V2:512:512:54:54')
    sg = sgs.Create(segname='MTRX:V2:192:384:50:50') # the same as EPIXMASIC:V1
    sg = sgs.Create(segname='MTRXANY:V1') # the same as MTRX:V2 with posponed initialization
    sg = sgs.Create(segname='EPIXMASIC:V1') # the same as MTRX:V2:192:384:50:50

    sg.print_seg_info(pbits=0o377)
    size_arr = sg.size()
    rows     = sg.rows()
    cols     = sg.cols()
    shape    = sg.shape()
    pix_size = sg.pixel_scale_size()
    area     = sg.pixel_area_array()
    mask     = sg.pixel_mask(mbits=0o377)
    sizeX    = sg.pixel_size_array('X')
    sizeX, sizeY, sizeZ = sg.pixel_size_array()
    X        = sg.pixel_coord_array('X')
    X,Y,Z    = sg.pixel_coord_array()
    xmin = sg.pixel_coord_min('X')
    ymax = sg.pixel_coord_max('Y')
    xmin, ymin, zmin = sg.pixel_coord_min()
    xmax, ymax, zmax = sg.pixel_coord_max()
    ...

See:
 * :py:class:`GeometryObject`,
 * :py:class:`SegGeometry`,
 * :py:class:`SegGeometryCspad2x1V1`,
 * :py:class:`SegGeometryEpix100V1`,
 * :py:class:`SegGeometryEpix10kaV1`,
 * :py:class:`SegGeometryEpixHR2x2V1`
 * :py:class:`SegGeometryEpixHR1x4V1`
 * :py:class:`SegGeometryEpixM320V1`
 * :py:class:`SegGeometryJungfrauV1`,
 * :py:class:`SegGeometryMatrixV1`,
 * :py:class:`SegGeometryMatrixV2`,
 * :py:class:`SegGeometryMatrixAnyV1`,
 * :py:class:`SegGeometryArchonV1`,
 * :py:class:`SegGeometryArchonV2`,
 * :py:class:`SegGeometryStore`

For more detail see `Detector Geometry <https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry>`_.

This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

Created: 2013-03-08 by Mikhail Dubrovin
2020-09-04 - converted to py3
"""

import logging
logger = logging.getLogger(__name__)

def segment_geometry(**kwa):
    """Factory method returns segment geomentry object for specified segname."""
    segname = kwa.get('segname', 'SENS2X1:V1')
    wpc     = kwa.get('use_wide_pix_center', False)
    logger.debug('segment geometry of %s is requested, use_wide_pix_center=%s' % (segname, str(wpc)))

    # NOTE (pscalib vendoring, US-006): this vendored closure supports ONLY the
    # jungfrau + epix10ka LEAF segment geometries -- the two detector families
    # pscalib applies.  The original psana factory dispatched ~20 more segnames
    # (cspad/epix100/epixhr/archon/matrix/...); those branches lazily imported
    # SegGeometry* modules intentionally NOT vendored here, so they are dropped.
    #
    # The fall-through returns None (EXACTLY as the original psana factory did
    # for an unimplemented segname).  This is load-bearing: a geometry file's
    # *container/parent* objects (e.g. 'JFDET:V1', 'IP:V1', 'QUAD:V1') are
    # non-leaf GeometryObjects with NO pixel geometry of their own -- psana
    # builds them with a None algo and only the LEAF objects ('JUNGFRAU:V2',
    # 'EPIX10KA:V1') carry pixel coords.  Raising here would break that.  To
    # extend leaf coverage, vendor the corresponding SegGeometry*.py + branch.
    if segname=='EPIX10KA:V1':
        from .SegGeometryEpix10kaV1 import epix10ka_one, epix10ka_wpc
        return epix10ka_wpc if wpc else epix10ka_one
    elif segname=='JUNGFRAU:V1':
        from .SegGeometryJungfrauV1 import jungfrau_one
        return jungfrau_one
    elif segname=='JUNGFRAU:V2':
        from .SegGeometryJungfrauV2 import jungfrau_front
        return jungfrau_front
    else:
        # container/parent object OR an unvendored leaf segment -> None,
        # matching psana's original behavior.
        logger.debug('segment "%s" geometry is None (container parent or '
                     'unvendored leaf)' % segname)
        return None


class SegGeometryStore():
    def __init__(sp):
        sp.dict_dets = {} # {<det-object>:{segname:<seg_geo-object>}}

    def create_single_segment_geometry(sp, **kwa):
        """returns segment_geometry singleton for detector and segname
           - update_seggeo - enforce update for segment_geometry
        """
        detector = kwa.get('detector', None)
        segname  = kwa.get('segname', None)
        update   = kwa.get('update_seggeo', False)
        logger.debug('segname: %s det: %s' % (segname, str(detector)))
        if segname is None: return None
        dict_segs = sp.dict_dets.get(detector, {})
        seg_geo = dict_segs.get(segname, None)
        if seg_geo is None or update:
            seg_geo = segment_geometry(**kwa)
            dict_segs[segname] = seg_geo
            sp.dict_dets[detector] = dict_segs
        return seg_geo

    def Create(sp, **kwa):
        return sp.create_single_segment_geometry(**kwa)
        #return segment_geometry(**kwa)

sgs = SegGeometryStore()

# EOF - See test_SegGeometryStore.py
