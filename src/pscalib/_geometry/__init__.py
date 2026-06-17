"""pscalib._geometry -- vendored numpy-only geometry closure.

Vendored from psana's pure-numpy geometry package
(``lcls2/psana/psana/pscalib/geometry/``) so that deriving the per-pixel
image index maps (``ix``/``iy``) from a detector's geometry text needs **no
psana import**.  Only the minimal closure for jungfrau + epix10ka is included
(7 modules).  See :mod:`pscalib.geometry` for the entry point
(``pixel_coord_indexes_from_text``).

The original psana ``__init__`` only set ``__all__`` (no side effects); this
mirrors that for the vendored subset.
"""

__all__ = [
    'GeometryObject', 'GeometryAccess', 'SegGeometry', 'SegGeometryStore',
    'SegGeometryEpix10kaV1', 'SegGeometryJungfrauV1', 'SegGeometryJungfrauV2',
]
