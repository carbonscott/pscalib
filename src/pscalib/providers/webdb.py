#!/usr/bin/env python3
"""pscalib.providers.webdb -- vendored ``requests``-only web-DB retrieval.

The web retrieval provider: a pure-Python HTTP client over the LCLS-II
calibration web service (the same service psana itself reads from).  It fetches
the *exact* ``{ctype: (data, doc)}`` dict psana stores as ``det.raw._calibconst``
-- byte-for-byte identical -- **without importing the psana framework**.

Why vendored, not imported
--------------------------
psana's read path is itself a thin ``requests`` HTTP client (it never opens a
Mongo socket for reads, touches no compiled code, and needs no credentials for
GET).  But importing it via ``psana.pscalib.calib.MDBWebUtils`` would run
``psana/__init__.py`` and drag in ``mpi4py`` + ``dgram`` on *any* ``psana.*``
import.  So the traced pure-Python read closure of ``calib_constants_all_types``
is vendored here instead.  This module imports only ``requests`` + ``numpy`` +
``bson`` (the pure-Python ``ObjectId``; no Mongo connection); it never imports
psana / mpi4py / dgram / pymongo.

Provenance
----------
Faithfully lifted from the psana source tree
``…/software/lcls2/psana/psana/pscalib/calib/``.  Function bodies are kept
behaviorally identical to psana (so the bytes match); only the write/delete
half, the Kerberos requirement, and the unused ``pyalgos`` helpers are dropped.
Each vendored function carries a ``-- psana <file>:<line>`` provenance note.

  * ``CalibConstants.py``    -> the base-URL config (:42-52); Kerberos is
    write-only (:32 ``from krtc import KerberosTicket`` is made optional here).
  * ``MDBUtils.py``          -> ``db_prefixed_name`` (:102),
    ``dbnames_collection_query`` (:382), ``sec_and_ts_from_id`` (:165, uses
    ``bson.objectid.ObjectId``), ``object_from_data_string`` (:338),
    ``_short_for_partial_name`` (:515), ``_pro_detector_name`` (:544).
  * ``CalibDoc.py``          -> ``CalibDoc`` (run-range validity sort).
  * ``Time.py`` / ``TimeFormat.py`` -> ``Time.parse`` (stdlib-only id timestamp).
  * ``MDBWebUtils.py``       -> ``request``/``get`` (:104/:62), ``find_docs``
    (:138), ``select_doc_in_run_range`` (:196), ``select_latest_doc`` (:167),
    ``get_data_for_doc`` (:235), ``pro_detector_name``/``_short_detector_name``
    (:642/:593), ``dbnames_collection_query`` wrapper (:251),
    ``calib_constants`` (:265), ``calib_constants_of_missing_types`` (:287),
    ``calib_constants_all_types`` (:327).

The read path that the psana framework reaches NEVER calls the
``pyalgos.generic.Utils`` (``gu.*``) helpers nor ``NDArrUtils`` -- those are all
write-side / file-side -- so they are deliberately not vendored.

Public API
----------
``get_constants(uniqueid, exp, run, ...)`` -- the standalone equivalent of
``det.raw._calibconst``: pass the detector unique id (``det.raw._uniqueid``),
experiment, and run; get back ``{ctype: (data, doc)}`` byte-exact vs psana.
``calib_constants_all_types`` is exposed under its psana name too.
"""

import os
import re
import sys
import time as _time_module
from subprocess import call
from time import gmtime, localtime, strftime

import numpy as np

# requests + bson are the 'web' extra; imported at module top because this
# module IS the web path.  Importing pscalib (or pscalib.providers) does NOT
# import this module, so the numpy-only import surface is preserved.
import requests as req
from bson.objectid import ObjectId

__all__ = [
    "get_constants",
    "calib_constants_all_types",
    "calib_constants",
    "URL",
    "URL_ENV",
    "IS_OFFSITE",
    "MAX_DETNAME_SIZE",
    "DETNAMESDB",
    "DBNAME_PREFIX",
]

import logging
logger = logging.getLogger(__name__)


# ==========================================================================
# Base-URL config -- psana CalibConstants.py:42-52
#
# On-site default: https://psdmint.sdf.slac.stanford.edu/calib_ws/
# Off-site:        https://psextapi.slac.stanford.edu/calib_ws/  (SIT_PSDM_OFFSITE)
# Explicit override: LCLS_CALIB_HTTP
#
# Kerberos (krtc) is WRITE-only in psana; reads are anonymous HTTPS GET, so the
# top-level ``from krtc import KerberosTicket`` is dropped -- this module needs
# no krtc.  These are read at import time, mirroring psana, but every public
# entry point also accepts an explicit ``url=`` so the env need not be set.
# ==========================================================================
IS_OFFSITE = os.environ.get("SIT_PSDM_OFFSITE", None) is not None
URL_ENV = os.environ.get("LCLS_CALIB_HTTP", None)

URL = ("https://psextapi.slac.stanford.edu/calib_ws/" if IS_OFFSITE else
       "https://psdmint.sdf.slac.stanford.edu/calib_ws/" if URL_ENV is None else
       URL_ENV)

DBNAME_PREFIX = "cdb_"
DETNAMESDB = "%sdetnames" % DBNAME_PREFIX
MAX_DETNAME_SIZE = 20  # psana CalibConstants.py:63

TSFORMAT = "%Y-%m-%dT%H:%M:%S%z"  # psana CalibConstants.py:74


# ==========================================================================
# Time parsing -- psana Time.py / TimeFormat.py (stdlib only)
#
# Needed by sec_and_ts_from_id to derive a Mongo _id's creation time, which is
# how docs are time-sorted in select_latest_doc.  Faithful copy of the psana
# parser (pure stdlib: time + re); only the parts the read path reaches.
# ==========================================================================
_ffmtre = re.compile(r"%([.](\d+))?f")
_DATE_RE = r"(\d{4})(?:-?(\d{2})(?:-?(\d{2}))?)?"
_TIME_RE = r"(\d{1,2})(?::?(\d{2})(?::?(\d{2})(?:[.](\d{1,9}))?)?)?"
_TZ_RE = r"Z|(?:([-+])(\d{2})(?::?(\d{2}))?)"
_dtre = re.compile("^" + _DATE_RE + "(?:(?: +|T)(?:" + _TIME_RE + ")?(" +
                   _TZ_RE + ")?)?$")
_secre = re.compile(r"^S(\d{0,10})(?:[.](\d{1,9}))?$")


def _getNsec(nsecStr):
    ndig = min(len(nsecStr), 9)
    nsecStr = nsecStr[:ndig] + "0" * (9 - ndig)
    return int(nsecStr)


def _cmp_tm(lhs, rhs):
    for i in range(6):
        if lhs[i] != rhs[i]:
            return False
    if lhs[8] >= 0 and rhs[8] >= 0:
        if lhs[8] != rhs[8]:
            return False
    return True


def _mktime_from_utc(t):
    """psana TimeFormat.py:148 -- struct_tm (UTC) -> time_t."""
    try:
        tl = _time_module.mktime(t)
    except Exception:
        t = (t[0], t[1], t[2], t[3] - 1, t[4], t[5], t[6], t[7], t[8])
        tl = _time_module.mktime(t)
        tl += 3600
    tg = _time_module.gmtime(tl)
    tg = (tg[0], tg[1], tg[2], tg[3], tg[4], tg[5], tg[6], tg[7], 0)
    try:
        tb = _time_module.mktime(tg)
    except Exception:
        tg = (tg[0], tg[1], tg[2], tg[3] - 1, tg[4], tg[5], tg[6], tg[7], 0)
        tb = _time_module.mktime(tg)
        tb += 3600
    return tl - (tb - tl)


def _parseTime(timeStr):
    """psana TimeFormat.py:84 -- parse a date/time string -> (sec, nsec)."""
    match = _secre.match(timeStr)
    if match:
        sec = int(match.group(1))
        nsec = 0
        if match.group(2):
            nsec = _getNsec(match.group(2))
        return (sec, nsec)

    match = _dtre.match(timeStr)
    if match:
        year = int(match.group(1))
        month = int(match.group(2) or 1)
        if month < 1 or month > 12:
            raise ValueError("parseTime: month value out of range: " + timeStr)
        day = int(match.group(3) or 1)
        if day < 1 or day > 31:
            raise ValueError("parseTime: day value out of range: " + timeStr)
        hour = int(match.group(4) or 0)
        if hour > 23:
            raise ValueError("parseTime: hour value out of range: " + timeStr)
        minute = int(match.group(5) or 0)
        if minute > 59:
            raise ValueError("parseTime: minute value out of range: " + timeStr)
        sec = int(match.group(6) or 0)
        if sec > 60:
            raise ValueError("parseTime: second value out of range: " + timeStr)
        nsec = 0
        if match.group(7):
            nsec = _getNsec(match.group(7))

        if match.group(8):
            tzoffset_min = 0
            if match.group(8) != "Z":
                tz_hour = int(match.group(10))
                tz_min = int(match.group(11) or 0)
                if tz_hour > 12 or tz_min > 59:
                    raise ValueError(
                        "parseTime: timezone out of range: " + timeStr)
                tzoffset_min = tz_hour * 60 + tz_min
                if match.group(9) == "-":
                    tzoffset_min = -tzoffset_min
            isdst = 0
            t = (year, month, day, hour, minute, sec, -1, -1, isdst)
            sec = _mktime_from_utc(t)
            tval = _time_module.gmtime(sec)
            if not _cmp_tm(t, tval):
                raise ValueError(
                    "parseTime: input time validation failed: " + timeStr)
            sec -= tzoffset_min * 60
        else:
            isdst = -1
            t = (year, month, day, hour, minute, sec, -1, -1, isdst)
            sec = _time_module.mktime(t)
            tval = _time_module.localtime(sec)
            if not _cmp_tm(t, tval):
                raise ValueError(
                    "parseTime: input time validation failed: " + timeStr)
        return (sec, nsec)

    raise ValueError("parseTime: failed to parse string: " + timeStr)


# ==========================================================================
# MDBUtils -- the DB-agnostic helpers the read path reaches
# ==========================================================================
def db_prefixed_name(name, prefix=DBNAME_PREFIX):
    """psana MDBUtils.py:102 -- 'exp12345' -> 'cdb_exp12345'."""
    if name is None:
        return None
    assert isinstance(name, str), "db_prefixed_name parameter should be str"
    assert len(name) < 128, "name length should be <128 characters"
    return "%s%s" % (prefix, name)


def sec_and_ts_from_id(id, fmt="%Y%m%d_%H%M%S", gmt=False):
    """psana MDBUtils.py:165 -- Mongo (str) _id -> (int sec, str ts).

    Uses ``bson.objectid.ObjectId`` (pure-Python; no Mongo connection) to get
    the document's generation time -- the sort key for selecting the latest doc.
    """
    assert isinstance(id, str)
    assert len(id) == 24
    oid = ObjectId(id)
    str_ts = str(oid.generation_time)  # '2018-03-14 21:59:37+00:00'
    sec, nsec = _parseTime(str_ts)
    tsec = int(sec)
    if fmt is not None:
        str_ts = strftime(fmt, gmtime(tsec) if gmt else localtime(tsec))
    return tsec, str_ts


def _dict_from_data_string(s):
    """psana MDBUtils.py:324 -- xtcav 'str' ctypes that hold a serialized dict.

    Only reached for the xtcav ctypes ('xtcav_lasingoff', 'xtcav_pedestals',
    'lasingoffreference').  psana then calls ``MDBConvertUtils.deserialize_dict``
    to in-place-convert nested base64-encoded ndarrays back to arrays.  That
    converter is itself pure (numpy + base64), but no area-detector ctype on the
    pscalib read path is an xtcav dict, so it is vendored lazily and only if
    actually hit -- keeping the common path dependency-free.
    """
    import ast
    try:
        d = ast.literal_eval(s)
    except Exception as err:
        logger.error('literal_eval failed: %s' % err)
        return None
    if not isinstance(d, dict):
        return None
    _deserialize_dict(d)
    return d


def _deserialize_dict(d):
    """psana MDBConvertUtils.deserialize_dict -- in-place base64 -> ndarray.

    Vendored faithfully; only exercised for xtcav dict ctypes (not on the
    area-detector calib path).  Pure-python (numpy + base64).
    """
    import base64
    for k, v in d.items():
        if isinstance(v, dict):
            if "__ndarray__" in v:
                data = base64.b64decode(v["__ndarray__"])
                d[k] = np.frombuffer(data, v["dtype"]).reshape(v["shape"])
            else:
                _deserialize_dict(v)
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, dict):
                    _deserialize_dict(item)


def object_from_data_string(s, doc):
    """psana MDBUtils.py:338 -- raw GridFS bytes -> str / ndarray / dict.

    The byte-exact deserialization: ``np.frombuffer`` for ndarrays (with the
    doc's recorded dtype/shape), ``.decode()`` for str, ``pickle.loads`` for
    'any'.  This is where the bytes become the exact arrays psana returns.
    """
    data_type = doc.get("data_type", None)
    if data_type is None:
        logger.warning("object_from_data_string: data_type is None: %s" % doc)
        return None
    if data_type == "str":
        data = s.decode()
        if doc.get("ctype", None) in (
                "xtcav_lasingoff", "xtcav_pedestals", "lasingoffreference"):
            return _dict_from_data_string(data)
        return data
    elif data_type == "ndarray":
        str_dtype = doc.get("data_dtype", None)
        nda = np.frombuffer(s, dtype=str_dtype)
        nda.shape = eval(doc.get("data_shape", None))  # str shape -> tuple
        return nda
    elif data_type == "any":
        import pickle
        return pickle.loads(s)
    else:
        logger.warning("object_from_data_string: UNEXPECTED data_type: %s" %
                       data_type)
        return None


def dbnames_collection_query(det, exp=None, ctype="pedestals", run=None,
                             time_sec=None, vers=None, dtype=None):
    """psana MDBUtils.py:382 -- build (db_det, db_exp, colname, query).

    ``query`` is the Mongo-style selector, e.g.
    ``{'detector': <short>, 'run': {'$lte': runnum}}``.  Note ``det`` here must
    already be the SHORT detector name (the wrapper resolves it via HTTP first).
    """
    cond = (run is not None) or (time_sec is not None) or (vers is not None)
    assert cond, "Not sufficient info for query: run, time_sec, vers all None"
    _det = det
    query = {"detector": _det}
    if ctype is not None:
        query["ctype"] = ctype
    if dtype is not None:
        query["dtype"] = dtype
    runq = run if not (run in (0, None)) else 9999  # cpo 2020-01-16
    query["run"] = {"$lte": runq}
    if time_sec is not None:
        query["time_sec"] = {"$lte": int(time_sec)}
    if vers is not None:
        query["version"] = vers
    db_det, db_exp = db_prefixed_name(_det), db_prefixed_name(str(exp))
    if None in (db_det, db_exp):
        return None, None, None, None
    if "None" in db_det:
        db_det = None
    if "None" in db_exp:
        db_exp = None
    return db_det, db_exp, _det, query


def _short_for_partial_name(detname, ldocs):
    """psana MDBUtils.py:515 -- resolve a (possibly partial) long detname to its
    short name from the list of cdb_detnames docs."""
    name_fields = detname.split("_")
    if len(name_fields) < 2:
        logger.warning("Partial detname %s lacks fields to find long name." %
                       detname)
        return None
    pnames = name_fields[1:]
    for doc in ldocs:
        longname = doc.get("long", None)
        if longname is None:
            continue
        if all([name in longname for name in pnames]):
            return doc.get("short", None)
    return None


# ==========================================================================
# CalibDoc -- psana CalibDoc.py (run-range validity sort)
# ==========================================================================
class CalibDoc:
    """psana CalibDoc.py -- wraps a metadata doc; validates + sorts by run
    range.  Drives ``select_doc_in_run_range`` (which doc is valid for a run)."""
    rnum_max = 9999

    def __init__(self, doc):
        self.doc = doc
        begin = doc["run"]
        end = doc["run_end"]
        self.tsec_id, self.tstamp_id = sec_and_ts_from_id(doc["_id"])
        self.valid = False

        self.begin = int(begin)
        if self.begin > self.rnum_max:
            self._set_invalid("INVALID run '%s' - begin too big" % str(begin))
            return

        if str(end).isdigit():
            self.end = int(end)
            if self.end > self.rnum_max:
                self._set_invalid("INVALID run '%d' - end too big" % self.end)
                return
        elif end == "end":
            self.end = self.rnum_max
        else:
            self._set_invalid("INVALID run end value '%s'" % str(end))
            return

        self.valid = True

    def _set_invalid(self, msg):
        logger.warning(msg)
        self.valid = False

    def _cmp_tsec_id(self, other):
        if self.tsec_id < other.tsec_id:
            return -1
        elif self.tsec_id > other.tsec_id:
            return 1
        return 0

    def _cmp(self, other):
        if self.begin < other.begin:
            return -1
        elif self.begin > other.begin:
            return 1
        else:
            if self.end > other.end:
                return -1
            elif self.end < other.end:
                return 1
            return self._cmp_tsec_id(other)

    def __eq__(self, other):
        return self._cmp(other) == 0

    def __ne__(self, other):
        return self._cmp(other) != 0

    def __lt__(self, other):
        return self._cmp(other) < 0

    def __le__(self, other):
        return self._cmp(other) <= 0

    def __gt__(self, other):
        return self._cmp(other) > 0

    def __ge__(self, other):
        return self._cmp(other) >= 0


# ==========================================================================
# HTTP primitives -- psana MDBWebUtils.py:62-118
# ==========================================================================
get = req.get


def request(url, query=None):
    """psana MDBWebUtils.py:104 -- the one GET wrapper all reads go through.

    Reads are anonymous HTTPS GET; no Kerberos, no credentials.  Mirrors
    psana's behavior including the 180s timeout and 503 handling.
    """
    r = get(url, query, timeout=180)
    if r.ok:
        return r
    s = ("get url: %s query: %s\n  response status: %s status_code: %s "
         "reason: %s" % (url, str(query), r.ok, r.status_code, r.reason))
    s += '\nTry command: curl -s "%s"' % url
    logger.debug(s)
    if r.status_code == 503:
        logger.warning(s)
        sys.exit(1)
    return None


def find_docs(dbname, colname, query={}, url=None):
    """psana MDBWebUtils.py:138 -- GET the list of metadata docs for a query."""
    if url is None:
        url = URL
    uri = "%s/%s/%s" % (url.rstrip("/"), dbname, colname)
    query_string = str(query).replace("'", '"')
    r = request(uri, {"query_string": query_string})
    if r is None:
        return None
    try:
        return r.json()
    except Exception:
        logger.debug("find_docs: json conversion failed for query: %s" % query)
        return None


def select_latest_doc(docs, query):
    """psana MDBWebUtils.py:167 -- pick the doc with the newest _id timestamp."""
    if docs is None:
        return None
    if len(docs) == 0:
        return None
    for d in docs:
        d["tsec_id"], d["tstamp_id"] = sec_and_ts_from_id(d["_id"])
    key_sort = "tsec_id"
    vals = [int(d[key_sort]) for d in docs]
    vals.sort(reverse=True)
    val_sel = int(vals[0])
    for d in docs:
        if d[key_sort] == val_sel:
            return d
    return None


def select_doc_in_run_range(docs, rnum):
    """psana MDBWebUtils.py:196 -- pick the doc whose [run, run_end] covers
    rnum (the latest-deployed among valid candidates).  THE selection rule for
    ``calib_constants_all_types`` -- governs which constant version you get."""
    cdocs = [CalibDoc(d) for d in docs]
    cdocs_sorted = sorted([cd for cd in cdocs if cd.valid])
    for d in cdocs_sorted[::-1]:
        if d.valid and d.begin <= rnum and rnum <= d.end:
            return d.doc
    return None  # no matching found


def get_data_for_doc(dbname, doc, url=None):
    """psana MDBWebUtils.py:235 -- GET the GridFS blob for a doc and decode it
    into the exact ndarray / str / dict the doc describes."""
    if url is None:
        url = URL
    idd = doc.get("id_data", None)
    if idd is None:
        logger.debug("get_data_for_doc: 'id_data' missing in doc")
        return None
    r2 = request("%s/%s/gridfs/%s" % (url.rstrip("/"), dbname, idd))
    if r2 is None:
        return None
    return object_from_data_string(r2.content, doc)


# ==========================================================================
# Detector long-name -> short-name (an HTTP lookup, not a local hash)
# psana MDBWebUtils.py:593-646
# ==========================================================================
def _short_detector_name(detname, dbname=DETNAMESDB, add_shortname=False,
                         url=None):
    """psana MDBWebUtils.py:593 -- resolve a long detname to its short name via
    the ``cdb_detnames`` DB (a server-side mapping, replicated over HTTP).

    ``add_shortname`` (which would *register* a new short name) is a write op and
    is unsupported in this read-only client; passing it True raises."""
    if add_shortname:
        raise NotImplementedError(
            "add_shortname=True registers a new short name (a write op); the "
            "vendored read-only client does not support writes")
    colname = detname.split("_", 1)[0]
    query = {"long": detname}
    ldocs = find_docs(dbname, colname, query=query, url=url)

    if ldocs is None:
        logger.warning("db/collection %s/%s NO DOC for long detname %s" %
                       (dbname, colname, detname))
        return None
    if len(ldocs) > 1:
        logger.warning("db/collection %s/%s has >1 doc for detname %s" %
                       (dbname, colname, detname))
    if len(ldocs) == 1:
        shortname = ldocs[0].get("short", None)
        if shortname is not None:
            return shortname

    # fall back: scan all docs in the collection for a partial-name match
    ldocs = find_docs(dbname, colname, query={}, url=url)
    shortname = _short_for_partial_name(detname, ldocs)
    if shortname is not None:
        return shortname
    return None


def pro_detector_name(detname, maxsize=MAX_DETNAME_SIZE, add_shortname=False,
                      url=None):
    """psana MDBWebUtils.py:642 -- the detname used as the Mongo collection key.

    Short names are used verbatim; long names (>= maxsize) are resolved to their
    short form via the HTTP ``cdb_detnames`` lookup.  This is the only step that
    cannot be computed locally -- the long->short map lives server-side."""
    if detname is None:
        return None
    assert isinstance(detname, str), "unexpected detname: %s" % str(detname)
    if len(detname) < maxsize:
        return detname
    return _short_detector_name(detname, add_shortname=add_shortname, url=url)


def _dbnames_collection_query_web(det, exp=None, ctype="pedestals", run=None,
                                  time_sec=None, vers=None, dtype=None,
                                  url=None, max_detname_size=MAX_DETNAME_SIZE):
    """psana MDBWebUtils.py:251 -- resolve det -> short name (HTTP) then build
    the (db_det, db_exp, colname, query) tuple."""
    short = pro_detector_name(det, maxsize=max_detname_size, url=url)
    return list(dbnames_collection_query(short, exp, ctype, run, time_sec, vers,
                                         dtype))


# ==========================================================================
# The fetch entry points -- psana MDBWebUtils.py:265-370
# ==========================================================================
def calib_constants(det, exp=None, ctype="pedestals", run=None, time_sec=None,
                    vers=None, url=None):
    """psana MDBWebUtils.py:265 -- single-ctype fetch -> (data, doc) or None."""
    if url is None:
        url = URL
    db_det, db_exp, colname, query = _dbnames_collection_query_web(
        det, exp, ctype, run, time_sec, vers, dtype=None, url=url)
    dbname = db_det if (exp is None) else db_exp
    docs = find_docs(dbname, colname, query, url)
    if docs is None:
        return None
    doc = select_latest_doc(docs, query)
    if doc is None:
        logger.debug("document not available for query: %s" % str(query))
        return None
    return (get_data_for_doc(dbname, doc, url), doc)


def calib_constants_of_missing_types(resp, det, time_sec=None, vers=None,
                                     url=None):
    """psana MDBWebUtils.py:287 -- fill ctypes absent from the experiment DB
    using the per-detector DB (the second pass psana always runs)."""
    if url is None:
        url = URL
    exp = None
    run = 9999
    ctype = None
    db_det, db_exp, colname, query = _dbnames_collection_query_web(
        det, exp, ctype, run, time_sec, vers, dtype=None, url=url)
    dbname = db_det
    docs = find_docs(dbname, colname, query, url)
    if docs is None:
        return resp

    ctypes = set([d.get("ctype", None) for d in docs])
    ctypes.discard(None)

    ctypes_resp = resp.keys()
    _ctypes = [ct for ct in ctypes if not (ct in ctypes_resp)]

    for ct in _ctypes:
        docs_for_type = [d for d in docs if d.get("ctype", None) == ct]
        doc = select_latest_doc(docs_for_type, query)
        if doc is None:
            continue
        resp[ct] = (get_data_for_doc(dbname, doc, url), doc)

    return resp


def calib_constants_all_types(det, exp=None, run=None, time_sec=None,
                              vers=None, url=None):
    """psana MDBWebUtils.py:327 -- THE function psana itself calls to populate
    ``det.raw._calibconst``.  Returns ``{ctype: (data, doc)}`` for every ctype.

    Byte-exact reproduction of psana's read path: same query, same run-range
    doc selection, same GridFS decode, same experiment-then-detector-DB
    two-pass.  ``det`` is the detector long unique id (``det.raw._uniqueid``)."""
    if url is None:
        url = URL
    ctype = None

    db_det, db_exp, colname, query = _dbnames_collection_query_web(
        det, exp, ctype, run, time_sec, vers, dtype=None, url=url)
    dbname = db_det if (exp is None) else db_exp

    docs = find_docs(dbname, colname, query, url)

    resp = {}
    if docs is not None:
        ctypes = set([d.get("ctype", None) for d in docs])
        ctypes.discard(None)

        for ct in ctypes:
            docs_for_type = [d for d in docs if d.get("ctype", None) == ct]
            doc = select_doc_in_run_range(docs_for_type, run)
            if doc is None:
                continue
            resp[ct] = (get_data_for_doc(dbname, doc, url), doc)

    resp = calib_constants_of_missing_types(resp, det, time_sec, vers, url)

    return resp


# ==========================================================================
# Public pscalib entry point
# ==========================================================================
def get_constants(uniqueid, exp, run, url=None):
    """Fetch a detector's full calibration-constant dict over HTTP -- the
    standalone, psana-free equivalent of ``det.raw._calibconst``.

    Parameters
    ----------
    uniqueid : str
        The detector's long unique id -- ``det.raw._uniqueid`` (e.g.
        ``jungfrau_<serial>...``).  This is what psana passes as ``det`` to
        ``calib_constants_all_types``.
    exp : str
        Experiment id (e.g. ``"mfx100848724"``).
    run : int
        Run number.  Per-ctype validity ranges are honored exactly as psana
        does (``select_doc_in_run_range``).
    url : str, optional
        Override the base URL.  Defaults to the on-site/off-site/``LCLS_CALIB_HTTP``
        resolution in :data:`URL`.

    Returns
    -------
    dict
        ``{ctype: (data, doc)}`` -- ``data`` is the exact ndarray / str the
        constant holds (byte-identical to ``det.raw._calibconst[ctype][0]``),
        ``doc`` the metadata document (carries the validity range).
    """
    return calib_constants_all_types(uniqueid, exp=exp, run=int(run), url=url)
