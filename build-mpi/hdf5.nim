## Minimal HDF5 wrapper for writing seq[float64] datasets.
##
## Supports:
##   var f = H5File("out.h5", "rw")   # create/overwrite
##   f["/group/subgroup/dataset"] = mySeq
##   f.close()
##
## Only dependency: libhdf5 (-lhdf5).

import std/[strutils]

# Locate HDF5 at compile time: try pkg-config, then find.
const hdf5LinkFlags = block:
  let pc = staticExec("pkg-config --libs hdf5 2>/dev/null || pkg-config --libs hdf5-serial 2>/dev/null").strip()
  if pc.len > 0: pc
  else:
    let lib = staticExec("find /usr/lib /usr/local/lib -name 'libhdf5*.so' 2>/dev/null | head -1").strip()
    if lib.len > 0:
      "-L" & staticExec("dirname " & lib).strip() & " -l" &
        staticExec("basename " & lib & " | sed 's/^lib//;s/\\.so$//'").strip()
    else: "-lhdf5"

const hdf5CompileFlags = block:
  let pc = staticExec("pkg-config --cflags hdf5 2>/dev/null || pkg-config --cflags hdf5-serial 2>/dev/null").strip()
  if pc.len > 0: pc
  else: ""

{.passL: hdf5LinkFlags.}
when hdf5CompileFlags.len > 0:
  {.passC: hdf5CompileFlags.}

const libhdf5 = "(libhdf5|libhdf5_serial).so"

type
  hid_t  = int64
  hsize_t = uint64
  herr_t = int32
  htri_t = int32

type H5File* = object
  fid: hid_t

var H5T_NATIVE_DOUBLE_g {.importc: "H5T_NATIVE_DOUBLE_g", dynlib: libhdf5.}: hid_t

template H5T_NATIVE_DOUBLE(): hid_t = H5T_NATIVE_DOUBLE_g

const
  H5F_ACC_TRUNC  = 0x0002'u32   # overwrite existing
  H5P_DEFAULT: hid_t = 0
  H5S_ALL: hid_t     = 0

proc H5Fcreate(
  filename: cstring; 
  flags: uint32;
  create_plist, access_plist: hid_t
): hid_t {.cdecl, importc: "H5Fcreate", dynlib: libhdf5.}

proc H5Fclose(file_id: hid_t): herr_t
  {.cdecl, importc: "H5Fclose", dynlib: libhdf5.}

proc H5Lexists(loc_id: hid_t; name: cstring; lapl_id: hid_t): htri_t
  {.cdecl, importc: "H5Lexists", dynlib: libhdf5.}

proc H5Gcreate2(loc_id: hid_t; name: cstring;
                lcpl_id, gcpl_id, gapl_id: hid_t): hid_t
  {.cdecl, importc: "H5Gcreate2", dynlib: libhdf5.}

proc H5Gopen2(loc_id: hid_t; name: cstring; gapl_id: hid_t): hid_t
  {.cdecl, importc: "H5Gopen2", dynlib: libhdf5.}

proc H5Gclose(group_id: hid_t): herr_t
  {.cdecl, importc: "H5Gclose", dynlib: libhdf5.}

proc H5Screate_simple(
  rank: cint; 
  dims: ptr hsize_t;
  maxdims: ptr hsize_t
): hid_t {.cdecl, importc: "H5Screate_simple", dynlib: libhdf5.}

proc H5Sclose(space_id: hid_t): herr_t
  {.cdecl, importc: "H5Sclose", dynlib: libhdf5.}

proc H5Dcreate2(
  loc_id: hid_t; 
  name: cstring;
  dtype_id, space_id, lcpl_id, dcpl_id, dapl_id: hid_t
): hid_t {.cdecl, importc: "H5Dcreate2", dynlib: libhdf5.}

proc H5Dwrite(
  dataset_id, mem_type_id, mem_space_id, file_space_id, plist_id: hid_t;
  buf: pointer
): herr_t {.cdecl, importc: "H5Dwrite", dynlib: libhdf5.}

proc H5Dclose(dataset_id: hid_t): herr_t
  {.cdecl, importc: "H5Dclose", dynlib: libhdf5.}

proc newH5File*(filename: string; mode: string = "rw"): H5File =
  ## Open (always creating/truncating) an HDF5 file.
  ## `mode` is accepted for API compatibility but always truncates.
  let fid = H5Fcreate(filename.cstring, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)
  if fid < 0:
    raise newException(IOError, "H5File: failed to create/open " & filename)
  result.fid = fid

proc close*(f: var H5File): int =
  ## Close the HDF5 file. Returns 0 on success.
  result = H5Fclose(f.fid).int
  f.fid = -1

proc requireGroup(fid: hid_t; path: string): hid_t =
  ## Open the group at `path` (relative, no leading slash), creating any
  ## missing intermediate groups along the way.
  var current = fid
  var owned = false  # whether we own `current` (need to close it)
  for part in path.split('/'):
    if part.len == 0: continue
    let exists = H5Lexists(current, part.cstring, H5P_DEFAULT)
    let child =
      if exists > 0:
        H5Gopen2(current, part.cstring, H5P_DEFAULT)
      else:
        H5Gcreate2(current, part.cstring, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)
    if owned: discard H5Gclose(current)
    current = child
    owned = true
  return current

proc `[]=`*(f: var H5File; path: string; data: seq[float]) =
  ## Write `data` as a 1-D float64 dataset at `path` (e.g. "/group/dataset").
  ## Intermediate groups are created automatically.
  let path = if path.len > 0 and path[0] == '/': path[1..^1] else: path

  # Split into group path + dataset name
  let slash = path.rfind('/')
  let (groupPath, dsName) = (
    if slash < 0: ("", path)
    else: (path[0..<slash], path[slash+1..^1])
  )

  # Open/create the parent group
  let gid = (
    if groupPath.len == 0: f.fid
    else: requireGroup(f.fid, groupPath)
  )
  let ownGroup = groupPath.len > 0

  # Create dataspace
  var dim = hsize_t(data.len)
  let sid = H5Screate_simple(1, addr dim, nil)
  if sid < 0:
    if ownGroup: discard H5Gclose(gid)
    raise newException(IOError, "H5File[]=: failed to create dataspace for " & path)

  # Create dataset
  let did = H5Dcreate2(
    gid, 
    dsName.cstring,
    H5T_NATIVE_DOUBLE(), 
    sid,
    H5P_DEFAULT, 
    H5P_DEFAULT, 
    H5P_DEFAULT
  )
  if did < 0:
    discard H5Sclose(sid)
    if ownGroup: discard H5Gclose(gid)
    raise newException(IOError, "H5File[]=: failed to create dataset " & path)

  # Write data
  let err = H5Dwrite(
    did, H5T_NATIVE_DOUBLE(), 
    H5S_ALL, 
    H5S_ALL,
    H5P_DEFAULT, 
    data[0].unsafeAddr
  )
  discard H5Dclose(did)
  discard H5Sclose(sid)
  if ownGroup: discard H5Gclose(gid)
  if err < 0:
    raise newException(IOError, "H5File[]=: write failed for " & path)
