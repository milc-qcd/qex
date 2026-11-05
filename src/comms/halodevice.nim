## Halo exchange for fields whose authoritative values live on the device.
## Local neighbors are copied on device. Only packed remote boundary values
## cross host memory for QMP/MPI; full local fields never need a download.
import qex
import backend/[accel, cgfield]
import comms/[halo, halogpu, gather]
export accel, cgfield, halo, halogpu

# These exchange buffers contain scalar color matrices/vectors, never SIMD
# proxy objects. Their host and device representations have identical layout.
proc toGpu*[T:Color](g: var GpuSeq[T], c: seq[T], pgm: ptr GpuMem) =
  pgm.copyIn(unsafeAddr c[0])
proc fromGpu*[T:Color](c: var seq[T], g: GpuSeq[T], pgm: ptr GpuMem) =
  pgm.copyOut(addr c[0])

template scalarSite*(i: SomeInteger): SiteV[1] = SiteV[1](int(i))

type DeviceHaloExchange*[T] = ref object
  map: GatherMap
  localSrc, localDst, sendIdx, recvIdx: seq[int32]
  send, recv: seq[T]

proc resident*(f: Field) =
  ## Suppress automatic host transfers; callers mark explicit host updates.
  f.gpuFlagsExcl {gmCpuRead, gmCpuWrite}
proc hostUpdated*(f: Field) =
  f.gpuFlagsIncl {gmCpuWriteOnce}
proc download*(f: Field) =
  # A new host value is already readable before its first lazy upload.
  if gmCpuWriteOnce in getGpuMem(addr f[0]).flags: return
  f.gpuFlagsIncl {gmCpuRead, gmGpuWrite}
  f.fromGpu()
  f.gpuFlagsExcl {gmCpuRead}

proc newDeviceHaloExchange*(h: Halo, hm: HaloMap): auto =
  type T = evalType(h.field{0})
  var ex: DeviceHaloExchange[T]
  new ex
  ex.map = hm.gather
  ex.localSrc = hm.gather.lidx
  ex.localDst = hm.gather.ldest
  ex.sendIdx = hm.gather.sidx
  ex.recvIdx = hm.gather.rdest
  ex.send = newSeq[T](ex.sendIdx.len)
  ex.recv = newSeq[T](ex.recvIdx.len)
  # Captured sequences must alias this owned storage: ordinary Nim assignment
  # copies a seq, losing its transfer flags and leaving stale cache entries.
  shallow(ex.localSrc)
  shallow(ex.localDst)
  shallow(ex.sendIdx)
  shallow(ex.recvIdx)
  shallow(ex.send)
  shallow(ex.recv)
  for indices in [ex.localSrc, ex.localDst, ex.sendIdx, ex.recvIdx]:
    if indices.len > 0:
      indices.gpuFlagsExcl {gmGpuWrite, gmCpuWrite}
      indices.gpuFlagsIncl {gmCpuWriteOnce}
  if ex.send.len > 0:
    ex.send.gpuFlagsExcl {gmCpuWrite, gmGpuRead}
  if ex.recv.len > 0:
    ex.recv.gpuFlagsExcl {gmCpuWrite, gmGpuWrite}
  h.gpuFlagsExcl {gmCpuRead, gmCpuWrite}
  result = ex

proc exchange*(ex: DeviceHaloExchange, input: Halo) =
  tic("device halo exchange")
  var h = input
  h.gpuFlagsIncl {gmGpuWrite}
  var localSrc = ex.localSrc
  var localDst = ex.localDst
  var sendIdx = ex.sendIdx
  var recvIdx = ex.recvIdx
  var send = ex.send
  var recv = ex.recv
  let base = h.nOut * h.field.V
  let nl = localSrc.len
  let ns = send.len
  let nr = recv.len
  let comm = getDefaultComm()
  # Post receives before sends, following the existing QEX gather protocol.
  for msg in ex.map.rmsginfo:
    comm.pushRecv(msg.rank, addr recv[msg.start], msg.count.int * sizeof(recv[0]))
  comm.barrier()
  if ns > 0:
    onGpu(ns):
      for k in gpuRange(ns):
        send[k] := h.field[scalarSite(sendIdx[k])]
  if nl > 0:
    onGpu(nl):
      for k in gpuRange(nl):
        h[scalarSite(base + localDst[k].int)] := h.field[scalarSite(localSrc[k])]
  for msg in ex.map.smsginfo:
    comm.pushSend(msg.rank, addr send[msg.start], msg.count.int * sizeof(send[0]))
  for i in 0..<ex.map.rmsginfo.len:
    comm.waitRecv(i, free=false)
  comm.freeRecvs(ex.map.rmsginfo.len)
  comm.waitSends(ex.map.smsginfo.len)
  if nr > 0:
    recv.gpuFlagsIncl {gmCpuWriteOnce}
    onGpu(nr):
      for k in gpuRange(nr):
        h[scalarSite(base + recvIdx[k].int)] := recv[k]
  toc("device halo exchange done")

proc free*(ex: DeviceHaloExchange) =
  for indices in [ex.localSrc, ex.localDst, ex.sendIdx, ex.recvIdx]:
    if indices.len > 0: freeGpuMem(unsafeAddr indices[0])
  if ex.send.len > 0: freeGpuMem(addr ex.send[0])
  if ex.recv.len > 0: freeGpuMem(addr ex.recv[0])

proc deviceCopy*(destination, source: Field) =
  var dst = destination
  var src = source
  dst.gpuFlagsIncl {gmGpuWrite}
  onGpu(dst.l):
    for s in gpuSites(dst): dst[s] := src[s]

proc deviceCombine*(destination: Field; a: float; x: Field;
                    b: float; y: Field; c: float; z: Field) =
  ## Per-site vector combination; destination may alias any input field.
  var dst = destination
  var fx = x
  var fy = y
  var fz = z
  dst.gpuFlagsIncl {gmGpuWrite}
  onGpu(dst.l):
    for s in gpuSites(dst):
      let value = a*fx[s] + b*fy[s] + c*fz[s]
      dst[s] := value
