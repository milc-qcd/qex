import qex
import comms/halodevice
qexInit()
let lo = newLayout(intSeqParam("lat", @[4,4,4,8]))
var f = lo.ColorVector()
let hl = lo.makeHaloLayout([3,3,3,3], [3,3,3,3])
var h = makeHalo(hl, f)
let reference = makeHalo(hl, f)
var offsets: seq[array[4,int32]]
for mu in 0..<4:
  for hop in [-3, -1, 1, 3]:
    var offset: array[4,int32]
    offset[mu] = hop.int32
    offsets.add offset
let hm = hl.makeHaloMap(getDefaultComm(), offsets)
let ex = newDeviceHaloExchange(h, hm)
threads:
  let cc = lo.coords
  for i in f.sites:
    for c in 0..<3:
      f{i}[c].re := (cc[0][i] + 7*cc[1][i] + 31*cc[2][i] + 127*cc[3][i] + c).float
      f{i}[c].im := 0.1*c.float
f.resident()
f.hostUpdated()
ex.exchange(h)
# Download the halo for comparison against the established host gather.
h.gpuFlagsIncl {gmCpuRead}
h.fromGpu()
reference.update(hm, getDefaultComm())
for i in hm.gather.ldest:
  let j = i.int div lo.V
  let lane = i.int mod lo.V
  doAssert norm2(h.halo[j][asSimd(lane)] - reference.halo[j][asSimd(lane)]) < 1e-24
for i in hm.gather.rdest:
  let j = i.int div lo.V
  let lane = i.int mod lo.V
  doAssert norm2(h.halo[j][asSimd(lane)] - reference.halo[j][asSimd(lane)]) < 1e-24
ex.free()
freeGpuMem(addr f[0])
freeGpuMem(addr h.halo[0])
freeGpuMem(hl)
echo "PASS: device halo packing and local/remote exchange"
qexFinalize()
