include ../../src/alphas/hisqflowutils
import gaugeflowgpu
qexInit()
let lo = newLayout(intSeqParam("lat", @[4,4,4,8]))
var g = lo.newGauge()
var rngState = newRNGField(RngMilc6, lo, 271828.uint64)
threads: g.random rngState
for action in [GaugeActionCoeffs(plaq:1.0), gaugeActRect(1.2, -1.0/12.0)]:
  let w = newGaugeFlowGpu(g,action)
  var cpuForce = lo.newGauge()
  var diff = lo.newGauge()
  w.downloadGauge(diff)
  for mu in 0..<4:
    threads: diff[mu] -= g[mu]
  doAssert diff.norm2 == 0.0
  action.gaugeForce(g,cpuForce)
  w.gaugeForce()
  for mu in 0..<4:
    w.force[mu].download()
    threads: diff[mu] := w.force[mu] - cpuForce[mu]
  let forceError = sqrt(diff.norm2/cpuForce.norm2)
  echo "CHECK GPU gauge force: ", forceError
  doAssert (cast[uint64](forceError) and 0x7ff0000000000000'u64) != 0x7ff0000000000000'u64
  doAssert forceError < 1e-11
  w.smear()
  let stag = hisqStag(g)
  for k in 0..<8:
    w.fat[k].download()
    threads: diff[0] := w.fat[k] - stag.g[k]
    let error = sqrt(diff[0].norm2/stag.g[k].norm2)
    echo "CHECK GPU HISQ link ", k, ": ", error
    doAssert (cast[uint64](error) and 0x7ff0000000000000'u64) != 0x7ff0000000000000'u64
    doAssert error < 1e-11
  var evolved = lo.newGauge()
  var momentum = lo.newGauge()
  threads: evolved := g
  for stage in 0..<3:
    action.gaugeForce(evolved,cpuForce)
    let a = (if stage == 0: -0.25 elif stage == 1: -8.0/9.0 else: -0.75)*0.01*3.0
    let b = (if stage == 0: 0.0 elif stage == 1: -17.0/9.0 else: -1.0)
    threads:
      for mu in 0..<4:
        for e in evolved[mu]:
          var v = load1(cpuForce[mu][e])
          v := a*cpuForce[mu][e]
          if stage != 0: v += b*momentum[mu][e]
          let next = exp(v)*evolved[mu][e]
          evolved[mu][e] := next
          momentum[mu][e] := v
    w.gaugeForce()
    w.advance(stage,0.01)
    w.downloadGauge(diff)
    threads:
      for mu in 0..<4: diff[mu] -= evolved[mu]
    let error = sqrt(diff.norm2/evolved.norm2)
    echo "CHECK GPU gauge RK stage ", stage, ": ", error
    doAssert (cast[uint64](error) and 0x7ff0000000000000'u64) != 0x7ff0000000000000'u64
    doAssert error < 1e-11
  w.free()
  w.free()
  doAssert dumpGpuMem().startsWith("GpuMem items: 0 "), dumpGpuMem()
echo "PASS: gauge force and HISQ smearing GPU kernels"
qexFinalize()
