# Compare the exact HISQ inversion operator used by the application, then
# exercise SYCL flow after QUDA releases its workspaces on the same device.
include ../../src/alphas/hisqflowutils
import hisqsolver
import quda/qudaWrapper
import std/math

when not haveQuda():
  {.error: "Build check_quda_hisq with QUDA_DIR set".}

qexInit()
let lo = newLayout(intSeqParam("lat", @[8,8,8,8]))
var
  g = lo.newGauge()
  source = lo.ColorVector()
  cpu = lo.ColorVector()
  accelerated = lo.ColorVector()
  residual = lo.ColorVector()
  difference = lo.ColorVector()
  rng = newRNGField(RngMilc6, lo, 271828182.uint64)
const mass = 0.06730
const requestedR2 = 1e-20

proc check(label: string; value, bound: float) =
  echo "CHECK ", label, ": ", value, " (limit ", bound, ")"
  doAssert (cast[uint64](value) and 0x7ff0000000000000'u64) != 0x7ff0000000000000'u64, label
  doAssert value <= bound, label

doAssert selectHisqSolver("qex") == sbQex
doAssert selectHisqSolver("cpu") == sbQex
doAssert selectHisqSolver("quda") == sbQuda
var invalidRejected = false
try:
  discard selectHisqSolver("invalid")
except ValueError:
  invalidRejected = true
doAssert invalidRejected

for gaugeKind in ["unit", "random"]:
  threads:
    if gaugeKind == "unit": g.unit()
    else: g.random rng
  let stag = hisqStag(g)
  for parity in ["even", "odd", "all"]:
    threads:
      source.gaussian rng
      if parity == "even": source.odd := 0
      if parity == "odd": source.even := 0
      cpu := 0
      accelerated := 0
    var cpuParams = initSolverParams()
    cpuParams.backend = sbQex
    cpuParams.r2req = requestedR2
    cpuParams.maxits = 10000
    var gpuParams = cpuParams
    gpuParams.backend = sbQuda
    stag.solve(cpu, source, mass, cpuParams)
    stag.solve(accelerated, source, mass, gpuParams)
    let sourceNorm = source.norm2
    # QEX's D is independent of QUDA's reported residual and normalizations.
    threads:
      stag.D(residual, accelerated, mass)
      threadBarrier()
      residual := source - residual
      difference := accelerated - cpu
    check(gaugeKind & " " & parity & " true QUDA residual squared",
      residual.norm2 / sourceNorm, 1.01*requestedR2)
    check(gaugeKind & " " & parity & " CPU/QUDA propagator relative error",
      sqrt(difference.norm2 / cpu.norm2), 1e-8)
    threads:
      stag.D(residual, cpu, mass)
      threadBarrier()
      residual := source - residual
    check(gaugeKind & " " & parity & " true CPU residual squared",
      residual.norm2 / sourceNorm, 1.01*requestedR2)
    doAssert gpuParams.iterations > 0

qudaShutdown()
qudaShutdown() # CPU selection and repeated cleanup must also be safe.
echo "PASS: QUDA HISQ inversion, both parities, full source, true residuals"

var gCpu = lo.newGauge()
var gGpu = lo.newGauge()
type Propagators = array[3, array[1, array[1, array[1, typeof(source)]]]]
var phiCpu, phiGpu: Propagators
threads:
  gCpu := g
  gGpu := g
for color in 0..<3:
  phiCpu[color][0][0][0] = source.newOneOf
  phiGpu[color][0][0][0] = source.newOneOf
  threads:
    phiCpu[color][0][0][0] := (color+1).float * cpu
    phiGpu[color][0][0][0] := (color+1).float * accelerated
let gc = GaugeActionCoeffs(plaq: 1.0)
gc.staggeredFlowImpl(gCpu, phiCpu, 2, 0.01, @[0.0], false):
  discard
gc.staggeredFlowImpl(gGpu, phiGpu, 2, 0.01, @[0.0], true):
  discard
var dg = lo.newGauge()
threads:
  for mu in 0..<4: dg[mu] := gGpu[mu] - gCpu[mu]
check("gauge flow after QUDA cleanup", sqrt(dg.norm2/gCpu.norm2), 1e-11)
for color in 0..<3:
  threads:
    difference := phiGpu[color][0][0][0] - phiCpu[color][0][0][0]
  check("QUDA propagator through coupled flow, color " & $color,
    sqrt(difference.norm2/phiCpu[color][0][0][0].norm2), 1e-8)
doAssert dumpGpuMem().startsWith("GpuMem items: 0 "), dumpGpuMem()
echo "PASS: QUDA solve followed by coupled SYCL gauge/fermion flow"
qexFinalize()
