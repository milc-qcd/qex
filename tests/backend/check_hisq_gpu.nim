# Include the actual production CPU reference and coupled flow implementation.
include ../../src/alphas/hisqflowutils
import std/math
import hisqflowgpu

qexInit()
let lo = newLayout(intSeqParam("lat", @[4,4,4,8]))
var
  g = lo.newGauge()
  savedGauge = lo.newGauge()
  psi = lo.ColorVector()
  reference = lo.ColorVector()
  difference = lo.ColorVector()
  gpuResult = lo.ColorVector()
  tmp = lo.ColorVector()
  rngState = newRNGField(RngMilc6, lo, 314159265.uint64)
let workspace = newHisqFlowGpu(g, psi)

proc checkRelative(label: string, error2, reference2: float) =
  let relative = sqrt(error2 / max(reference2, 1e-100))
  echo "CHECK ", label, ": relative error = ", relative
  doAssert relative < 1e-11, label

proc compare(label: string, actual, expected: auto) =
  threads:
    difference := actual - expected
  checkRelative(label, difference.norm2, expected.norm2)

# Independent free-field link construction avoids the CPU ShiftB restriction
# on three-hop shifts when a local outer extent is smaller than three.
if intParam("haloOnly", 0) != 0:
  var links = newSeq[typeof(g[0])](8)
  for k in 0..<8: links[k] = g[0].newOneOf
  var momentum: array[4,float]
  var eigenvalue = 0.0
  for mu in 0..<4:
    momentum[mu] = (if mu == 3: PI else: 2.0*PI) / lo.physGeom[mu].float
    let d = sin(momentum[mu]) * (1.0 + sin(momentum[mu])*sin(momentum[mu])/6.0)
    eigenvalue -= d*d
  threads:
    let coords = lo.coords
    for i in psi.sites:
      var theta = 0.0
      for mu in 0..<4: theta += momentum[mu] * coords[mu][i].float
      for color in 0..<3:
        psi{i}[color].re := (color+1).float * cos(theta)
        psi{i}[color].im := (color+1).float * sin(theta)
      for mu in 0..<4:
        var eta = 1.0
        for nu in 0..<mu:
          if coords[nu][i] mod 2 != 0: eta = -eta
        for term in 0..<2:
          let hop = if term == 0: 1 else: 3
          let coefficient = if term == 0: 9.0/8.0 else: -1.0/24.0
          var bc = 1.0
          if mu == 3 and ((coords[mu][i]+hop) div lo.physGeom[mu]) mod 2 != 0:
            bc = -1.0
          links[2*mu+term]{i} := coefficient * eta * bc
  workspace.updateLinks(links)
  gpuResult := workspace.laplacian(psi)
  reference := eigenvalue * psi
  compare("analytic HISQ eigenvalue across narrow rank domains", gpuResult, reference)
  workspace.free()
  echo "PASS: analytic Naik halo checks"
  qexFinalize()
  quit(0)

threads:
  g.unit()
  savedGauge := g
let freeStag = hisqStag(g)
workspace.updateLinks(freeStag.g)
for momentum in [
    [0.0,0.0,0.0,PI/lo.physGeom[3].float],
    [PI/2.0,PI,0.0,3.0*PI/lo.physGeom[3].float],
    [PI,PI,PI,7.0*PI/lo.physGeom[3].float]]:
  threads:
    let coords = lo.coords
    for i in psi.sites:
      var theta = 0.0
      for mu in 0..<4:
        theta += momentum[mu] * coords[mu][i].float
      for color in 0..<3:
        psi{i}[color].re := cos(theta) * (color+1).float
        psi{i}[color].im := sin(theta) * (color+1).float
  var eigenvalue = 0.0
  for p in momentum:
    let d = sin(p) * (1.0 + sin(p)*sin(p)/6.0)
    eigenvalue -= d*d
  gpuResult := workspace.laplacian(psi)
  reference := eigenvalue * psi
  compare("free-field HISQ eigenvalue", gpuResult, reference)
  reference := stagLap4D(g, freeStag, psi, 0.0)
  compare("free-field CPU reference", gpuResult, reference)

# Reusing the same workspace after changing links and sources detects stale
# uploads; directional checks exercise the one-/three-hop terms independently.
for iteration in 0..<2:
  threads:
    g.random rngState
    psi.gaussian rngState
    savedGauge := g
  let stag = hisqStag(g)
  workspace.updateLinks(stag.g)
  var energy = 0.0
  for mu in 0..<4:
    let dmu = stag.subdirs(@[2*mu, 2*mu+1])
    threads:
      dmu.D(reference, psi, 0.0)
    gpuResult := workspace.directional(psi, mu)
    compare("direction " & $mu, gpuResult, reference)
    energy += reference.norm2
  gpuResult := workspace.laplacian(psi)
  reference := stagLap4D(g, stag, psi, 0.0)
  compare("interacting directional Laplacian", gpuResult, reference)
  let dissipation = redot(psi, gpuResult)
  checkRelative("negative-semidefinite quadratic form",
    (dissipation+energy)*(dissipation+energy), energy*energy)
  doAssert dissipation < 0
  threads:
    stag.D(tmp, psi, 0.0)
    threadBarrier()
    stag.Ddag(reference, tmp, 0.0)
    threadBarrier()
    difference := gpuResult + reference
  doAssert sqrt(difference.norm2/gpuResult.norm2) > 1e-6,
    "Directional Laplacian must not become the square of the full D"
  var gaugeDifference = lo.newGauge()
  threads:
    for mu in 0..<4:
      gaugeDifference[mu] := g[mu] - savedGauge[mu]
  checkRelative("thin gauge unchanged", gaugeDifference.norm2, savedGauge.norm2)

var rejectedMass = false
try:
  discard workspace.laplacian(psi, 0.1)
except AssertionDefect:
  rejectedMass = true
doAssert rejectedMass
workspace.free()
workspace.free() # Explicit cleanup is idempotent.
echo "PASS: accelerator HISQ directional Laplacian checks"

# Exercise the very same coupled RK implementation in both execution modes.
# Three color sources and two complete steps cover all three stage rebuilds,
# workspace reuse, gauge evolution and propagation of errors between steps.
var gCpu = lo.newGauge()
var gGpu = lo.newGauge()
type Propagators = array[3, array[1, array[1, array[1, typeof(psi)]]]]
var phiCpu, phiGpu: Propagators
threads:
  gCpu := g
  gGpu := g
for color in 0..<3:
  phiCpu[color][0][0][0] = psi.newOneOf
  phiGpu[color][0][0][0] = psi.newOneOf
  threads:
    phiCpu[color][0][0][0].gaussian rngState
    phiGpu[color][0][0][0] := phiCpu[color][0][0][0]
let gc = GaugeActionCoeffs(plaq: 1.0)
gc.staggeredFlowImpl(gCpu, phiCpu, 2, 0.01, @[0.0], false):
  discard
gc.staggeredFlowImpl(gGpu, phiGpu, 2, 0.01, @[0.0], true):
  discard
for color in 0..<3:
  compare("two coupled RK steps, color " & $color,
    phiGpu[color][0][0][0], phiCpu[color][0][0][0])
var gaugeDifference = lo.newGauge()
threads:
  for mu in 0..<4:
    gaugeDifference[mu] := gGpu[mu] - gCpu[mu]
checkRelative("coupled gauge evolution", gaugeDifference.norm2, gCpu.norm2)
echo "PASS: coupled HISQ flow CPU/accelerator checks"
qexFinalize()
