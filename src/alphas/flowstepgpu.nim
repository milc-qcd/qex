## Persistent coupled RK3 flow. CPU code controls stages and MPI; all gauge,
## HISQ-link and fermion arithmetic executes through the selected backend.
import qex
import hisqflowgpu, gaugeflowgpu
export hisqflowgpu, gaugeflowgpu

type CoupledFlowGpu*[G,S,F] = ref object
  gauge: G
  stencil: S
  phi, stage1, delta0, delta1, work: seq[F]
  closed: bool

proc newCoupledFlowGpu*(g, sources: auto, coefficients: GaugeActionCoeffs): auto =
  doAssert sources.len > 0
  let gauge = newGaugeFlowGpu(g, coefficients)
  let stencil = newHisqFlowGpu(g, sources[0])
  var w: CoupledFlowGpu[typeof(gauge),typeof(stencil),typeof(sources[0])]
  new w
  w.gauge = gauge
  w.stencil = stencil
  for sourceIndex in 0..<sources.len:
    let src = sources[sourceIndex]
    w.phi.add src.newOneOf
    w.stage1.add src.newOneOf
    w.delta0.add src.newOneOf
    w.delta1.add src.newOneOf
    w.work.add src.newOneOf
    let k = w.phi.high
    for f in [w.phi[k],w.stage1[k],w.delta0[k],w.delta1[k],w.work[k]]:
      f.resident()
    threads: w.phi[k] := src
    w.phi[k].hostUpdated()
  result = w

proc step*(w: CoupledFlowGpu, eps: float) =
  doAssert not w.closed
  tic("GPU coupled RK step")
  # W0: retain L0*phi0 unscaled, and form phi1 = phi0 + eps/4 L0*phi0.
  w.gauge.smear()
  w.stencil.updateDeviceLinks(w.gauge.fat)
  w.gauge.gaugeForce()
  for k in 0..<w.phi.len:
    w.stencil.laplacianInto(w.delta0[k], w.phi[k])
    deviceCombine(w.stage1[k], 1.0,w.phi[k], 0.25*eps,w.delta0[k], 0.0,w.phi[k])
  w.gauge.advance(0,eps)
  # W1: form phi2 using both first and second derivatives.
  w.gauge.smear()
  w.stencil.updateDeviceLinks(w.gauge.fat)
  for k in 0..<w.phi.len:
    w.stencil.laplacianInto(w.delta1[k], w.stage1[k])
    deviceCombine(w.work[k], 1.0,w.phi[k], (8.0/9.0)*eps,w.delta1[k],
                  (-2.0/9.0)*eps,w.delta0[k])
  w.gauge.gaugeForce()
  w.gauge.advance(1,eps)
  # W2: delta0 is no longer needed, so reuse it for L2*phi2.
  w.gauge.smear()
  w.stencil.updateDeviceLinks(w.gauge.fat)
  for k in 0..<w.phi.len:
    w.stencil.laplacianInto(w.delta0[k], w.work[k])
    deviceCombine(w.phi[k], 1.0,w.stage1[k], 0.75*eps,w.delta0[k], 0.0,w.stage1[k])
  w.gauge.gaugeForce()
  w.gauge.advance(2,eps)
  toc("GPU coupled RK step done")

proc download*(w: CoupledFlowGpu, gauge, sources: auto) =
  ## Explicit snapshot for the existing CPU measurement/output interface.
  doAssert not w.closed and sources.len == w.phi.len
  w.gauge.downloadGauge(gauge)
  for k in 0..<sources.len:
    w.phi[k].download()
    threads: sources[k] := w.phi[k]

proc free*(w: CoupledFlowGpu) =
  if w.closed: return
  w.gauge.free()
  w.stencil.free()
  for collection in [w.phi,w.stage1,w.delta0,w.delta1,w.work]:
    for f in collection: freeGpuMem(addr f[0])
  w.closed = true
