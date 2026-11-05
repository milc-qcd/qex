## Device Wilson/rectangle gauge evolution and two-level HISQ link construction.
## Explicit path products favor a transparent CPU-reference comparison. Gauge
## and spinor halo traffic uses compact host-staged MPI via halodevice.
import qex, gauge
import gauge/[gaugeAction, hisqsmear, fat7l]
import comms/halodevice
export halodevice

# Expand scalar matrix helpers on the CPU fallback exactly as for a device
# kernel. This also keeps their temporary values inside the site loop.
template onFlowGpu(n: untyped, body: untyped) =
  when backendIsCpu:
    onGpu(n):
      inlineProcs:
        body
  else:
    onGpu(n):
      body

type
  GaugePath = object
    steps: seq[int32] # +/- (axis+1), traversed from the output site
    weight: float
  DevicePaths = ref object
    steps, lengths: seq[int32]
    weights: seq[float]
  GaugeFlowGpu*[H,E,F] = ref object
    thin*, phased, projected: seq[H]
    exchanges: seq[E]
    fat*, force*, momentum*: seq[F]
    first: seq[F]
    phase: seq[int32]
    fatPaths1, fatPaths2, longPaths, forcePaths: seq[DevicePaths]
    closed: bool

proc path(steps: seq[int32], weight: float): GaugePath =
  GaugePath(steps: steps, weight: weight)
proc wrap(p: seq[int32], d: int32): seq[int32] = @[d] & p & @[-d]
proc fatPaths(mu: int, c: Fat7lCoefs): seq[GaugePath] =
  let straight = @[int32(mu+1)]
  result.add path(straight, c.oneLink - 6*c.lepage)
  for nu in 0..<4:
    if nu == mu: continue
    for sn in [-1'i32,1'i32]:
      let p3 = wrap(straight, sn*int32(nu+1))
      result.add path(p3, c.threeStaple)
      for sl in [-1'i32,1'i32]:
        result.add path(wrap(p3, sl*int32(nu+1)), c.lepage)
      for rho in 0..<4:
        if rho == mu or rho == nu: continue
        for sr in [-1'i32,1'i32]:
          let p5 = wrap(p3, sr*int32(rho+1))
          result.add path(p5, c.fiveStaple)
          for sigma in 0..<4:
            if sigma == mu or sigma == nu or sigma == rho: continue
            for ss in [-1'i32,1'i32]:
              result.add path(wrap(p5, ss*int32(sigma+1)), c.sevenStaple)

proc staples(mu: int, c: GaugeActionCoeffs): seq[GaugePath] =
  let m = int32(mu+1)
  for nu in 0..<4:
    if nu == mu: continue
    for sign in [-1'i32,1'i32]:
      let n = sign*int32(nu+1)
      result.add path(@[n,m,-n], c.plaq/3)
      result.add path(@[n,n,m,-n,-n], c.rect/3)
      result.add path(@[n,m,m,-n,-m], c.rect/3)
      result.add path(@[-m,n,m,m,-n], c.rect/3)

proc devicePaths(paths: seq[GaugePath]): DevicePaths =
  new result
  for p in paths:
    if p.weight == 0: continue
    result.lengths.add p.steps.len.int32
    result.weights.add p.weight
    for i in 0..<8:
      result.steps.add (if i < p.steps.len: p.steps[i] else: 0'i32)
  # Stable addresses are required by the pointer-keyed GPU allocation cache.
  shallow(result.steps)
  shallow(result.lengths)
  shallow(result.weights)
  for indices in [result.steps, result.lengths]:
    if indices.len > 0:
      indices.gpuFlagsExcl {gmCpuWrite, gmGpuWrite}
      indices.gpuFlagsIncl {gmCpuWriteOnce}
  if result.weights.len > 0:
    result.weights.gpuFlagsExcl {gmCpuWrite, gmGpuWrite}
    result.weights.gpuFlagsIncl {gmCpuWriteOnce}

proc newGaugeFlowGpu*(g: auto, c: GaugeActionCoeffs): auto =
  doAssert g.len == 4 and g[0][0].nrows == 3
  doAssert c.pgm == 0 and c.adjplaq == 0,
    "GPU gauge flow supports plaquette and rectangle actions"
  let lo = g[0].l
  let hl = lo.makeHaloLayout([3,3,3,3], [3,3,3,3])
  let hc = newHisq(0.0, 1.0)
  var firstPaths, secondPaths, longPaths, forcePaths: array[4,seq[GaugePath]]
  var offsets: array[4,seq[array[4,int32]]]
  for mu in 0..<4:
    firstPaths[mu] = fatPaths(mu, hc.fat7first)
    secondPaths[mu] = fatPaths(mu, hc.fat7second)
    longPaths[mu] = @[path(@[int32(mu+1),int32(mu+1),int32(mu+1)], hc.naik)]
    forcePaths[mu] = staples(mu,c)
    # Only gather the exact link displacements visited by nonzero paths.
    for paths in [firstPaths[mu], secondPaths[mu], longPaths[mu], forcePaths[mu]]:
      for p in paths:
        if p.weight == 0: continue
        var position: array[4,int32]
        for step in p.steps:
          let axis = abs(step).int-1
          if step < 0: dec position[axis]
          if position notin offsets[axis]: offsets[axis].add position
          if step > 0: inc position[axis]
  type H = typeof(makeHalo(hl,g[0]))
  type E = typeof(newDeviceHaloExchange(default(H), default(HaloMap[typeof(lo)])))
  var w: GaugeFlowGpu[H,E,typeof(g[0])]
  new w
  w.fat = newSeq[typeof(g[0])](8)
  w.phase = newSeq[int32](4*lo.nSites)
  let phaseMasks = [8,9,11,0] # QEX stagPhase convention, not a different eta convention
  let coords = lo.coords
  for mu in 0..<4:
    w.thin.add makeHalo(hl,g[0].newOneOf)
    w.phased.add makeHalo(hl,g[0].newOneOf)
    w.projected.add makeHalo(hl,g[0].newOneOf)
    w.exchanges.add newDeviceHaloExchange(w.thin[mu], hl.makeHaloMap(getDefaultComm(),offsets[mu]))
    for collection in [w.thin, w.phased, w.projected]:
      collection[mu].field.resident()
      collection[mu].gpuFlagsExcl {gmCpuRead,gmCpuWrite}
    w.force.add g[0].newOneOf
    w.momentum.add g[0].newOneOf
    w.first.add g[0].newOneOf
    for f in [w.force[mu],w.momentum[mu],w.first[mu]]: f.resident()
    for k in 2*mu..2*mu+1:
      w.fat[k] = g[0].newOneOf
      w.fat[k].resident()
    threads: w.thin[mu].field := g[mu]
    w.thin[mu].field.hostUpdated()
    for i in 0..<lo.nSites:
      var sign = 0
      for nu in 0..<4:
        sign += (phaseMasks[mu] shr nu) and coords[nu][i].int
      if mu == 3 and coords[3][i] == lo.physGeom[3]-1: inc sign
      w.phase[mu*lo.nSites+i] = (if (sign and 1) == 0: 1'i32 else: -1'i32)
    w.fatPaths1.add devicePaths(firstPaths[mu])
    w.fatPaths2.add devicePaths(secondPaths[mu])
    w.longPaths.add devicePaths(longPaths[mu])
    w.forcePaths.add devicePaths(forcePaths[mu])
  shallow(w.thin)
  shallow(w.phased)
  shallow(w.projected)
  shallow(w.phase)
  w.phase.gpuFlagsExcl {gmCpuWrite,gmGpuWrite}
  w.phase.gpuFlagsIncl {gmCpuWriteOnce}
  result = w

proc evaluatePaths(destination: Field, input: seq[Halo], paths: DevicePaths) =
  var dst = destination
  var u = input
  let hl = input[0].layout
  var steps = paths.steps
  var lengths = paths.lengths
  var weights = paths.weights
  let count = lengths.len
  let n = dst.l.nSites
  dst.gpuFlagsIncl {gmGpuWrite}
  if count == 0:
    onFlowGpu(n):
      for i in gpuRange(n): dst[scalarSite(i)] := 0
    return
  onFlowGpu(n):
    for i in gpuRange(n):
      let s = scalarSite(i)
      var total {.noinit.}: typeof(load1(dst[s]))
      total := 0
      for p in 0..<count:
        var product {.noinit.}: typeof(load1(dst[s]))
        product := 1
        var site = s
        for k in 0..<lengths[p]:
          let step = steps[8*p+k]
          let axis = abs(step).int-1
          if step < 0:
            site = hl.nbrBck(axis, site)
            let next = product*u[axis][site].adj
            product := next
          else:
            let next = product*u[axis][site]
            product := next
            site = hl.nbrFwd(axis, site)
        total += weights[p]*product
      dst[s] := total

proc smear*(w: GaugeFlowGpu) =
  ## Construct phased fat/Naik links from unphased thin links entirely on device.
  doAssert not w.closed
  tic("GPU HISQ smearing")
  let n = w.thin[0].field.l.nSites
  var phases = w.phase
  for mu in 0..<4:
    var src = w.thin[mu].field
    var dst = w.phased[mu].field
    dst.gpuFlagsIncl {gmGpuWrite}
    onFlowGpu(n):
      for i in gpuRange(n):
        let s = scalarSite(i)
        dst[s] := phases[mu*n+i].float * src[s]
    w.exchanges[mu].exchange(w.phased[mu])
  for mu in 0..<4:
    evaluatePaths(w.first[mu], w.phased, w.fatPaths1[mu])
    var src = w.first[mu]
    var dst = w.projected[mu].field
    dst.gpuFlagsIncl {gmGpuWrite}
    onFlowGpu(n):
      for i in gpuRange(n):
        let s = scalarSite(i)
        var value = load1(src[s])
        value.projectU()
        dst[s] := value
    w.exchanges[mu].exchange(w.projected[mu])
  for mu in 0..<4:
    evaluatePaths(w.fat[2*mu], w.projected, w.fatPaths2[mu])
    evaluatePaths(w.fat[2*mu+1], w.projected, w.longPaths[mu])
  toc("GPU HISQ smearing done")

proc gaugeForce*(w: GaugeFlowGpu) =
  doAssert not w.closed
  tic("GPU gauge force")
  for mu in 0..<4: w.exchanges[mu].exchange(w.thin[mu])
  for mu in 0..<4:
    evaluatePaths(w.force[mu], w.thin, w.forcePaths[mu])
    var u = w.thin[mu].field
    var f = w.force[mu]
    let n = u.l.nSites
    onFlowGpu(n):
      for i in gpuRange(n):
        let s = scalarSite(i)
        let value = u[s]*f[s].adj
        f[s].projectTAH(value)
  toc("GPU gauge force done")

proc flowExp(m: Mat1): auto =
  ## QEX exp(Mat) uses order-four expm1 with scale=20, followed by squaring.
  ## Spell out that fixed choice so device inlining does not expand every
  ## unused order and Pade branch of the runtime ExpParam dispatcher.
  let scaled = (1.0/1048576.0)*m
  let squared = scaled*scaled
  var polynomial = (1.0/24.0)*squared
  polynomial += (1.0/6.0)*scaled
  polynomial += 0.5
  var r = polynomial*squared + scaled
  for i in 0..<20:
    let old = r
    var plusTwo = r
    plusTwo += 2.0
    r := old*plusTwo
  r += 1.0
  r

proc advance*(w: GaugeFlowGpu, stage: int, eps: float) =
  ## Same three RK gauge updates and normalization as staggeredFlowImpl.
  doAssert not w.closed and stage in 0..2
  let a = (if stage == 0: -0.25 elif stage == 1: -8.0/9.0 else: -0.75) * eps * 3.0
  let b = (if stage == 0: 0.0 elif stage == 1: -17.0/9.0 else: -1.0)
  for mu in 0..<4:
    var u = w.thin[mu].field
    var f = w.force[mu]
    var p = w.momentum[mu]
    let n = u.l.nSites
    onFlowGpu(n):
      for i in gpuRange(n):
        let s = scalarSite(i)
        var v = load1(f[s])
        v := a*f[s]
        if stage != 0: v += b*p[s]
        let next = asColor(flowExp(v[]))*u[s]
        u[s] := next
        p[s] := v

proc downloadGauge*(w: GaugeFlowGpu, destination: auto) =
  for mu in 0..<4:
    w.thin[mu].field.download()
    threads: destination[mu] := w.thin[mu].field

proc free*(p: DevicePaths) =
  if p.steps.len > 0: freeGpuMem(unsafeAddr p.steps[0])
  if p.lengths.len > 0: freeGpuMem(unsafeAddr p.lengths[0])
  if p.weights.len > 0: freeGpuMem(unsafeAddr p.weights[0])
proc free*(w: GaugeFlowGpu) =
  if w.closed: return
  for ex in w.exchanges: ex.free()
  for collection in [w.thin,w.phased,w.projected]:
    for h in collection:
      freeGpuMem(addr h.field[0])
      freeGpuMem(addr h.halo[0])
    if collection.len > 0: freeGpuMem(unsafeAddr collection[0])
  for collection in [w.fat,w.force,w.momentum,w.first]:
    for f in collection: freeGpuMem(addr f[0])
  freeGpuMem(addr w.phase[0])
  for collection in [w.fatPaths1,w.fatPaths2,w.longPaths,w.forcePaths]:
    for p in collection: p.free()
  freeGpuMem(w.thin[0].layout)
  w.closed = true
