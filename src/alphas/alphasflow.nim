## Gauge gradient flow with Wilson, rectangle and Zeuthen actions.
##
## CPU builds use QEX's gauge routines. GPU builds keep links, forces and
## Runge--Kutta work fields on the device throughout each `gradientFlow` call.
## Only reduced measurements and MPI boundary buffers return to the host;
## the evolved links are copied back when the call finishes.
import qex
import alphas
import backend/accelbase
when backendIsGpu:
  import backend/[accel, cgfield]
  import comms/[halo, halogpu, gather]
  export accel, cgfield, halogpu

import std/[json, strutils, sequtils, tables]
from std/math import round, PI


const
  banner = """
|-----------------------------------------------------------------------|
 Quantum EXpressions (QEX) gradient flow

 QEX GitHub: https://github.com/jcosborn/qex

 QEX authors: James Osborn & Xiao-Yong Jin (Argonne National Laboratory)
 QEX gradient flow authors: 
   - James Osborn (Argonne National Laboratory)
   - Xiao-Yong Jin (Argonne National Laboratory)
 Modified and extended by:  
   - Curtis Taylor Peterson (Michigan State University)
     * Contact: curtistaylorpetersonwork@gmail.com
 Cite: Proceedings of Science (PoS) LATTICE2016 (2017) 271
|-----------------------------------------------------------------------|
"""

let # defaults for testing/regression
  defaultLat = @[8, 8, 8, 16]
  lastFlowTime = (0.5*defaultLat[^1].float)*(0.5*defaultLat[^1].float)/8.0
  defaultMeasurements = ["plaquette", "rectangle", "clover", "topology", "polyakov"]
  defaultInputs = %* {
    "lattice-geometry": defaultLat,
    #"rank-geometry": [2, 2, 1, 2],
    #"simd-geometry": [2, 2, 1, 2],
    "flows": {
      "wilson": {
        "step-sizes": [0.02, 0.1],
        "step-size-transition-flow-times": [5.0, lastFlowTime],
        "measurements": defaultMeasurements
      }, 
      "symanzik": {
        "step-sizes": [0.02, 0.1],
        "step-size-transition-flow-times": [5.0, lastFlowTime],
        "measurements": defaultMeasurements
      },
      "zeuthen": {
        "step-sizes": [0.02, 0.1],
        "step-size-transition-flow-times": [5.0, lastFlowTime],
        "measurements": defaultMeasurements
      }
    }
  }

const CrSymanzik = -1.0/12.0

type
  GradientFlowKind* = enum 
    WilsonFlow, 
    RectangleFlow, 
    IwasakiFlow, 
    DBW2Flow, 
    SymanzikFlow,
    ZeuthenFlow
  MeasurementKind* = enum 
    Plaquette, 
    Clover, 
    Rectangle,
    Topology, 
    Polyakov

type
  Laplacian[U,U0] = object
    sf, sb: seq[Transporter[U,U,U0]]

type
  GradientFlow*[U,U0] = object
    case kind: GradientFlowKind
      of ZeuthenFlow: 
        lap*: Laplacian[U,U0]
      else: discard
    gc: GaugeActionCoeffs
    meas: seq[MeasurementKind]
    u*: seq[U]
    step*: int
    flowTime*: float
    log*: bool
    logFile*: File

#[ GPU infrastructure ]#

proc flowExp(v: auto): auto =
  ## Evaluate QEX's default matrix exponential inside a flow update.
  ## The explicit polynomial branch avoids GPU inlining of unused Pade code.
  when backendIsGpu and v.nrows > 1:
    const scale = newExpParam().scale
    var r = expm1Poly4((numberType(v)(1.0 / float(1 shl scale))*v)[])
    for i in 1..scale:
      let m = r
      var t = m
      t += 2
      r := m*t
    r += 1
    asColor(r)
  else:
    exp(v)

proc flowAdvance(g0, p0, f0: auto; a: static float; b: float; save: static bool) =
  ## Apply one RK stage: `v = a*p + b*f`, then `g = exp(v)*g`.
  ## Save `v` in `p` when the following stage needs it. Shallow aliases preserve
  ## the sequence addresses QEX uses as GPU allocation keys.
  var g, p, f: type(g0)
  shallowCopy(g, g0)
  shallowCopy(p, p0)
  shallowCopy(f, f0)
  template update(links, momentum, force, site, dimensions, coeff: untyped) =
    for mu in 0..<dimensions:
      var v = load1(force[mu][site])
      v := coeff*force[mu][site]
      when a != 0.0:
        v += a*momentum[mu][site]
      let t = flowExp(v)*links[mu][site]
      when save: momentum[mu][site] := v
      links[mu][site] := t
  let nd = g.len
  when backendIsGpu:
    onGpu(g[0].l.nSites):
      for e in gpuSites(g[0]):
        update(g, p, f, e, nd, b)
  else:
    threads:
      for e in g[0]:
        update(g, p, f, e, nd, b)

when backendIsGpu:
  # An alias prevents onGpu from capturing the SiteV type constructor.
  type FlowSite = SiteV[1]

  type FlowGather = object
    ## QEX communication schedule with immutable indices stored on the GPU.
    localSrc, localDst, sendSrc, recvDst: GpuSeq[int32]
    remote: GatherMap

  type FlowSum = array[4, GpuSum[float]]
    ## Four scalar QEX reductions; HIP cannot shuffle an array-valued sum.

  template toGpu*(x: GpuSeq|bool): auto =
    ## Let onGpu capture device buffers and boolean flags without copying.
    x
  template getGpu*(x, g: GpuSeq|bool): auto = g
  template fromGpu*(x, g: GpuSeq|bool) = discard

  proc flowResident(x: Field|Halo; upload = false) =
    ## Suppress automatic field transfers; optionally upload initial data once.
    x.gpuFlagsExcl {gmCpuRead, gmCpuWrite}
    if upload: x.gpuFlagsIncl {gmCpuWriteOnce}

  proc flowResident(x: seq; upload = false) =
    ## Upload the sequence descriptor once and keep its fields device-resident.
    if x.len > 0:
      x.gpuFlagsIncl {gmCpuWriteOnce}
      x.gpuFlagsExcl {gmGpuWrite}
      for field in x: flowResident(field, upload)

  proc flowFree(x: seq) =
    ## Release device storage for a sequence and its fields or halo buffers.
    ## A halo borrows its interior field, so only its boundary buffer is freed.
    for field in x:
      when field is Halo: freeGpuMem(unsafeAddr field.halo[0])
      else: freeGpuMem(unsafeAddr field[0])
    if x.len > 0: freeGpuMem(unsafeAddr x[0])

  proc flowHaloLayout(lo: auto; reach: int): auto =
    ## Build QEX's halo with interior numbering matching this rank's parity.
    ## QEX assumes an even rank origin; odd local extents can violate that.
    result = lo.makeHaloLayout(newSeqWith(lo.nDim, reach), newSeqWith(lo.nDim, reach))
    # makeHaloLayout numbers the interior using rank-zero coordinates. Reindex
    # it for ranks whose origin has odd parity (e.g. a local temporal extent 3).
    if lo.coordmin.sum mod 2 != 0:
      var perm = toSeq(0..<result.nExt).mapIt(int32(it))
      var x = newSeq[int32](lo.nDim)
      for i in 0..<result.nOut:
        lo.coord(x, 0, i*lo.V)
        for d in 0..<lo.nDim: x[d] += int32(lo.coordmin[d])
        perm[i] = int32(lo.rankIndex(x).index div lo.V)
      var lex = newSeq[int32](result.nExt)
      for i in 0..<result.nExt: lex[perm[i]] = result.lex[i]
      result.lex = lex
      for i in 0..<result.nExt: result.index[i] = perm[result.index[i]]
      for mu in 0..<lo.nDim:
        var fwd, bck = newSeq[int32](result.nExt)
        for i in 0..<result.nExt:
          let f = result.neighborFwd[mu][i]
          let b = result.neighborBck[mu][i]
          fwd[perm[i]] = if f < 0: f else: perm[f]
          bck[perm[i]] = if b < 0: b else: perm[b]
        result.neighborFwd[mu] = fwd
        result.neighborBck[mu] = bck

  proc flowIndices(indices: seq[int32]; offset = 0): GpuSeq[int32] =
    ## Upload gather indices once. The owning FlowGather releases the buffer.
    if indices.len == 0: return
    result = newGpuSeq[int32](indices.len)
    let data = indices.mapIt(it + int32(offset))
    gpuMemCpyToGpu(result.p, unsafeAddr data[0], data.len*sizeof(int32))

  proc newFlowGather(gm: GatherMap; offset = 0): FlowGather =
    ## Adapt a QEX gather to GPU packing. `offset` locates halo destinations
    ## after the interior sites; ordinary shifted fields use zero.
    result.localSrc = flowIndices(gm.lidx)
    result.localDst = flowIndices(gm.ldest, offset)
    result.sendSrc = flowIndices(gm.sidx)
    result.recvDst = flowIndices(gm.rdest, offset)
    # Reuse QEX's exchange with compact, scalar-site send/receive buffers.
    result.remote = GatherMap(smsginfo: gm.smsginfo, rmsginfo: gm.rmsginfo,
      sidx: toSeq(0..<gm.sidx.len).mapIt(int32(it)),
      rdest: toSeq(0..<gm.rdest.len).mapIt(int32(it)))

  proc free(gm: FlowGather) =
    for indices in [gm.localSrc, gm.localDst, gm.sendSrc, gm.recvDst]:
      if indices.n > 0: gpuFree(indices.p)

  proc flowGather(dest0, src: auto; gm: FlowGather) =
    ## Gather scalar sites on the GPU, using QEX for inter-rank exchange.
    ## Local copies stay on-device. Only packed remote sites cross the host,
    ## allowing ordinary MPI without copying complete lattice fields.
    var dest = dest0
    let (ls, ld, ss, rd) = (gm.localSrc, gm.localDst, gm.sendSrc, gm.recvDst)
    let (nl, ns, nr) = (ls.n, ss.n, rd.n)
    if nl > 0:
      onGpu(nl):
        for i in gpuRange(nl): dest[FlowSite(ld[i])] := src[FlowSite(ls[i])]
    if nRanks > 1:
      type T = evalType(src{0})
      var send = newGpuSeq[T](max(1, ns))
      var recv = newGpuSeq[T](max(1, nr))
      var sb = newSeq[T](max(1, ns))
      var rb = newSeq[T](max(1, nr))
      defer:
        gpuFree(send.p)
        gpuFree(recv.p)
      if ns > 0:
        onGpu(ns):
          for i in gpuRange(ns): send[i] := src[FlowSite(ss[i])]
        gpuMemCpyToCpu(addr sb[0], send.p, ns*sizeof(T))
      getDefaultComm().gather(gm.remote, sizeof(T), addr rb[0], addr sb[0])
      if nr > 0:
        gpuMemCpyToGpu(recv.p, addr rb[0], nr*sizeof(T))
        onGpu(nr):
          for i in gpuRange(nr): dest[FlowSite(rd[i])] := recv[i]

  proc flowPath(h, hl, e: auto; path: auto): auto =
    ## Multiply links along a path from site e, using halo neighbor tables.
    ## Directions are signed and one-based: +d uses U_d(x), -d uses
    ## U_d(x-d).adj. Initialize from the first link to avoid multiplying by
    ## the identity. All callers supply nonempty, fixed-size path arrays.
    var x = e
    let first = abs(path[0])-1
    if path[0] < 0: x = hl.nbrBck(first, x)
    var product = load1(h[first][x])
    if path[0] < 0: product := h[first][x].adj
    else: x = hl.nbrFwd(first, x)
    for j in 1..<path.len:
      let step = path[j]
      let d = abs(step)-1
      if step < 0:
        x = hl.nbrBck(d, x)
        let t = product*h[d][x].adj
        product := t
      else:
        let t = product*h[d][x]
        product := t
        if j < path.len-1: x = hl.nbrFwd(d, x)
    product

  proc toGpu*(x: FlowSum): auto =
    var g: array[4, GpuSumObj[float]]
    for i in 0..<4: g[i] = toGpu(x[i])
    g
  template getGpu*(x: FlowSum; g: array[4, GpuSumObj[float]]): auto = g
  template fromGpu*(x: FlowSum; g: array[4, GpuSumObj[float]]) = discard
  proc value(x: FlowSum): array[4, float] =
    for i in 0..<4: result[i] = x[i].value

  proc flowSum(ns: int): FlowSum =
    ## Reuse reducers by local volume. QEX's pinned result buffers are retained
    ## for the process lifetime, avoiding allocation on every measurement.
    var sums {.global.}: Table[int, FlowSum]
    if ns notin sums:
      var gs: FlowSum
      for i in 0..<4: gs[i] = newGpuSum[float](ns)
      sums[ns] = gs
    sums[ns]

  proc flowReduce(gs, sums: auto) =
    ## Reduce thread-local sums. The header supplies HIP's cooperative groups
    ## to QEX's reduction implementation without modifying the backend.
    const Backend {.strdefine.} = "CPU"
    when Backend == "HIP":
      proc hipGroups() {.header: "<hip/hip_cooperative_groups.h>", importcpp: "@".}
      hipGroups()
    for i in 0..<4: gs[i].reduce sums[i]

  proc flowShift(dest, src: auto; sh: seq[int]) =
    ## Set dest(x) = src(x-sh), following gaugeUtils.PathPlan/gaugeProd.
    ## Cache the gather by layout and displacement, including device indices.
    let lo = src.l
    # Keep shift indices resident too, as QEX does for halo neighbor tables.
    var maps {.global.}: Table[(pointer, seq[int]), tuple[lo: type(lo), gm: FlowGather]]
    let key = (cast[pointer](lo), sh)
    if key notin maps:
      var rl = newSeq[RecvList](lo.nSites)
      var x = newSeq[int32](lo.nDim)
      for i in 0..<lo.nSites:
        lo.coord(x, i)
        for d in 0..<sh.len:
          x[d] = int32((int(x[d])-sh[d]+lo.physGeom[d]) mod lo.physGeom[d])
        let ri = lo.rankIndex(x)
        rl[i] = RecvList(didx: int32(i), srank: int32(ri.rank), sidx: int32(ri.index))
      maps[key] = (lo, newFlowGather(getDefaultComm().makeGatherMap(rl)))
    flowGather(dest, src, maps[key].gm)

  proc flowWilsonLine(g: auto; line: seq[int]): auto =
    ## Evaluate a normalized Wilson line using QEX's optimal product plan.
    ## Shared subpaths avoid a multiply per link for long Polyakov lines;
    ## reference counts recycle intermediate fields after their last use.
    ## A small workspace pool per layout avoids repeated host/device allocation
    ## during measurements. Like shift maps, it lives for the process lifetime.
    let plan = [line].optimalPairs.plan(false)
    var products = initTable[seq[int], type(g[0])]()
    var uses = initCountTable[seq[int]]()
    var spare {.global.}: Table[pointer, seq[type(g[0])]]
    let layoutKey = cast[pointer](g[0].l)
    if layoutKey notin spare: spare[layoutKey] = @[]
    template workspace(): untyped =
      (if spare[layoutKey].len > 0: spare[layoutKey].pop() else: g[0].newOneOf())
    for s in plan.steps:
      uses.inc s.l
      uses.inc s.r
    for o in plan.outs: uses.inc o.key
    template node(k: untyped): untyped =
      (if k.len == 1: g[k[0]-1] else: products[k])
    template consumed(k: untyped) =
      if k.len > 1:
        uses.inc(k, -1)
        if uses[k] == 0:
          spare[layoutKey].add products[k]
          products.del k
    let ns = g[0].l.nSites
    for s in plan.steps:
      let l = node(s.l)
      var r = node(s.r)
      let shifted = s.sh.anyIt(it != 0)
      if shifted:
        r = workspace()
        flowResident(r)
        flowShift(r, node(s.r), s.sh)
      let dest = workspace()
      flowResident(dest)
      let la = s.la
      let ra = s.ra
      onGpu(ns):
        for e in gpuSites(dest):
          var a = load1(l[e])
          var b = load1(r[e])
          if la: a := l[e].adj
          if ra: b := r[e].adj
          dest[e] := a*b
      if shifted: spare[layoutKey].add r
      products[s.key] = dest
      consumed(s.l)
      consumed(s.r)
    let output = node(plan.outs[0].key)
    let gs = flowSum(ns)
    onGpu(ns):
      var sums: array[4, float]
      for e in gpuSites(output):
        let z = trace(output[e])
        sums[0] += z.re.simdSum
        sums[1] += z.im.simdSum
      gs.flowReduce sums
    var sums = gs.value
    rankSum sums
    consumed(plan.outs[0].key)
    if plan.outs[0].adj: sums[1] = -sums[1]
    newComplex(sums[0], sums[1]) / float(g[0][0].nrows*g[0].l.physVol)

  proc flowForce(f0, h0, hl: auto; cp, cr: float) =
    ## Sum the two plaquette and six rectangle staples in each (mu, nu) plane.
    ## Paths run from x to x+mu; project U_mu(x)*staple.adj onto the traceless
    ## anti-Hermitian algebra. Coefficients include QEX's 1/Nc normalization.
    var f: type(f0)
    var h: type(h0)
    shallowCopy(f, f0)
    shallowCopy(h, h0)
    let nd = h.len
    let plaqCoeff = cp
    let rectCoeff = cr
    onGpu(hl.lo.nSites):
      for e in gpuSites(h[0].field):
        for mu in 0..<nd:
          let m = mu+1
          var staple = load1(h[mu][e])
          staple := 0
          for nu in 0..<nd:
            if nu == mu: continue
            for n in [nu+1, -nu-1]:
              staple += plaqCoeff*flowPath(h, hl, e, [n,m,-n])
              if rectCoeff != 0.0:
                for path in [[n,n,m,-n,-n], [n,m,m,-n,-m], [-m,n,m,m,-n]]:
                  staple += rectCoeff*flowPath(h, hl, e, path)
          let t = h[mu][e]*staple.adj
          f[mu][e].projectTAH t

  proc flowZeuthen(h0, hf0, hl, z0: auto) =
    ## Apply the same 1/24 covariant second difference as the CPU Zeuthen
    ## correction. Read the original force from hf and write separately to z.
    var h, hf: type(h0)
    var z: type(z0)
    shallowCopy(h, h0)
    shallowCopy(hf, hf0)
    shallowCopy(z, z0)
    let nd = h.len
    onGpu(hl.lo.nSites):
      for e in gpuSites(h[0].field):
        for mu in 0..<nd:
          let ep = hl.nbrFwd(mu, e)
          let em = hl.nbrBck(mu, e)
          z[mu][e] := hf[mu][e] + (1.0/24.0)*(h[mu][e]*hf[mu][ep] +
            h[mu][em].adj*hf[mu][em] - 2.0*hf[mu][e])

#[ constructors ]#

converter strToGradientFlowKind*(str: string): GradientFlowKind =
  case str: # I'm only dealing with these flows at the moment
    of "wilsonflow", "WilsonFlow", "wilson", "Wilson": WilsonFlow
    of "symanzikflow", "SymanzikFlow", "symanzik", "Symanzik": SymanzikFlow
    of "zeuthenflow", "ZeuthenFlow", "zeuthen", "Zeuthen": ZeuthenFlow
    else:
      qexError "unsupported flow kind: " & str
      WilsonFlow

converter strToMeasurementKind*(str: string): MeasurementKind =
  case str:
    of "plaquette", "Plaquette", "plaq", "Plaq": Plaquette
    of "clover", "Clover", "clov", "Clov": Clover
    of "rectangle", "Rectangle", "rect", "Rect": Rectangle
    of "topology", "Topology", "topo", "Topo": Topology
    of "polyakov", "Polyakov", "poly", "Poly": Polyakov
    else:
      qexError "unsupported measurement kind: " & str
      Plaquette

proc newLaplacian[U](u: openArray[U]): auto =
  type U0 = evalType(u[0][0])
  result = Laplacian[U,U0](
    sf: u.newTransporters(u[0], 1),
    sb: u.newTransporters(u[0], -1)
  )

proc newGradientFlow*[U](
  u: seq[U]; 
  kind: GradientFlowKind;
  plaq: float = 1.0;
  rect: float = CrSymanzik;
  meas: seq[MeasurementKind] = @[
    Plaquette,
    Rectangle, 
    Clover, 
    Topology, 
    Polyakov
  ];
  log: bool = true
): auto =
  type U0 = evalType(u[0][0])
  result = GradientFlow[U,U0](kind: kind, meas: meas, u: u.newOneOf(), log: log)
  result.gc = case result.kind
    of WilsonFlow: GaugeActionCoeffs(plaq: plaq)
    of RectangleFlow: gaugeActRect(plaq, rect)
    of IwasakiFlow: Iwasaki(plaq)
    of DBW2Flow: DBW2(plaq)
    of SymanzikFlow: Symanzik(plaq)
    of ZeuthenFlow: 
      result.lap = u.newLaplacian()
      Symanzik(plaq)

#[ main gradient flow templates/procedures ]#

template ioBlock(flow: var GradientFlow; filename: string; body: untyped): untyped =
  if flow.log:
    if flow.logFile.isNil: flow.logFile = filename.open(fmWrite)
    else: qexError "log file already opened"
    body
    if not flow.logFile.isNil: flow.logFile.close()

template gradient[U](flow: var GradientFlow; f: var seq[U]; u: seq[U]; force: untyped) =
  let nd = u[0].l.nDim
  force
  case flow.kind:
    of ZeuthenFlow: # arXiv:1411.6706; Eqn. (6.2)
      flow.lap.sf.setLinks(u)
      flow.lap.sb.setLinks(u)
      let coeff = -0.5*CrSymanzik
      var
        sf = flow.lap.sf
        sb = flow.lap.sb
      var ft = f.newOneOf()
      threads:
        for mu in 0..<nd:
          ft[mu] := f[mu]
        threadBarrier()
        for mu in 0..<nd:
          f[mu] += coeff*(sf[mu]^*ft[mu] + sb[mu]^*ft[mu] - 2.0*ft[mu])
    else: discard

when backendIsGpu:
  proc flowObservables(h0, hl: auto; meas: seq[MeasurementKind]): array[MeasurementKind, float] =
    ## Measure all requested local observables in one GPU pass, reusing each
    ## clover tensor for energy and topology. Normalizations match gaugeUtils:
    ## mean loops, 1-E/Nc for Clover, and topoQ for Topology (four dimensions).
    var h: type(h0)
    shallowCopy(h, h0)
    let nd = h.len
    let ns = hl.lo.nSites
    let wantP = Plaquette in meas
    let wantR = Rectangle in meas
    let wantF = Clover in meas or Topology in meas
    let wantQ = Topology in meas
    let gs = flowSum(ns)
    onGpu(ns):
      var sums: array[4, float]
      for e in gpuSites(h[0].field):
        var f: array[6, type(load1(h[0][e]))]
        for mu in 1..<nd:
          for nu in 0..<mu:
            let m = mu+1
            let n = nu+1
            if wantP:
              # Same contraction as backend/examples/behalo1.testPlaq.
              let a = h[mu][e]*h[nu][hl.nbrFwd(mu, e)]
              let b = h[nu][e]*h[mu][hl.nbrFwd(nu, e)]
              sums[0] += redot(a, b).simdSum
            if wantR:
              sums[1] += trace(flowPath(h, hl, e, [m,m,n,-m,-m,-n])).re.simdSum
              sums[1] += trace(flowPath(h, hl, e, [m,n,n,-m,-n,-n])).re.simdSum
            if wantF:
              # gaugeUtils.fmunu(loop=1): four corners, averaged and projected.
              var t = flowPath(h, hl, e, [-m,-n,m,n])
              t += flowPath(h, hl, e, [-n,m,n,-m])
              t += flowPath(h, hl, e, [m,n,-m,-n])
              t += flowPath(h, hl, e, [n,-m,-n,m])
              let k = mu*(mu-1) div 2 + nu
              f[k].projectTAH(0.25*t)
              sums[2] += norm2(f[k]).simdSum
        if wantQ:
          # gaugeUtils.topoQ, using F.adj = -F.
          sums[3] += (redot(f[0],f[5])-redot(f[1],f[4])+redot(f[2],f[3])).simdSum
      gs.flowReduce sums
    var sums = gs.value
    rankSum sums
    let nc = h[0].field[0].nrows.float
    let vol = hl.lo.physVol.float
    result[Plaquette] = sums[0]/(vol*nc*float(nd*(nd-1) div 2))
    result[Rectangle] = sums[1]/(vol*nc*float(nd*(nd-1)))
    result[Clover] = 1.0-sums[2]/(vol*nc)
    result[Topology] = sums[3]/(4.0*PI*PI)

proc flowObservables[U](u: seq[U]; meas: seq[MeasurementKind]): array[MeasurementKind, float] =
  ## CPU measurements using QEX's existing gauge routines and normalizations.
  let nd = u[0].l.nDim.float
  let nc = u[0][0].nrows.float
  if Plaquette in meas:
    let plq = u.plaq
    let nl = plq.len div 2
    result[Plaquette] = plq[0..<nl].sum + plq[nl..^1].sum
  if Rectangle in meas:
    let rc = GaugeActionCoeffs(rect: -1.0/u[0].l.physVol.float/nd/(nd-1.0))
    result[Rectangle] = rc.gaugeAction1(u)
  if Clover in meas or Topology in meas:
    let f = u.fmunu(1)
    if Clover in meas:
      let (es, et) = f.densityE()
      result[Clover] = (nc-es-et)/nc
    if Topology in meas: result[Topology] = f.topoQ()

template flowMeasurements(flow, u: untyped; device: static bool; values: untyped) =
  ## Format measurements in the requested order, with spatial/temporal complex
  ## Polyakov lines occupying four columns. Both backends share the log format.
  let observables = values
  let nd = u[0].l.nDim
  var output = @["FLOW " & flow.flowTime.formatFloat(ffDecimal, 3)]
  for meas in flow.meas:
    if meas == Polyakov:
      let geom = u[0].l.physGeom
      var poly = newSeq[type(u.wline @[1])](nd)
      for d in 0..<nd:
        when device: poly[d] = flowWilsonLine(u, repeat(d+1, geom[d]))
        else: poly[d] = u.wline repeat(d+1, geom[d])
      let spatial = poly[0..<nd-1].sum / float(nd-1)
      let temporal = poly[^1]
      for x in [spatial.re, spatial.im, temporal.re, temporal.im]:
        output.add x.formatFloat(ffDecimal, 18)
    else:
      output.add observables[meas].formatFloat(ffDecimal, 18)
  let line = output.join(" ") & "\n"
  if flow.log: flow.logFile.write(line)
  echo line.replace("FLOW", $flow.kind)

template measurements*[U,U0](flow: var GradientFlow[U,U0]; u: seq[U]): untyped =
  ## Measure a host-resident gauge field and write one FLOW record.
  flow.flowMeasurements(u, false, flowObservables(u, flow.meas))

proc gradientFlow*[U,U0](flow: var GradientFlow[U,U0]; steps: int; eps: float) =
  ## Advance flow.u by `steps` steps of size `eps`, measuring after each step.
  ## `eps` and `flowTime` are in lattice units (t/a^2). GPU builds upload links
  ## once and download them on return; forces and RK work fields stay resident.
  ##
  ## Originally written by James Osborn & Xiaoyong Jin.
  ## d/dt Vt = Z(Vt) Vt
  ## Runge-Kutta:
  ## W0 <- Vt
  ## W1 <- exp(1/4 Z0) W0
  ## W2 <- exp(8/9 Z1 - 17/36 Z0) W1
  ## V(t+eps) <- exp(3/4 Z2 - 8/9 Z1 + 17/36 Z0) W2
  ## where Zi = eps Z(Wi)
  let 
    nc = flow.u[0][0].nrows.float
    nd = flow.u[0].l.nDim
  let epsnc = eps * nc  # compensate force normalization
  var
    p = flow.u[0].l.newGauge  # mom
    f = flow.u[0].l.newGauge  # force
  var n = 0
  var u = flow.u.newOneOf()
  let saved = flow.u
  
  # gradient flow evolution and measurements
  threads:
    for mu in 0..<u.len: u[mu] := saved[mu]
  when backendIsGpu:
    let lo = u[0].l
    let reach = if flow.gc.rect != 0.0 or Rectangle in flow.meas: 2 else: 1
    # All local force/measurement paths lie in one coordinate plane.
    var offsets: seq[seq[int32]]
    for mu in 1..<nd:
      for nu in 0..<mu:
        for a in -reach..reach:
          for b in -reach..reach:
            var x = newSeq[int32](nd)
            x[mu] = int32(a)
            x[nu] = int32(b)
            if x notin offsets: offsets.add x
    let hl = flowHaloLayout(lo, reach)
    let hm = hl.makeHaloMap(getDefaultComm(), offsets)
    let gm = newFlowGather(hm.gather, lo.nSites)
    var h = u.mapIt(makeHalo(hl, it))
    var hf: type(h)
    var z: type(f)
    if flow.kind == ZeuthenFlow:
      hf = f.mapIt(makeHalo(hl, it))
      z = f.newOneOf()
    flowResident(u, upload = true)
    flowResident(p)
    flowResident(f)
    flowResident(h)
    flowResident(hf)
    flowResident(z)
    defer:
      flowFree(h)
      flowFree(hf)
      flowFree(u)
      flowFree(p)
      flowFree(f)
      flowFree(z)
      hl.freeGpuMem()
      free(gm)
  template advance(a: static float; b: float; save: static bool) =
    when backendIsGpu:
      for mu in 0..<nd: flowGather(h[mu], u[mu], gm)
      flowForce(f, h, hl, flow.gc.plaq/nc, flow.gc.rect/nc)
      if flow.kind == ZeuthenFlow:
        for mu in 0..<nd: flowGather(hf[mu], f[mu], gm)
        flowZeuthen(h, hf, hl, z)
        flowAdvance(u, p, z, a, b, save)
      else: flowAdvance(u, p, f, a, b, save)
    else:
      flow.gradient(f, u):
        flow.gc.gaugeForce(u, f)
      flowAdvance(u, p, f, a, b, save)
  while true: # arXiv:1006.4518
    advance(0.0, (-1.0/4.0)*epsnc, true)
    advance(-17.0/9.0, (-8.0/9.0)*epsnc, true)
    advance(-1.0, (-3.0/4.0)*epsnc, false)
    inc n
    flow.step += 1
    flow.flowTime += eps
    when backendIsGpu:
      for mu in 0..<nd: flowGather(h[mu], u[mu], gm)
      flow.flowMeasurements(u, true, flowObservables(h, hl, flow.meas))
    else:
      flow.measurements(u)
    if n > (steps - 1): 
      when backendIsGpu:
        for mu in 0..<nd:
          u[mu].gpuFlagsIncl {gmCpuRead}
          fromGpu(u[mu])
      threads:
        for mu in 0..<u.len: saved[mu] := u[mu]
      break

when isMainModule:
  echo banner

  proc getMeasurements*(self: JsonNode): seq[MeasurementKind] =
    result = newSeq[MeasurementKind]()
    if self.hasKey("measurements"):
      for m in self["measurements"].items: 
        result.add m.getStr().strToMeasurementKind()
    else: qexError "no measurements specified for flow"
  
  qexInit()

  # command line information
  let 
    cmd = readCMD()
    flowInfo = case cmd.hasKey("json")
      of true: readJSON(cmd["json"].getStr())
      of false:
        echo "using default lattice parameters"
        defaultInputs
    cfg = case cmd.hasKey("configuration")
      of true: $cmd["configuration"].getInt()
      of false:
        echo "no configuration specified, using default"
        "0"
    filename = case cmd.hasKey("base-filename")
      of true: cmd["base-filename"].getStr()
      of false: "__REGRESSIONMODE__"
    outputDir = case cmd.hasKey("output-dir")
      of true: cmd["output-dir"].getStr()
      of false: ""
    latLayout = case flowInfo.hasKey("lattice-geometry")
      of true: flowInfo["lattice-geometry"].getIntSeq()
      of false: 
        echo "using default lattice geometry"
        defaultLat
    lo = case flowInfo.hasKey("rank-geometry")
      of true:
        let rnkLayout = flowInfo["rank-geometry"].getIntSeq()
        case flowInfo.hasKey("simd-geometry"):
          of true:
            let vecLayout = flowInfo["simd-geometry"].getIntSeq()
            newLayout(latLayout, VLEN, rnkLayout, vecLayout)
          of false: newLayout(latLayout, rnkLayout)
      of false: newLayout(latLayout)
  var u = lo.newGauge()

  # echo information
  echo "command line info: ", $cmd
  echo "gradient flow info: ", $flowInfo

  # read in gauge field
  if filename != "__REGRESSIONMODE__":
    u.readGauge(filename & "_" & cfg & ".lat")
  else: alphasrng.random(u) # for testing/regression

  # cycle through flows
  for flow in flowInfo["flows"].pairs:
    let 
      flowKind = flow.key.strToGradientFlowKind()
      measKinds = flow.val.getMeasurements()
      stepSizes = flow.val["step-sizes"].getFloatSeq()
      maxFlowTimes = flow.val["step-size-transition-flow-times"].getFloatSeq()
      transitions = maxFlowTimes.len
    let outputFilename = outputDir & flow.key & "_" & $cfg & ".log"
    var flowObject = u.newGradientFlow(flowKind, meas = measKinds)

    # run flow for specified step sizes
    if stepSizes.len != transitions:
      qexError "number of step sizes must match number of transition flow times"
    flowObject.ioBlock(outputFilename):
      threads:
        for mu in 0..<u.len: flowObject.u[mu] := u[mu]
      flowObject.measurements(u) # t/a^2 = 0.0 measurement
      for transition in 0..<transitions:
        let 
          eps = stepSizes[transition]
          steps = round((maxFlowTimes[transition] - flowObject.flowTime)/eps)
        flowObject.gradientFlow(steps.int, eps)

  qexFinalize()
