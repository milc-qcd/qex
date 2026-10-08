## Fermion flow of fermion bilinears
## 
## Fermion flow of fermion bilinears can be done using a noisy estimator for the
## fermion bilinears. Taking O to be some operator, the fermion flowed bilinear
## if given by
## (1) <chi-bar(x,t) O chi_t(x,t)> = N^{-1} 
##     x sum_{i = 1}^{N} eta_i^{dagger}(x,t) O propagator_i(x,t)
## where each eta_i is a random source vector and propagator_i is its corresponding
## propagator. 
## 
## Based on:
## [1] JHEP 04 (2013) 123
## [2] PoS INPC2016 (2017) 342
## 
## Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

import qex

import hdf5

import std/[math]
import std/[os]
import std/[sequtils]
import std/[times]

import physics/[stagSolve]
import gauge/[hisqsmear]

type StaggeredLaplacian[U] = object
  u: seq[U]

letParam:
  lattice = @[8, 8, 8, 16]

  gaugeConfigFilename = ""
  gaugeFlowFilename = "gauge.h5"
  fermionFlowFilename = "fermion.h5"

  mass = 0.01

  conjugateGradientTolerance = 1e-20
  conjugateGradientMaxIterations = 10000

  numSources = 10

  stepSizes = @[0.01, 0.02, 0.05]
  maxFlowTimes = @[5.0, 10.0, pow(0.5*lattice[0], 2)/8.0]

  seed: uint64 = int(1000*epochTime())

proc newStaggeredLaplacian[U](u: seq[U]): StaggeredLaplacian[U] = 
  return StaggeredLaplacian[U](u: u)

template rephase[U](u: seq[U]): untyped =
  threads:
    u.setBC()
    threadBarrier()
    u.stagPhase()

proc force[G, F](lap: StaggeredLaplacian[G]; f, p: F) =
  let lo = lap.u[0].l
  let stag = lap.u.newStag()
  var tmpa = lo.ColorVector()
  var tmpb = lo.ColorVector()
  lap.u.rephase()
  stag.D(tmpa, p, 0.0)
  stag.Ddag(tmpb, tmpa, 0.0)
  threads: f := -tmpb
  lap.u.rephase()

template runFermionFlow[G, F](u: seq[G]; s, p: seq[F]; measure: untyped): untyped =
  block:
    assert s.len == numSources
    assert p.len == numSources

    let lo = lattice.newLayout()
    let nc = u[0][0].nrows.float
    var v {.inject.} = u.newOneOf()
    var (fg, pg) = (u.newOneOf(), u.newOneOf())
    var (ps, pp) = (newSeq[F](s.len), newSeq[F](s.len))
    var (fs, fp) = (s[0].newOneOf(), s[0].newOneOf())
    var lap = v.newStaggeredLaplacian()
    var (step, phase) = (0, 0)
    var t = 0.0

    threads:
      for mu in 0..<lo.nDim: v[mu] := u[mu]

    for f in 0..<numSources:
      ps[f] = lo.ColorVector()
      pp[f] = lo.ColorVector()

    while true:
      let flowTime {.inject.} = t

      block: measure
      if phase >= stepSizes.len: break
      if flowTime > maxFlowTimes[^1]: break

      let epsG = stepSizes[phase]
      let epsF = nc * stepSizes[phase]

      fg.gaugeForce v
      threads:
        for mu in 0..<fg.len:
          for e in v[mu]:
            var vv {.noinit.}: type(load1(fg[0][0]))
            vv := (-1.0/4.0)*epsG*fg[mu][e]
            let tv = exp(vv)*v[mu][e]
            pg[mu][e] := vv
            v[mu][e] := tv
      
      for f in 0..<s.len:
        lap.force(fs, s[f])
        lap.force(fp, p[f])
        threads:
          for e in fs:
            ps[f][e] := (1.0/4.0)*epsF*fs[e]
            pp[f][e] := (1.0/4.0)*epsF*fp[e]
            s[f][e] += ps[f][e]
            p[f][e] += pp[f][e]

      fg.gaugeForce v
      threads:
        for mu in 0..<fg.len:
          for e in v[mu]:
            var vv {.noinit.}: type(load1(fg[0][0]))
            vv := (-8.0/9.0)*epsG*fg[mu][e] + (-17.0/9.0)*pg[mu][e]
            let tv = exp(vv)*v[mu][e]
            pg[mu][e] := vv
            v[mu][e] := tv

      for f in 0..<numSources:
        lap.force(fs, s[f])
        lap.force(fp, p[f])
        threads:
          for e in fs:
            ps[f][e] := (8.0/9.0)*epsF*fs[e] - (17.0/9.0)*ps[f][e]
            pp[f][e] := (8.0/9.0)*epsF*fp[e] - (17.0/9.0)*pp[f][e]
            s[f][e] += ps[f][e]
            p[f][e] += pp[f][e]
      
      fg.gaugeForce v
      threads:
        for mu in 0..<fg.len:
          for e in v[mu]:
            var vv {.noinit.}: type(load1(fg[0][0]))
            vv := (-3.0/4.0)*epsG*fg[mu][e] - pg[mu][e]
            let tv = exp(vv)*v[mu][e]
            v[mu][e] := tv
    
      for f in 0..<numSources:
        lap.force(fs, s[f])
        lap.force(fp, p[f])
        threads:
          for e in fs:
            ps[f][e] := (3.0/4.0)*epsF*fs[e] - ps[f][e]
            pp[f][e] := (3.0/4.0)*epsF*fp[e] - pp[f][e]
            s[f][e] += ps[f][e]
            p[f][e] += pp[f][e]

      inc step
      t += stepSizes[phase]
      if round(t, 2) == round(maxFlowTimes[phase], 2): inc phase

when isMainModule:
  qexInit()

  let lo = lattice.newLayout()
  type F = typeof(lo.ColorVector())
  type G = typeof(lo.newGauge())
  var (s, p) = (newSeq[F](numSources), newSeq[F](numSources))
  var r = lo.newRNGField(RngMilc6, seed)
  var (u, su, sul) = (lo.newGauge(), lo.newGauge(), lo.newGauge())

  if gaugeConfigFilename.len != 0:
    if fileExists(gaugeConfigFilename):
      if u.loadGauge(gaugeConfigFilename) != 0: 
        qexError "unable to read " & gaugeConfigFilename
    else: qexError gaugeConfigFilename & " does not exist"
  else: u.random(r)

  let hisq = newHisq(0.0, 1.0)
  u.rephase()
  discard hisq.smearGetForce(u, su, sul)
  u.rephase()

  let stag = newStag3(su, sul)

  var solverParams = initSolverParams()
  solverParams.r2req = conjugateGradientTolerance
  solverParams.maxits = conjugateGradientMaxIterations

  for f in 0..<numSources:
    s[f] = lo.ColorVector()
    p[f] = lo.ColorVector()
    threads: s[f].z4(r)
    stag.solve(p[f], s[f], mass, solverParams)
        
  var gaugeLog, fermionLog: H5File
  if myRank == 0:
    gaugeLog = newH5File(gaugeFlowFilename, "rw")
    fermionLog = newH5File(fermionFlowFilename, "rw")

  # gauge measurement storage
  var flowTimes: seq[float]
  var plaqs, plaqt: seq[float]
  var rect: seq[float]
  var clovs, clovt: seq[float]
  var topo: seq[float]
  var repolys, impolys: seq[float]
  var repolyt, impolyt: seq[float]

  # fermion measurement storage
  var chiralCondensates = newSeq[seq[float]](numSources)
  var fermionActions = newSeq[seq[float]](numSources)

  for f in 0..<numSources:
    (chiralCondensates[f], fermionActions[f]) = (newSeq[float](), newSeq[float]())

  u.runFermionFlow(s, p):
    var (gaugeOutput, fermionOutput) = ("", "")
    let nd = lo.nDim 
    let fmunu = v.fmunu(1)

    flowTimes.add flowTime

    # plaquette
    let
      plq = v.plaq
      nl = plq.len div 2
    plaqs.add plq[0..<nl].sum
    plaqt.add plq[nl..^1].sum

    # rectangle
    let
      frct = 1.0/lo.physVol.float/nd.float/(nd.float - 1.0)
      rc = GaugeActionCoeffs(plaq: 0.0, rect: -frct)
    rect.add rc.gaugeAction1(v)
    
    # clover
    let (es, et) = fmunu.densityE()
    clovs.add es
    clovt.add et

    # topological charge
    topo.add fmunu.topoQ()

    # Polyakov loop
    type P = typeOf(v.wline @[1])
    var poly = newSeq[P](lo.nDim)
    for mu in 0..<nd: poly[mu] = v.wline repeat(mu + 1, lo[mu])
    let (pls, plt) = (poly[0..^(nd-2)].sum() / (nd.float - 1.0), poly[^1])
    repolys.add pls.re
    impolys.add pls.im
    repolyt.add plt.re
    impolyt.add plt.im

    # chiral condensate & kinetic action
    v.rephase()
    discard hisq.smearGetForce(v, su, sul)
    v.rephase()
    for f in 0..<numSources:
      var (pbp, kin) = (0.0, 0.0)

      # chiral condensate
      threads:
        var tpbp = 0.0
        for e in s[f]: tpbp += simdSum(redot(s[f][e], p[f][e]))
        threadBarrier()
        threadSum(tpbp)
        threadBarrier()
        threadMaster: pbp = tpbp
      rankSum(pbp)
      chiralCondensates[f].add pbp / lo.physVol.float

      # kinetic action
      var tmp = lo.ColorVector()
      stag.D(tmp, p[f], mass)
      threads:
        var tkin = 0.0
        for e in s[f]:
          tkin += 2.0*simdSum(redot(s[f][e], tmp[e]))
        threadBarrier()
        threadSum(tkin)
        threadBarrier()
        threadMaster: kin = tkin
      rankSum(kin)
      fermionActions[f].add kin / lo.physVol.float
    
    # echo gauge measurements for logging purposes
    echo "FLOW: ", flowTime, " plaq: ", plq.sum, " rect: ", rect[^1], " clov: ", es + et, " topo: ", topo[^1], " poly: ", pls, " polyt: ", plt
    
  if myRank == 0:
    # gauge flow measurements
    gaugeLog["/flow-times"]          = flowTimes
    gaugeLog["/plaquette/spatial"]  = plaqs
    gaugeLog["/plaquette/temporal"] = plaqt
    gaugeLog["/rectangle"]          = rect
    gaugeLog["/clover/spatial"]     = clovs
    gaugeLog["/clover/temporal"]    = clovt
    gaugeLog["/topology"]           = topo
    gaugeLog["/polyakov-loop/spatial/re"]  = repolys
    gaugeLog["/polyakov-loop/spatial/im"]  = impolys
    gaugeLog["/polyakov-loop/temporal/re"] = repolyt
    gaugeLog["/polyakov-loop/temporal/im"] = impolyt

    # fermion flow measurements
    fermionLog["/flow-times"] = flowTimes
    for f in 0..<numSources:
      fermionLog["/chiral-condensate/source-" & $f] = chiralCondensates[f]
      fermionLog["/fermion-action/source-" & $f] = fermionActions[f]

  if myRank == 0:
    discard gaugeLog.close()
    discard fermionLog.close()

  qexFinalize()
