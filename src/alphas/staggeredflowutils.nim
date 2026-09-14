import qex
import gauge
import gauge/[gaugeUtils, hisqsmear]
import physics/[stagSolve]
import json, sequtils, strutils

proc rephase[U](u: seq[U]) =
  threads:
    u.setBC()
    threadBarrier()
    u.stagPhase()

proc newLaplacian4D[U, F](lap: var auto, u: U, f: F): auto =
  lap.sf.setlinks(u)
  lap.sb.setlinks(u)
  var r = newOneOf(f)
  r := lap.sf[0]^*f + lap.sb[0]^*f + lap.sf[1]^*f + lap.sb[1]^*f + lap.sf[
      2]^*f + lap.sb[2]^*f + lap.sf[3]^*f + lap.sb[3]^*f - 8.0*f
  return r


proc get*(info: JsonNode; key: string): seq[float] =
  result = newSeq[float]()
  for el in info[key].getElems(): result.add getFloat(el)

proc formatMeasurements*(
    measurements: JsonNode;
    style: string = "default"
  ): string =
  result = ""
  let tau = measurements["flow-time"].getFloat()
  case style:
    of "default":
      var concatenation = @["FLOW", tau.formatFloat(ffDecimal, 3)]
      result = concatenation.join(" ")
      for key, measurement in measurements:
        let O = measurement.getFloat()
        concatenation = @[result, O.formatFloat(ffDecimal, 13)]
        result = concatenation.join(" ")
    of "KS_nHYP_FA": # Style of https://github.com/daschaich/KS_nHYP_FA
      let
        standIn = %* {"stand-in": 0.0}
        observables = @[
          measurements["plaquette"],
          measurements["clover"],
          measurements["t2E-clover"],
          standIn["stand-in"], # i.e., nothing; where rectangle could be
          measurements["t2E-plaquette"],
          measurements["topological-charge"],
          measurements["t2E-spacelike-clover"],
          measurements["t2E-timelike-clover"],
          measurements["Re(timelike-Polyakov-loop)"],
          measurements["Im(timelike-Polyakov-loop)"],
          measurements["Re(spacelike-Polyakov-loop)"],
          measurements["Im(spacelike-Polyakov-loop)"]
        ]
      var concatenation = @["FLOW", tau.formatFloat(ffDecimal, 2)]
      result = concatenation.join(" ")
      for observable in observables:
        let O = observable.getFloat()
        concatenation = @[result, O.formatFloat(ffDecimal, 13)]
        result = concatenation.join(" ")
    else: discard

proc flowMeasurements*(u: auto; loop: int; tau: float): JsonNode =
  var
    pls, plt: ComplexProxy[ComplexObj[float64, float64]]
    poly: seq[ComplexProxy[ComplexObj[float64, float64]]]
    t2Ess, t2Est, t2Ees, t2Eet: float
  let
    f = u.fmunu(loop)
    (es, et) = f.densityE
    q = f.topoQ
    pl = u.plaq
    nl = pl.len div 2
    ss = 6.0*pl[0..<nl].sum
    st = 6.0*pl[nl..^1].sum
    pg = u[0].l.physGeom
  poly = newSeq[ComplexProxy[ComplexObj[float64, float64]]](pg.len)
  for i in 0..<pg.len: poly[i] = u.wline repeat(i+1, pg[i])
  pls = poly[0..^2].sum/float(poly.len-1)
  plt = poly[^1]
  (t2Ess, t2Est) = (6.0*tau*tau*(3.0-ss), 6.0*tau*tau*(3.0-st))
  (t2Ees, t2Eet) = (tau*tau*es, tau*tau*et)
  result = %* {
    "flow-time": tau,
    "plaquette": 0.5*ss+0.5*st, #(1-Puv)
    "clover": es+et,
    "t2E-plaquette": t2Ess+t2Est,
    "t2E-clover": t2Ees+t2Eet,
    "t2E-spacelike-plaquette": t2Ess,
    "t2E-timelike-plaquette": t2Est,
    "t2E-spacelike-clover": t2Ees,
    "t2E-timelike-clover": t2Eet,
    "topological-charge": q,
    "Re(spacelike-Polyakov-loop)": 3.0*pls.re,
    "Im(spacelike-Polyakov-loop)": 3.0*pls.im,
    "Re(timelike-Polyakov-loop)": 3.0*plt.re,
    "Im(timelike-Polyakov-loop)": 3.0*plt.im,
  }

proc laplace4D[U, F](u: U; psi_in: F): F =
  let
    tf = u.newTransporters(psi_in, 1)  #forward
    tb = u.newTransporters(psi_in, -1) #backward
  var r = newOneOf(psi_in)
  threads:
    r := -8.0*psi_in
    for mu in 0..<u.len:
      r := tf[mu]^*psi_in + tb[mu]^*psi_in + r
  return r

proc hisqStag[U](u: U): auto =
  let lo = u[0].l
  let hisq = newHisq(0.0, 1.0)
  var su = lo.newGauge()
  var sul = lo.newGauge()
  discard hisq.smearGetForce(u, su, sul)
  return newStag3(su, sul)

proc stagLap4D[U, S, F](g: U; stag: S; psi_in: F; mass: float): F =
  let lo = psi_in.l
  var v1 = lo.ColorVector()
  var v2 = lo.ColorVector()
  var v3 = lo.ColorVector()
  g.rephase()
  stag.D(v1, psi_in, mass)
  stag.Ddag(v2, v1, mass)
  g.rephase()
  v3 := -v2
  return v3

template staggeredFlow(
    gc: GaugeActionCoeffs;
    g: auto; #array|seq;
    phi0: auto; #array|seq;
    steps: int;
    eps: float;
    M: seq[float];
    measure: untyped;
  ): untyped =
  #[ Gradient flow w/ Wilson or rectangle action
    Originally written by James Osborn & Xiaoyong Jin.
    d/dt Vt = Z(Vt) Vt
    Runge-Kutta:
    W0 <- Vt
    W1 <- exp(1/4 Z0) W0
    W2 <- exp(8/9 Z1 - 17/36 Z0) W1
    V(t+eps) <- exp(3/4 Z2 - 8/9 Z1 + 17/36 Z0) W2
    where
    Zi = eps Z(Wi)

  Fermion flow w/ Staggered fermion
  d/dt chi = Delta Vt chi
  Runge-Lutta: (arxiv:1302.5246, app.D4)
  phi0 = chi(t)
  phi1 = phi0 + 1/4 Delta0 phi0
  phi2 = phi0 + 8/9 Delta1 phi1 - 2/9 Delta0 phi0
  chi(t+eps) <- phi1 + 3/4 Delta2 phi2 = phi1 + 3/4 delta2 (8/9 d1p1 -2/9 d0p0)
  where
  Deltai = eps*laplace(Wi) (i=0,1,2)
  save d0p0, d1p1 and phi1 but not phi2 or d2p2
]#
  proc flowProc {.gensym.} =
    tic("flowProc")
    const nc = g[0][0].nrows.float
    const nci = g[0][0].nrows
    const nsrcTime = phi0[0].len
    const ncorners = phi0[0][0].len
    const nmasses = phi0[0][0][0].len
    let lo = g[0].l
    var
      p = g[0].l.newGauge # mom
      f = g[0].l.newGauge # force
      n = 1
    var
      d0p0, d1p1, phi1: array[nci, array[nsrcTime, array[ncorners, array[
          nmasses, typeof(lo.ColorVector)]]]] #array[5,typeof()] is a type
    echo "Initializing arrays"
    # threads:
      # Initialize the arrays with the same structure as phi0
    for color in 0..<nci:
      for srct in 0..<nsrcTime:
        for corner in 0..<ncorners:
          for mass in 0..<nmasses:
            d0p0[color][srct][corner][mass] = lo.ColorVector()
            d1p1[color][srct][corner][mass] = lo.ColorVector()
            phi1[color][srct][corner][mass] = lo.ColorVector()
    while true:
      let stag = newStag(g) # use hisqStag(g) for hisq
      echo "first step"
      g.echoPlaq
      let tau = n * eps
      let epsnc = eps * nc # compensate force normalization
      gc.gaugeForce(g, f)
      for color in 0..<nci:
        for srct in 0..<nsrcTime:
          for corner in 0..<ncorners:
            for mass in 0..<nmasses:
              d0p0[color][srct][corner][mass] := eps*stagLap4D(g, stag, phi0[
                  color][srct][corner][mass], M[mass]) #Delta0
              threads:
                phi1[color][srct][corner][mass] := phi0[color][srct][corner][
                    mass] +
                  0.25*d0p0[color][srct][corner][mass]
      threads:
        for mu in 0..<f.len:
          for e in g[mu]:
            var v {.noinit.}: type(load1(f[0][0]))
            v := (-1.0/4.0)*epsnc*f[mu][e] # 1/4 Z0
            let t = exp(v)*g[mu][e] # exp(1/4 Z0) w0
            p[mu][e] := v # 1/4 Z0
            g[mu][e] := t # W1
      #fermion flow step 1: d1p1 = eps*laplace4D[g,phi1] #Delta1
      for color in 0..<nci:
        for srct in 0..<nsrcTime:
          for corner in 0..<ncorners:
            for mass in 0..<nmasses:
              d1p1[color][srct][corner][mass] := eps*stagLap4D(g, stag, phi1[
                  color][srct][corner][mass], M[mass]) #Delta1
      gc.gaugeForce(g, f) # f is now -dS/dU at U = W1
      threads:
        for mu in 0..<f.len:
          for e in g[mu]:
            var v {.noinit.}: type(load1(f[0][0]))
            v := (-8.0/9.0)*epsnc*f[mu][e] + (-17.0/9.0)*p[mu][e] # 8/9 Z1 - 17/36 Z0
            let t = exp(v)*g[mu][e] # exp(8/9 Z1 - 17/36 Z0) W1
            p[mu][e] := v # 8/9 Z1 - 17/36 Z0
            g[mu][e] := t # W2
      #fermion flow step 2&3: d2p2 = epsnc*laplace4D[g,phi2] #Delta2
      for color in 0..<nci:
        for srct in 0..<nsrcTime:
          for corner in 0..<ncorners:
            for mass in 0..<nmasses:
              var phi2 = newOneOf(phi0[color][srct][corner][mass])
              phi2 := phi0[color][srct][corner][mass] + 8.0/9.0*d1p1[color][
                  srct][corner][mass] - 2.0/9.0*d0p0[color][srct][corner][mass]
              phi0[color][srct][corner][mass] := phi1[color][srct][corner][
                  mass] +
                3.0/4.0 * eps*stagLap4D(g, stag, phi2, M[mass]) #3.0/4.0 * d2p2
      gc.gaugeForce(g, f) # f is now -dS/dU at U = W2
      threads:
        for mu in 0..<f.len:
          for e in g[mu]:
            var v {.noinit.}: type(load1(f[0][0]))
            v := (-3.0/4.0)*epsnc*f[mu][e] - p[mu][e] #3/4 Z2 - 8/9 Z1 + 17/36 Z0
            let t = exp(v)*g[mu][e] # exp(3/4 Z2 - 8/9 Z1 + 17/36 Z0) W2
            g[mu][e] := t # V(t+eps)
      let wflowT {.inject.} = tau
      measure
      inc n
      if steps > 0 and n > steps: break
    toc("end")
  flowProc()

template staggeredFlow(
    gc: GaugeActionCoeffs;
    g: auto; 
    phi: auto;
    eps: float;
    M: seq[float];
    measure: untyped
  ): untyped =
  gc.staggeredFlow(g, phi, 0, eps, M):
    let flowTime {.inject.} = wflowT
    measure

template staggeredFlow*(u: auto; phi: auto; M: seq[float]; info: JsonNode;
    body: untyped) =
  var
    v = u[0].l.newGauge
    f {.inject, used.}: File
    tau {.inject.}: float
    measurements {.inject.}: JsonNode
  for flow, flowInfo in info:
    let
      loops = case info[flow].hasKey("loops")
        of true: info[flow]["loops"].getInt()
        of false: 1
      beta = case info[flow].hasKey("beta")
        of true: info[flow]["beta"].getFloat()
        of false: 1.0
      cr = case info[flow]["action"].getStr()
        of "Rectangle": info[flow]["cr"].getFloat()
        else: 0.0
      gc = case info[flow]["action"].getStr()
        of "Rectangle": gaugeActRect(beta, cr)
        else: GaugeActionCoeffs(plaq: beta)
      dts = info[flow].get("time-increments")
      maxFlts = info[flow].get("maximum-flow-times")
    var lastMaxFlt = 0.0
    threads: v := u
    v.reunit
    for flowTimeInfo in zip(dts, maxFlts):
      let (dtau, tmax) = flowTimeInfo
      gc.staggeredFlow(v, phi, dtau, M):
        tau = flowTime + lastMaxFlt
        measurements = v.flowMeasurements(loops, tau)
        body
        if tau >= tmax: break
      lastMaxFlt = tmax
