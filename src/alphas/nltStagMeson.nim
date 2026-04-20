## Original version of src/observables/nlStagMesonSpec.nim
## Author: Xiao-Yong Jin

#[
  This file is based on ltStagMesons.nim, implements time-split operators A_0 g0g5 x g5
  Author: Mingwei Dai 2026-04-17
]#

import qex, gauge/hisqsmear, gauge/gaugefix, physics/stagSolve
import times, strformat, os, math
import std/[os, times, sugar, tables]
import std/[parseopt, strutils, strformat]
import std/[json, sequtils]

import maths/complexType
import maths/complexProxy
# import maths/complexConcept
import physics/colorOld
import maths/types
import maths/matrixConcept

#dotim not verified
# proc dotim*(x: C2; y: C3): auto {.inline, noInit.} =
#   # x.re*y.re + x.im*y.im
#   mixin mul, imadd
#   var r{.noInit.}: type(x.re*y.re)
#   mul(r, x.re, y.im)
#   imsub(r, x.im, y.re)
#   r
template dotim*(x: Color; y: Color2): untyped =
  dotim(x[], y[])
template dotim*(x: ComplexProxy; y: ComplexProxy2): untyped =
  #(x.adj * y).re
  # below needed to workaround C++ backend issue (duplicate variable name)
  let xx = x
  let yy = y
  xx.re*yy.im - xx.im*yy.re
template dotim*(x: AsVar; y: AsVar): untyped =
  mixin dotim
  dotim(x[], y[])
proc dotim*(x: Vec2; y: Vec3): auto {.inline, noInit.} =
  result = dotim(x[0], y[0])
  forO i, 1, x.len.pred:
    result += dotim(x[i], y[i])
proc dotim*(x: Mat2; y: Mat3): auto {.inline, noInit.} =
  result = dotim(x[0, 0], y[0, 0])
  forO j, 1, x.len.pred:
    result += dotim(x[0, j], y[0, j])
  forO i, 1, x.len.pred:
    forO j, 0, x.len.pred:
      result += dotim(x[i, j], y[i, j])


const
  DefaultCoulombGaugeFixEpsilon = 1e-8
  DefaultCoulombGaugeFixRelax = 1.75

const
  DefaultConjugateGradientTolerance = 1e-20
  DefaultConjugateGradientMaximumIterations = 10000
  DefaultConjugateGradientVerbosty = 1

const Corners = 1
let
  defaultInputs = %* {
    "lattice-geometry": [16, 16, 16, 48],         # lattice geometry
                                                  #"rank-geometry": [1,1,2,2], # MPI rank geometry -- if not specified, guessed
                                                  #"simd-geometry": [2,2,1,2], # SIMD (local) geometry -- if not specified, guessed
    "solver": {
      "minimum-squared-residual": DefaultConjugateGradientTolerance,
      "maximum-iterations": DefaultConjugateGradientMaximumIterations,
      "verbosity": DefaultConjugateGradientVerbosty
    },
    "coulomb-gauge-fix": {                        # specify gauge fixing information
      "eps": DefaultCoulombGaugeFixEpsilon,       # stopping criterion for gauge fixing
      "relax": DefaultCoulombGaugeFixRelax        # gauge fixing relaxation factor
    },
    "spectrum": {
      "mass": 0.013,
      "source-time": 0
    }
  }

# useful for getting the time split at the source
# proc trans(u: auto): auto =
#   var r {.noinit.}: type(load1(u))
#   for i in 0..2:
#     for j in 0..2:
#       r[i, j] := u[j, i]
#   return r

# proc conjgt*(u: auto): auto =
#   var r {.noinit.}: type(load1(u))
#   for i in 0..2:
#     for j in 0..2:
#       r[i, j].re := u[i, j].re
#       r[i, j].im := -u[i, j].im
#   return r


proc readJSON(fn: string): JsonNode = fn.parseFile

proc readCMD: JsonNode =
  var cmd = initOptParser()
  result = parseJson("{}")
  while true:
    cmd.next()
    case cmd.kind:
      of cmdShortOption, cmdLongOption, cmdArgument:
        try: result[cmd.key] = %* parseInt(cmd.val)
        except ValueError:
          try: result[cmd.key] = %* parseFloat(cmd.val)
          except ValueError: result[cmd.key] = %* cmd.val
      of cmdEnd: break
# gets float or integer sequence from JsonNode object
proc getSeq[T](input: JsonNode; t: typedesc[T]): seq[T] =
  result = newSeq[T]()
  for elem in input.getElems():
    let telem = elem.getFloat().T
    result.add telem
# rephases links
proc rephase[U](u: seq[U]) =
  threads:
    u.setBC()
    threadBarrier()
    u.stagPhase()
# reads gauge configuration
proc readGauge[U](u: seq[U]; config: string) =
  if fileExists(config):
    if 0 != u.loadGauge(config): qexError "unable to read " & config
    else: discard
  else: qexError config & " does not exist"
# constructs lattice layout
proc newLayout(info: JsonNode): auto =
  let latLayout = info["lattice-geometry"].getSeq(int)
  result = case info.hasKey("rank-geometry")
    of true:
      case info.hasKey("simd-geometry"):
        of true:
          let rgeom = info["rank-geometry"].getSeq(int)
          let sgeom = info["simd-geometry"].getSeq(int)
          newLayout(latLayout, VLEN, rgeom, sgeom)
        of false: newLayout(latLayout, info["rank-geometry"].getSeq(int))
    of false: newLayout(latLayout)
  assert(latLayout.len == 4)

# XOR helper that accepts any number of integer arguments.
# calculates the total phase
func ps(a: varargs[int]): int =
  for x in a:
    result = result xor x

# Extract 0/1 corner bits.
template px(i: int): int = (i and 1)
template py(i: int): int = ((i shr 1) and 1)
template pz(i: int): int = ((i shr 2) and 1)
template pt(i: int): int = ((i shr 3) and 1)

# Convert a list of directions (1=x,2=y,3=z,4=t) into the corner index.
# given the forward corner and delta, return the backward corner to contract with
# e.g. for 1 link in y direction, you contract forward (1,1,0) with backward (1,0,0)
func shiftIdx(idx: int; delta: seq[int]): int =
  result = idx
  for d in delta:
    result = result xor (1 shl d)

proc shift4D(dest: Field; src: Field2; dt: int; delta: seq[int]) =
  # spatial shift part is not verified yet
  threads:
    dest := src
  if dt != 0:
    var sf = newShifter(dest, 3, 1)
    discard sf ^* dest
    threads:
      dest := sf.field

  for mu in delta: #each loop shifts in one direction
    var sf = newShifter(dest, mu, 1)
    var sb = newShifter(dest, mu, -1)
    discard sf ^* dest
    discard sb ^* dest
    threads:
      # TODO: normalize
      dest := sf.field + sb.field #U psi(x+mu)+U_dag psi(x-mu)
proc shift4D(u: auto; dest: Field; src: Field2; dt: int; delta: seq[int]) =
  # spatial shift part is not verified yet
  threads:
    dest := src
  if dt != 0:
    var sf = u.newTransporter(dest, 3, 1)
    discard sf ^* dest
    threads:
      dest := sf.field

  for mu in delta: #each loop shifts in one direction
    var sf = newShifter(dest, mu, 1)
    var sb = newShifter(dest, mu, -1)
    discard sf ^* dest
    discard sb ^* dest
    threads:
      # TODO: normalize
      dest := sf.field + sb.field #U psi(x+mu)+U_dag psi(x-mu)

# proc from the nlStagMeson, with changes to do different operators at the soruce and sink, cannot be called from outside of this file to avoid confusion with the one in nlStagMeson
proc shiftAndRephase(dest: Field; src: Field2; dt: seq[int]; delta: seq[seq[
    int]]; phase1: int; phase2: int; base: array[4, int]) =
  # shift part is not implemented yet
  threads:
    dest := src
  if dt[0] != 0:
    var sf = newShifter(dest, 3, 1)
    discard sf ^* dest
    threads:
      dest := sf.field

  for mu in delta[0]: #each loop shifts in one direction
    var sf = newShifter(dest, mu, 1)
    var sb = newShifter(dest, mu, -1)
    discard sf ^* dest
    discard sb ^* dest
    threads:
      # TODO: normalize
      dest := sf.field + sb.field #U psi(x+mu)+U_dag psi(x-mu)

  let bits1 = [phase1.px == 1, phase1.py == 1, phase1.pz == 1, phase1.pt == 1]
  let bits2 = [phase2.px == 1, phase2.py == 1, phase2.pz == 1, phase2.pt == 1]
  threads:
    for i in dest.sites:
      var expo = dt[3]-dt[0]
      for mu in 0..3:
        if bits1[mu]:
          expo += dest.l.coords[mu][i].int # phase from the first operator (the one at t,x,y,z)
          if dt[0] == dt[1] and mu == 0: # this is for when chibar and chi are both at t=1, there should be an extra -1 from the t direction, mu == 3 should be it but mu==0 also works. Need to verify this, but it doesn't affect the A0P correlator.
            expo += dt[0]
        if bits2[mu]:
          expo -= base[mu] # phase from the second operator(the one at 0,0,0,0)
        expo += dest.l.coords[mu][i].int # epsilon ((-1)^{x-x') from substituting G with G^dagger, asusming not delta, this needs addingi dt0 - dt3, since this is already shifted by dt[0], only dt3 is needed
      if (expo and 1) == 1:
        dest{i} *= -1

# inner‑product (real part) accumulated over the lattice and binned by t‑slice, added imaginary part, but needs verification, cannot be called from outside of thie file
proc dotByTimeslice(v1, v2: auto; real: bool = true): seq[float] =
  let nt = v1.l.physGeom[^1]
  result = newSeq[float](nt)
  for i in v1.sites:
    let t = v1.l.coords[3][i]
    if real:
      result[t] += v1{i}.redot v2{i}
    else:
      # imaginary part, code not verified yet, do not use for now
      # let dotim = dot(v1[i], v2[i])
      # result[t] += dotim.im
      result[t] += dotim(v1{i}, v2{i})
  result.rankSum

# -------------------------------------------------------------------
# Time split operators.
# IMPORTANT: We are counting from 0.
# -------------------------------------------------------------------

type MesonTerm* = tuple[delta: seq[int]; phase: int; srcT: seq[int]; factor: int]

const eta = [0, 1, 3, 7]
const zeta = [14, 12, 8, 0]
const eps = 15 # (1,1,1,1)

const klm = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
const klma = [(0, 1, 2), (0, 2, 1), (1, 0, 2),
              (1, 2, 0), (2, 0, 1), (2, 1, 0)]
#*****************************************************************************************************
proc buildnltMesonOps*(): seq[seq[MesonTerm]] =
  ## Return 3 meson ops, 
  ## 1. not used
  ## 2. A_0 (t+1/2) = \phi(t) \bar\chi \chi_+ (t+1)
  ## 3. Goldstone pion (t) = \bar\chi \chi (t) #g5 x g5 +- g0 x g0
  # TODO: label the states in the output
  result.newSeq(3)
  template add(n: int; d: seq[int]; ph: int; srcT: seq[int]; f: int = 1) =
    result[n].add (d, ph, srcT, f)

  add(0, @[], 0, @[0, 1], 1) #\bar\chi \chi_+
  add(0, @[], 0, @[1, 0], -1) #- \bar\chi_+ \chi
  add(1, @[], ps(eps), @[0, 1], 1) #\eps bar\chi \chi_+ A_0
  add(2, @[], ps(eta[3], zeta[3]), @[0, 0], 1) 
#*****************************************************************************************************
# Given any two operators, construct the correlator (inserting U_0 into the operator if asked to) 
# (not verified for link operators in spatial directions yet)
proc buildMesonCorrelator*(u: auto; opList1: seq[MesonTerm]; opList2: seq[
    MesonTerm]; props: auto; obar: array[2, int] = [1, 0]; flavor: seq[int];
    link: bool = false; realpart: bool = true): auto =
  # only the connected contraction
  # props must be props[color][srcT][corner][flavor]
  # obar means if the any of the ops is Obar, default to [1,0]
  # flavor has 4 elements specifying the flavor of the 4 \chi's, flavor[0]==flavor[3] and flavor[1]==flavor[2]
  # opList1 is the Obar operator if two ops are the same
  # assert obar[1] == 0, "only does <Obar O> for now"
  let lo = props[0][0][0][0].l
  let nt = lo.physGeom[^1]
  result = newSeq[float](nt)
  for t in 0..<nt:
    result[t] = 0.0
  var mflavor = @[0, 0, 0, 0]
  if obar == [0, 0]:
    mflavor = @[flavor[0], flavor[1], flavor[2], flavor[3]]
  elif obar == [0, 1]:
    mflavor = @[flavor[0], flavor[1], flavor[3], flavor[2]]
  elif obar == [1, 0]:
    mflavor = @[flavor[1], flavor[0], flavor[2], flavor[3]]
  elif obar == [1, 1]:
    mflavor = @[flavor[1], flavor[0], flavor[3], flavor[2]]
  assert mflavor[0] == mflavor[3] and mflavor[1] == mflavor[2], "Flavor doesn't match"
  for color in 0..<props.len:
    for delta2 in 0..<props[color][0].len:
      for termIdx1, term1 in opList1:
        for termIdx2, term2 in opList2:
          var deltas: seq[seq[int]] = @[newSeq[int](), newSeq[int](), newSeq[
              int](), newSeq[int]()] # deltas for the 4 \chi's
          if obar == [0, 0]:
            deltas = @[@[], opList1[termIdx1].delta, @[], opList2[
                termIdx2].delta]
          elif obar == [0, 1]:
            deltas = @[@[], opList1[termIdx1].delta, opList2[termIdx2].delta, @[]]
          elif obar == [1, 0]:
            deltas = @[opList1[termIdx1].delta, @[], @[], opList2[
                termIdx2].delta]
          elif obar == [1, 1]:
            deltas = @[opList1[termIdx1].delta, @[], opList2[termIdx2].delta, @[]]

          var dt = @[term1.srcT[0], term1.srcT[1], term2.srcT[0], term2.srcT[1]]
          if obar == [0, 1]:
            dt = @[term1.srcT[0], term1.srcT[1], term2.srcT[1], term2.srcT[0]]
          elif obar == [1, 0]:
            dt = @[term1.srcT[1], term1.srcT[0], term2.srcT[0], term2.srcT[1]]
          elif obar == [1, 1]:
            dt = @[term1.srcT[1], term1.srcT[0], term2.srcT[1], term2.srcT[0]]
          let delta3 = shiftIdx(delta2, term2.delta)
          #shift the props if delta is nonzero
          echo "dt ", dt
          echo "deltas ", deltas
          var tmp1 = lo.ColorVector()
          var tmp11 = lo.ColorVector()
          var tmp2 = lo.ColorVector()
          var tmp22 = lo.ColorVector()
          #prepare the forward prop
          let cc = lo.coords
          shift4D(tmp1, props[color][dt[2]][delta2][mflavor[2]], dt[1], deltas[1])
          # alternative way to add gauge link for when the sink is timesplit, but not correct for the source yet
          # if link == false:
          #   shift4D(tmp1, props[color][dt[2]][delta2][mflavor[2]], dt[1],
          #       deltas[1])
          # else:
          #   shift4D(u, tmp1, props[color][dt[2]][delta2][mflavor[2]], dt[1],
          #       deltas[1])
          # echo "Start multiplying "
          #prepare the backward prop
          var vec2 = [px(delta2), py(delta2), pz(delta2), dt[2]] #base
          if dt[2] != dt[3]:
            vec2 = [px(delta2), py(delta2), pz(delta2), 0]
          var vec3 = [px(delta3), py(delta3), pz(delta3), dt[3]] #base
          shiftAndRephase(tmp2, props[color][dt[3]][delta3][mflavor[3]], dt,
              deltas, term1.phase, term2.phase, vec2)
          #insert the gauge link into the operator
          if link:
            # if link==true, insert the gauge link manually.
            # u0 might be usefual for the timesplit operator at the source
            # var u0 {.noinit.}: type(load1(u[0]))
            # for site in u.sites:
            #   if (cc[0][site] == 0) and (cc[1][site] == 0) and (cc[2][site] ==
            #       0) and (cc[3][site] == 0):
            #     echo "site ", site
            #     for i in 0..2:
            #       for j in 0..2:
            #         u0[i, j].re := u{site}[i, j].re[][]
            #         u0[i, j].im := u{site}[i, j].im[][]
            #     echo u0
            #     break
            if dt[0] == 1 and dt[1] == 0:
              threads:
                for site in tmp1:
                  tmp1[site] := u[site].adj * tmp1[site]
            elif dt[0] == 0 and dt[1] == 1:
              threads:
                for site in tmp1:
                  tmp1[site] := u[site] * tmp1[site]
            # else:
            #   tmp11 := tmp1
            # if dt[2] == 1 and dt[3] == 0:
            #   threads:
            #     for site in tmp2:
            #       tmp2[site] := trans(u0) * tmp2[site]
            # elif dt[2] == 0 and dt[3] == 1:
            #   threads:
            #     for site in tmp2:
            #       tmp2[site] := conjgt(u0) * tmp2[site]
          let corr = dotByTimeslice(tmp2, tmp1, realpart)
          for t in 0..<nt:
            result[t] -= term1.factor.float * term2.factor.float * corr[t]


when isMainModule:
  qexInit()
  # command line information
  # print timing information & rank/thread info
  echo "rank ", myRank, "/", nRanks
  threads: echo "thread ", threadNum, "/", numThreads

  #[ setup ]#

  # command line information
  let cmd = readCMD()

  # information from stored JSON file on disk
  let lattice = case cmd.hasKey("lattice-json")
    of true: readJSON(cmd["lattice-json"].getStr())
    of false: defaultInputs
  let info = case cmd.hasKey("json")
    of true: defaultInputs #readJSON(cmd["json"].getStr())
    of false: defaultInputs

  # location of saved JSON file
  let corrFile = case cmd.hasKey("output")
    of true: cmd["output"].getStr()
    of false: "__NOOUTPUT__"

  let
    # conjugate gradient parameters
    r2req = info["solver"]["minimum-squared-residual"].getFloat()
    maxits = info["solver"]["maximum-iterations"].getInt()

    # gauge fixing parameters
    fixGauge: bool = true
    gfstop = info["coulomb-gauge-fix"]["eps"].getFloat()
    gforf = info["coulomb-gauge-fix"]["relax"].getFloat()

    # nhyp smearing parameters
    smearGauge: bool = true
    hisq = newHISQ(0.0, 1.0)

    # lattice layout
    lo = lattice.newLayout()

    # source time
    srcT = info["spectrum"]["source-time"].getInt()

    # time extent
    nt = lattice["lattice-geometry"].getSeq(int)[^1]

  # read in gauge configuration or fill with random links
  var g = lo.newGauge()
  case cmd.hasKey("configuration"):
    of true:
      echo "reading gauge configuration from ", cmd["configuration"].getStr()
      g.readGauge(cmd["configuration"].getStr())
    of false: g.random
  # reunitarize
  threads:
    block:
      let d = g.checkSU
      echo "unitary deviation avg: ", d.avg, "  max: ", d.max
    threadBarrier()
    g.projectSU
    threadBarrier()
    block:
      let d = g.checkSU
      echo "new unitary deviation avg: ", d.avg, "  max: ", d.max
  g.echoPlaq

  # Coulomb gauge fix
  # if fixGauge:
  #   tic "Coulomb gauge fixing"

  #   var tmat = lo.ColorMatrix()
  #   threads: tmat := 1
  #   getGaugeFixTransform(tmat, g, @[0, 1, 2, 3], gfstop, gforf, verb = 0)

  #   var fg = lo.newGauge
  #   fg.gaugeTransform(g, tmat)
  #   threads: g := fg

  #   qexLog "Coulomb gauge fixing done. Time: ", getElapsedTime()

  # rephase and smear
  var sg = lo.newGauge()
  var sgl = lo.newGauge()
  g.rephase()
  if smearGauge:
    tic "HISQ gauge smearing"
    discard hisq.smearGetForce(g, sg, sgl, displayPerformance = true)
    qexLog "HISQ gauge smearing done. Time: ", getElapsedTime()

  # instantiate staggered Dirac operator
  let stag = newStag3(sg, sgl)

  # instantiate solver parameters
  var sp = initSolverParams()
  sp.r2req = r2req
  sp.maxits = maxits

  let mesonOps = buildnltMesonOps()

  var props: array[3, array[2, array[1, array[1, typeof(lo.ColorVector)]]]]

  var nMesonTot = 0
  for ops in mesonOps.items:
    if ops.len > 0: nMesonTot += ops.len

  let ntTotal = info["lattice-geometry"].getSeq(int)[^1]
  let opList = buildnltMesonOps()
  let clist = [[opList[1], opList[1]], [opList[1], opList[1]], [opList[1],
      opList[2]], [opList[2], opList[1]], [opList[1], opList[2]], [opList[2],
          opList[1]], [opList[2], opList[2]]]
  var corrs = newSeq[seq[float]](clist.len)
  for i in 0..<clist.len:
    corrs[i].newSeq(ntTotal)

  # Helper to map (operator n, local index i) → global meson index
  proc mesonIndex(n, i: int): int =
    var off = 0
    for k in 0..<n:
      off += mesonOps[k].len
    result = off + i

  for c in 0..<g[0][0].nrows:
    tic "for one color"

    for srct in 0..1:
      var src = lo.ColorVector()
      threads:
        src := 0
        threadBarrier()
        for i in src.sites:
          let cc = lo.coords
          if cc[3][i] == srct and # t = srct, x,y,z=0,0,0
            (cc[0][i] and 1) == px(0) and
            (cc[1][i] and 1) == py(0) and
            (cc[2][i] and 1) == pz(0):
            src{i}[c] := 1.0
        threadBarrier()
        echo "color: ", c, "  time slice: ", srcT, "  srct: ", srct,
            "  src norm2: ", src.norm2

      props[c][srct][0][0] = lo.ColorVector()
      stag.solve(props[c][srct][0][0], src, info["spectrum"]["mass"].getFloat(), sp)
      # let sftmp = g[3].newTransporter(src,3,(-1)^(srct))#direction 3, length 1
      # stag.solve(props[c][(srct+1) and 1][0][0], sftmp.field, info["spectrum"]["mass"].getFloat(), sp)

    qexLog "Solves for color ", c, " done. Time: ", getElapsedTime()
  g.rephase()
  # obar tells the code to use the complex conjugate for the operator with obar = 1, the first is the sink and the second is the source
  var obar = [[1, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0]]
  for i in 0..<clist.len:
    echo "doing correlator ", i
    if i == 1 or i == 4 or i == 5: # print the imaginary part, use g[3] for U_0
      corrs[i] = g[3].buildMesonCorrelator(clist[i][0], clist[i][1], props,
          obar[i], @[0, 0, 0, 0], true, false)
    else: #print the real part
      corrs[i] = g[3].buildMesonCorrelator(clist[i][0], clist[i][1], props,
          obar[i], @[0, 0, 0, 0], true, true)

  if corrFile != "__NOOUTPUT__":
    let
      corrJSON = %* {
        "A0+re": corrs[0],
        "A0+im": corrs[1], # imaginary part not verified yet, do not use for now
        "A0Pre": corrs[2],
        "PA0re": corrs[3],
        "A0Pim": corrs[4],
        "PA0im": corrs[5],
        "P-P": corrs[6]
      }
      corrStr = pretty(corrJSON)
    writeFile(corrFile, corrStr)


  qexFinalize()
