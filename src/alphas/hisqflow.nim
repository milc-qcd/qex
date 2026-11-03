import qex
#import examples/[hisqhmc_h]
import hisqflowutils #hisqtestutils
import json

import gauge/[gaugefix, hisqsmear]
import physics/[stagSolve]
import observables/[nlStagMeson]
import conservedA0#nltStagMeson

import std/[os, times, sugar, tables] 
import std/[parseopt, strutils, strformat]
import std/[json, sequtils]
import std/[math]
import std/[re]


const
  DefaultCoulombGaugeFixEpsilon = 1e-8
  DefaultCoulombGaugeFixRelax = 1.75

const
  DefaultConjugateGradientTolerance = 1e-20
  DefaultConjugateGradientMaximumIterations = 10000
  DefaultConjugateGradientVerbosty = 1

const Corners = 1
const Masses = 1
const nsrcTime = 1

#[ IO ]#

proc readGauge(u: auto; fn: string) =
  if fileExists(fn):
    if 0 != u.loadGauge(fn): qexError "unable to read " & fn
    else: discard
  else: qexError fn & " does not exist"

proc getIntSeq*(input: JsonNode): seq[int] =
  result = newSeq[int]()
  for elem in input.getElems(): result.add elem.getInt()

proc readJSON(fn: string): JsonNode = fn.parseFile

proc readCMD: JsonNode =
  var cmd = initOptParser()
  result = parseJson("{}")
  while true:
    cmd.next()
    case cmd.kind:
      of cmdShortOption,cmdLongOption,cmdArgument:
        try: result[cmd.key] = %* parseInt(cmd.val)
        except ValueError:
          try: result[cmd.key] = %* parseFloat(cmd.val)
          except ValueError: result[cmd.key] = %* cmd.val
      of cmdEnd: break

# const
#   mass = [0.01, 0.02, 0.03]
proc extractMFloats(inputStr: string): seq[float] =
  ## Extracts all numbers following 'm' in a string and returns them as 0.xxxx floats.
  ## Example:
  ##   let floats = extractMFloats("l3248f211b580m002426m06730m8447a")
  ##   # Returns @[0.002426, 0.0673, 0.8447]
  result = @[]  # Initialize an empty sequence

  # Find all patterns where 'm' is followed by digits
  for m in inputStr.findAll(re"m(\d+)"):
    echo m[1 .. ^1]
    let decimalStr = "0." & m[1 .. ^1]  # Prepend "0." to the matched digits
    result.add(decimalStr.parseFloat)  # Convert to float and add to result

proc rephase[U](u: seq[U]) = 
  threads:
    u.setBC()
    threadBarrier()
    u.stagPhase()

const 
  logStyle = "KS_nHYP_FA"
  banner = """
|---------------------------------------------------------------|
 Quantum EXpressions (QEX)

 QEX authors: James Osborn & Xiao-Yong Jin
 QEX gradient flow authors: 
   - James Osborn (Argonne National Laboratory)
   - Curtis Taylor Peterson [C.T.P.] (Michigan State University)
 QEX fermion flow authors:
   - Mingwei Dai (University of Illinois Urbana-Champaign)
 QEX GitHub: https://github.com/jcosborn/qex
 Gauge flow GitHub: https://github.com/ctpeterson/qex
 C.T.P. email: curtistaylorpetersonwork@gmail.com
 cite: Proceedings of Science (PoS) LATTICE2016 (2017) 271
|---------------------------------------------------------------|
"""

proc reunit*(g: auto) =
  tic()
  threads:
    let d = g.checkSU
    threadBarrier()
    echo "unitary deviation avg: ",d.avg," max: ",d.max
    g.projectSU
    threadBarrier()
    let dd = g.checkSU
    echo "new unitary deviation avg: ",dd.avg," max: ",dd.max
  toc("reunit")

qexInit()

var 
  #reads command line arguments into a json object
  cmd = readCMD() 
  #e.g. command: ./fermionflow --flow-json:flow-info.json --lattice-json:lattice-info.json --configuration:100 --base-filename:config_name_stem --output:fflow_corr
  flowInfo = case cmd.hasKey("flow-json")
    of true: readJSON(cmd["flow-json"].getStr())
    of false:
      qexError "json file for flow information not specified"
      parseJson("{}")
  latInfo = case cmd.hasKey("lattice-json")
    of true: readJSON(cmd["lattice-json"].getStr())
    of false:
      qexError "json file for lattice information not specified"
      parseJson("{}")
  
  #contractInfo not used yet, planning to use it to specify the meson operators and the correlators to be measured in a more flexible way, but for now it's hardcoded in the code
  contractInfo = case cmd.hasKey("contract-json") # not used yet
    of true: readJSON(cmd["contract-json"].getStr())
    of false:
      echo "json file for contract information not specified"
      parseJson("{}")
  cfg = case cmd.hasKey("configuration")
    of true: $cmd["configuration"].getInt()
    of false:
      qexError "configuration number not specified"
      "0"
  # name of the configuration without the extension
  filename = case cmd.hasKey("base-filename")
    of true: cmd["base-filename"].getStr()
    of false: "checkpoint"
  latLayout = case latInfo.hasKey("lattice-geometry")
    of true: latInfo["lattice-geometry"].getIntSeq()
    of false: 
      qexError "must specify lattice-geometry in lattice input file"
      @[8,8,8,8]
  lo = case latInfo.hasKey("rank-geometry")
    of true:
      case latInfo.hasKey("simd-geometry"):
        of true:
          newLayout(
            latLayout,
            VLEN,
            latInfo["rank-geometry"].getIntSeq(),
            latInfo["simd-geometry"].getIntSeq()
          )
        of false: newLayout(latLayout,latInfo["rank-geometry"].getIntSeq())
    of false: newLayout(latLayout)
  u = lo.newGauge()
# Set mass from filename
# let mass = extractMFloats(filename)[0..Masses]#@[0.013]# use this to work on all masses
let mass = @[extractMFloats(filename)[Masses]]#@[0.013]# only doing the strange quark mass for now to save time
echo "mass: ", mass

# find the name of the flow and insert it into the output filename
var flowname = ""
for flow, flowin in flowInfo:
  flowname = flow
  break
let corrFile = case cmd.hasKey("output")
  of true: cmd["output"].getStr() & "/correlator_" & flowname & "_" & cfg
  of false: "__NOOUTPUT__"
# put it into the gradient flow file name
let gfobsFile = case cmd.hasKey("output")
  of true: cmd["output"].getStr() & "/gaugeobs_" & flowname & "_" & cfg
  of false: "__NOOUTPUT__"
echo "corrFile: ", corrFile
echo "gfobsFile: ", gfobsFile
#read configuration
echo "Start leading configuration. "
u.readGauge(filename & "." & cfg & ".ildg")
# u.random
echo "Loading complete. "
#initialize stagSolve
let
    # conjugate gradient parameters
  r2req = DefaultConjugateGradientTolerance
  maxits = DefaultConjugateGradientMaximumIterations
    # gauge fixing parameters
  fixGauge: bool = true
  gfstop = DefaultCoulombGaugeFixEpsilon
  gforf = DefaultCoulombGaugeFixRelax
    # nhyp smearing parameters
  smearGauge: bool = true
  # nhyp = HypCoefs(alpha1: 0.4, alpha2: 0.5, alpha3: 0.5)
  hisq = newHisq(0.0,1.0) 
  hisq2 = newHisq(0.0,0.0)
    # source time
  srcT = 0
    # time extent
  nt = latInfo["lattice-geometry"].getIntSeq[^1]
threads:
  block:
    let d = u.checkSU
    echo "unitary deviation avg: ", d.avg, "  max: ", d.max
  threadBarrier()
  u.projectSU
  threadBarrier()
  block:
    let d = u.checkSU
    echo "new unitary deviation avg: ", d.avg, "  max: ", d.max
u.echoPlaq
# Coulomb gauge fix
if false:#fixGauge:
  tic "Coulomb gauge fixing"
  var tmat = lo.ColorMatrix()
  threads: tmat := 1
  getGaugeFixTransform(tmat, u, @[0,1,2], gfstop, gforf, verb=0) #@[012] for coulomb gague, [0123] for landau gauge
  var fg = lo.newGauge
  fg.gaugeTransform(u, tmat)
  threads: u := fg
  qexLog "Coulomb gauge fixing done. Time: ", getElapsedTime()
  # qexLog "Saving smeared gauge field"
  # discard u.saveGauge(filename & "_gaugefixed." & cfg & ".ildg")


# instantiate solver parameters
var sp = initSolverParams()
sp.r2req = r2req
sp.maxits = maxits

#[ spectrum ]#
# build meson operators & select mesons as subset
let 
  mesonOps = buildMesonOps()
  nltOps = buildnltMesonOps()
  # nltOps[1] is A_0, nltOps[2] is Goldstone pion operator, so this is < A_0 P>
  nltcorrList = [[nltOps[1],nltOps[2]]]
  mesons = [
    mesonOps[0],  # γ_0 γ_5 ⊗ γ_0 γ_5 (P = +σ) I     π_05  <--+ 
      mesonOps[1],  # γ_5 ⊗ ξ_5         (P = -σ) II    π_5      |
      # mesonOps[6],  # γ_5 ⊗ ξ_μ ξ_5     (P = +σ) VII   π_i5     |
      # mesonOps[7],  # γ_0 γ_5 ⊗ ξ_μ ξ_ν (P = -σ) VIII  π_ij     | scalar &
      # mesonOps[12], # γ_0 γ_5 ⊗ ξ_μ     (P = -σ) XIII  π_0i     | pseudoscalar
      # mesonOps[13], # γ_5 ⊗ ξ_μ         (P = -σ) XIV   π_i      |
      # mesonOps[16], # γ_0 γ_5 ⊗ I       (P = -σ) XVI   π_I      |
      # mesonOps[17], # γ_5 ⊗ ξ_0         (P = +σ) XVII  π_0   <--+
      mesonOps[2],  # γ_0 γ_k ⊗ ξ_0 ξ_μ (P = +σ) XIII  ρ_0i  <--+
      mesonOps[3],  # γ_μ ⊗ ξ_μ         (P = -σ) XIV   ρ_i      |
      # mesonOps[4],  # γ_μ ⊗ I           (P = +σ) XV    ρ_I      |
      # mesonOps[5],  # γ_0 γ_μ ⊗ ξ_0     (P = +σ) VI    ρ_0      |
      # mesonOps[8],  # γ_0 γ_μ ⊗ ξ_μ ξ_5 (P = +σ) XIV   ρ_i5     |
      # mesonOps[9],  # γ_μ ⊗ ξ_μ ξ_ν     (P = +σ) X     ρ_ij     | vector &
      # mesonOps[10], # γ_0 γ_μ ⊗ ξ_0 ξ_5 (P = -σ) XI    ρ_05     | pseudovector
      # mesonOps[11], # γ_0 γ_μ ⊗ ξ_5     (P = -σ) XII   ρ_5      |
      # mesonOps[14], # γ_μ ⊗ ξ_μ         (P = -σ) XIV   ρ_i*     |
      # mesonOps[15], # γ_0 γ_k ⊗ ξ_0 ξ_μ (P = -σ) XIII  ρ_0i*    |
      # mesonOps[18], # γ_0 γ_μ ⊗ ξ_μ ξ_5 (P = -σ) XVIII ρ_i5*    |
      # mesonOps[19], # γ_μ ⊗ ξ_μ ξ_ν     (P = +σ) XX    ρ_ij* <--+
    ]
  
  massname = [ # quark names that constitute the meson
    "ss",# "ss", #"cc",# "ls", "lc", "sc", "sl"
  ]
  mesonname = [
    "pi_05", 
    "pi_5",
    # "pi_i5",
    # "pi_ij",
    # "pi_0i",
    # "pi_i",
    # "pi_I",
    # "pi_0",
    "rho_0i",
    "rho_i",
    # "rho_I",
    # "rho_0",
    # "rho_i5",
    # "rho_ij",
    # "rho_05",
    # "rho_5",
    # "rho_i*",
    # "rho_0i*",
    # "rho_i5*",
    # "rho_ij*",
  ]
# initialize propagators and correlators
const colors = u[0][0].nrows
var 
  # initilize the propagators:
  props: array[colors, array[nsrcTime, array[Corners, array[Masses, typeof(lo.ColorVector)]]]]
  # specify what the flavor of the two quarks, 0,0 for ss in the case only one mass is considered
  mprops = [[0,0]] 
  # initialize the correlators, we will compute meson correlators for each pair of quarks in mprops for time local operators in mesons and time split operators in nltcorrList, so the total number of meson correlators for each pair of quarks is mesons.len + nltcorrList.len
  corrs = newSeqWith(mprops.len, newSeqWith(mesons.len+nltcorrList.len, newSeq[float]()))
for color in 0..<colors:
  for srct in 0..<nsrcTime:
    for massIdx in 0..<Masses:
      for corner in 0..<Corners: props[color][srct][corner][massIdx] = lo.ColorVector()
for propindex in 0..<mprops.len:
  for pionIdx in 0..<mesons.len+nltcorrList.len:
    corrs[propindex][pionIdx].newSeq(nt)
tic "HISQ Dirac inversion"
# get spectrum and do it at each color
block:
  # smear and rephase
  var su = lo.newGauge()
  var sul = lo.newGauge()
  u.rephase() #phase in
  if smearGauge:
    tic "HISQ gauge smearing"
    discard hisq.smearGetForce(u, su, sul)
    # threads: u := su ???????????
    qexLog "nHISQ gauge smearing done. Time: ", getElapsedTime()
  # u.rephase()
  
  let stag = newStag3(su,sul)
  for color in 0..<colors:
    for srct in 0..<nsrcTime:
      for corner in 0..<Corners:
        # generate color cource
        var src = lo.ColorVector()
        threads:
          let cc = lo.coords
          src := 0
          threadBarrier()
          for i in src.sites:
            if cc[3][i] == srct and
              (cc[0][i] and 1) == px(corner) and
              (cc[1][i] and 1) == py(corner) and
              (cc[2][i] and 1) == pz(corner):
              src{i}[color] := 1.0
      # invert D to get propagator
        for massIdx in 0..<Masses:
          threads: props[color][srct][corner][massIdx] := 0
          stag.solve(props[color][srct][corner][massIdx], src, mass[massIdx], sp)
qexGC("after HISQ Dirac inversion")
u.rephase() #phase out

# echo "Saving the correlatos at tau=0.00"
qexLog "HISQ propagators done. Time: ", getElapsedTime()
var gfobsJSON = %* {}
if gfobsFile != "__NOOUTPUT__":
  let tauStr = formatFloat(0, ffDecimal, 2)
  for flow, flowin in flowInfo:
    # there must be better ways to do this, but for now we just assume the number of loops is specified in the flow json file and we only do one flow
    let loop = case flowInfo[flow].hasKey("loops")
      of true: flowInfo[flow]["loops"].getInt()
      of false: 1
    gfobsJSON[tauStr] = u.flowMeasurements(loop, 0)
    break

var corrJSON = %* {}
if corrFile != "__NOOUTPUT__":
  let tauStr = formatFloat(0, ffDecimal, 2)
  corrJSON[tauStr] = %* {}
  qexLog "Measuring the propagators at tau=0.0"
  for color in 0..<colors:
    for pionIdx in 0..<mesons.len:
      for termIdx, term in mesons[pionIdx]:
        for forward in 0..<Corners:
          # echo "forward ", forward, "term.delta ", term.delta
          let 
            backward = shiftIdx(forward, term.delta)
            base = [px(forward), py(forward), pz(forward), srcT]
          # echo "forward ", forward, " backward ", backward
          var tmp = lo.ColorVector()
  
              # Build shifted + rephased copy of `props[backward]`.
          for propindex in 0..<mprops.len:
            shiftAndRephase(tmp, props[color][0][backward][mprops[propindex][0]], term.delta, 15 - term.phase, base)
      
                # Correlator for this pair, binned by timeslice
            let corr = dotByTimeslice(props[color][0][forward][mprops[propindex][1]], tmp)
            for t in 0..<nt: corrs[propindex][pionIdx][t] -= term.factor.float * corr[t]
  # Measure the thin-link and conserved A0P before allocating the HISQ2 links.
  block:
    u.rephase()
    var fu = lo.newGauge()
    var ful = lo.newGauge()
    tic "HISQ gauge smearing for time-split correlators at tau=" & tauStr
    discard hisq.smearGetForce(u, fu, ful)

    u.rephase()
    qexLog "HISQ gauge smearing for time-split correlators done at tau=",
        tauStr, ". Time: ", getElapsedTime()

    for propindex in 0..<mprops.len:
      for pionIdx in 0..<mesons.len:
        let combinedName = mesonname[pionIdx] & "_" & massname[propindex]
        corrJSON["0.00"][combinedName] = %* corrs[propindex][pionIdx]

      for nltindex in 0..<nltcorrList.len: 
        let
          flavor = @[mprops[propindex][0], mprops[propindex][1],
              mprops[propindex][0], mprops[propindex][1]]
          thinRe = u[3].buildMesonCorrelator(nltcorrList[nltindex][0],
              nltcorrList[nltindex][1], props, [0, 0], flavor, true, true)
          thinIm = u[3].buildMesonCorrelator(nltcorrList[nltindex][0],
              nltcorrList[nltindex][1], props, [0, 0], flavor, true, false)
          consRe = buildConservedA0PCorrelator(u, fu, ful, props, flavor, true)
          consIm = buildConservedA0PCorrelator(u, fu, ful, props, flavor, false)
        corrJSON[tauStr]["A0P_thin_re_" & massname[propindex]] = %* thinRe
        corrJSON[tauStr]["A0P_thin_im_" & massname[propindex]] = %* thinIm
        corrJSON[tauStr]["cA0P_re_" & massname[propindex]] = %* consRe
        corrJSON[tauStr]["cA0P_im_" & massname[propindex]] = %* consIm

  qexGC("after measuring thin and conserved correlators at tau=0.00")

  # Build the HISQ2 one-link insertion only after `fu` and `ful` leave scope.
  block:
    u.rephase()
    var fu2 = lo.newGauge()
    var ful2 = lo.newGauge()
    discard hisq2.smearGetForce(u, fu2, ful2)
    u.rephase()
    qexLog "second HISQ gauge smearing for time-split correlators done at tau=",
        tauStr, ". Time: ", getElapsedTime()

    for propindex in 0..<mprops.len:
      for nltindex in 0..<nltcorrList.len:
        let
          flavor = @[mprops[propindex][0], mprops[propindex][1],
              mprops[propindex][0], mprops[propindex][1]]
          hisqRe = fu2[3].buildMesonCorrelator(nltcorrList[nltindex][0],
              nltcorrList[nltindex][1], props, [0, 0], flavor, true, true)
          hisqIm = fu2[3].buildMesonCorrelator(nltcorrList[nltindex][0],
              nltcorrList[nltindex][1], props, [0, 0], flavor, true, false)
        corrJSON[tauStr]["A0P_hisq_re_" & massname[propindex]] = %* hisqRe
        corrJSON[tauStr]["A0P_hisq_im_" & massname[propindex]] = %* hisqIm
        corrJSON[tauStr]["A0P_re_" & massname[propindex]] = %* hisqRe
        corrJSON[tauStr]["A0P_im_" & massname[propindex]] = %* hisqIm

  qexGC("after measuring correlators at tau=0.00")
  
for flow in flowInfo.keys():
  flowInfo[flow]["filename"] = %* (flow & "_" & cfg & ".log")
echo "Starting fermion flow"
let flowMass = newSeq[float](mass.len)
echo "flowMass: ", flowMass
u.staggeredFlow(props,flowMass,flowInfo):
  echo "Flow at tau= ", tau
  echo "mass of the flow ", flowMass
  echo props[0][0][0][0][0][0] # print the first element of the first propagator as a sanity check to make sure the flow is actually doing something to the propagators
  if gfobsFile != "__NOOUTPUT__":
    echo "saving gauge observables at tau= ", tau
    let tauStr = formatFloat(tau, ffDecimal, 2)
    gfobsJSON[tauStr] = measurements
  if abs(tau.round(1) - tau) < 1e-10: # if tau is 0.1, 0.2 ...
    if corrFile != "__NOOUTPUT__":
      echo "clearing corrs cache"
      threads:
        for propindex in 0..<mprops.len:
          for pionIdx in 0..<mesons.len: 
            for t in 0..<nt:
              corrs[propindex][pionIdx][t] := 0
      echo "Measuring the propagators at tau= ", tau
      let tauStr = formatFloat(tau, ffDecimal, 2)
      corrJSON[tauStr] = %* {}
      for color in 0..<colors:
        for pionIdx in 0..<mesons.len:
          for termIdx, term in mesons[pionIdx]:
            for forward in 0..<Corners:
              let 
                backward = shiftIdx(forward, term.delta)
                base = [px(forward), py(forward), pz(forward), srcT]
              var tmp = lo.ColorVector()
      
              # Build shifted + rephased copy of `props[backward]`.
              for propindex in 0..<mprops.len:
                shiftAndRephase(tmp, props[color][0][backward][mprops[propindex][0]], term.delta, 15 - term.phase, base)
      
                # Correlator for this pair, binned by timeslice
                let corr = dotByTimeslice(props[color][0][forward][mprops[propindex][1]], tmp)
                for t in 0..<nt: corrs[propindex][pionIdx][t] -= term.factor.float * corr[t]
      
      # save output to convenient JSON format
      # echo "Saving correlation functions at tau= ", tau
      block:
        u.rephase()
        var fu = lo.newGauge()
        var ful = lo.newGauge()
        tic "HISQ gauge smearing for time-split correlators at tau=" & tauStr
        discard hisq.smearGetForce(u, fu, ful)
        
        u.rephase()
        qexLog "HISQ gauge smearing for time-split correlators done at tau=",
            tauStr, ". Time: ", getElapsedTime()
        
        for propindex in 0..<mprops.len:
          for pionIdx in 0..<mesons.len:
            let combinedName = mesonname[pionIdx] & "_" & massname[propindex]
            corrJSON[tauStr][combinedName] = %* corrs[propindex][pionIdx]
        
          for nltindex in 0..<nltcorrList.len:
            # A0P with the thin link, HISQ smeared one-link, and full conserved
            # HISQ current.  The legacy A0P key is kept as the HISQ-smeared result.
            let
              flavor = @[mprops[propindex][0], mprops[propindex][1],
                  mprops[propindex][0], mprops[propindex][1]]
              thinRe = u[3].buildMesonCorrelator(nltcorrList[nltindex][0],
                  nltcorrList[nltindex][1], props, [0, 0], flavor, true, true)
              thinIm = u[3].buildMesonCorrelator(nltcorrList[nltindex][0],
                  nltcorrList[nltindex][1], props, [0, 0], flavor, true, false)
              consRe = buildConservedA0PCorrelator(u, fu, ful, props, flavor, true)
              consIm = buildConservedA0PCorrelator(u, fu, ful, props, flavor, false)
            corrJSON[tauStr]["A0P_thin_re_" & massname[propindex]] = %* thinRe
            corrJSON[tauStr]["A0P_thin_im_" & massname[propindex]] = %* thinIm

            corrJSON[tauStr]["cA0P_re_" & massname[propindex]] = %* consRe
            corrJSON[tauStr]["cA0P_im_" & massname[propindex]] = %* consIm
            
      qexGC("after measuring correlators at tau=" & tauStr)
      block:
        u.rephase()
        var fu2 = lo.newGauge()
        var ful2 = lo.newGauge()
        discard hisq2.smearGetForce(u, fu2, ful2)
        u.rephase()
        qexLog "second HISQ gauge smearing for time-split correlators done at tau=",
            tauStr, ". Time: ", getElapsedTime()
        
        for propindex in 0..<mprops.len:
          # for pionIdx in 0..<mesons.len:
          #   let combinedName = mesonname[pionIdx] & "_" & massname[propindex]
          #   corrJSON[tauStr][combinedName] = %* corrs[propindex][pionIdx]
        
          for nltindex in 0..<nltcorrList.len:
            # A0P with the thin link, HISQ smeared one-link, and full conserved
            # HISQ current.  The legacy A0P key is kept as the HISQ-smeared result.
            let
              flavor = @[mprops[propindex][0], mprops[propindex][1],
                  mprops[propindex][0], mprops[propindex][1]]
              hisqRe = fu2[3].buildMesonCorrelator(nltcorrList[nltindex][0],
                  nltcorrList[nltindex][1], props, [0, 0], flavor, true, true)
              hisqIm = fu2[3].buildMesonCorrelator(nltcorrList[nltindex][0],
                  nltcorrList[nltindex][1], props, [0, 0], flavor, true, false)

            corrJSON[tauStr]["A0P_hisq_re_" & massname[propindex]] = %* hisqRe
            corrJSON[tauStr]["A0P_hisq_im_" & massname[propindex]] = %* hisqIm
            corrJSON[tauStr]["A0P_re_" & massname[propindex]] = %* hisqRe
            corrJSON[tauStr]["A0P_im_" & massname[propindex]] = %* hisqIm

echo "Saving correlators and gauge observables"
let gfobsStr = pretty(gfobsJSON)
let corrStr = pretty(corrJSON)
writeFile(gfobsFile, gfobsStr)
writeFile(corrFile, corrStr)


qexFinalize()
