import qex
import examples/[hisqhmc_h]
import staggeredflowutils
import json

import gauge/[gaugefix, hisqsmear]
import physics/[stagSolve]
import observables/[nlStagMeson]
import nltStagMeson

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
u.readGauge(filename & "_" & cfg & ".lat")
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
      # mesonOps[2],  # γ_0 γ_k ⊗ ξ_0 ξ_μ (P = +σ) XIII  ρ_0i  <--+
      # mesonOps[3],  # γ_μ ⊗ ξ_μ         (P = -σ) XIV   ρ_i      |
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
    "pi_i5",
    "pi_ij",
    "pi_0i",
    "pi_i",
    "pi_I",
    "pi_0",
    "rho_0i",
    "rho_i",
    "rho_I",
    "rho_0",
    "rho_i5",
    "rho_ij",
    "rho_05",
    "rho_5",
    "rho_i*",
    "rho_0i*",
    "rho_i5*",
    "rho_ij*",
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
  # Doing the time-split correlators
  for propindex in 0..<mprops.len:
    for nltindex in 0..<nltcorrList.len:
      corrs[propindex][mesons.len+nltindex] = u[3].buildMesonCorrelator(nltcorrList[nltindex][0],nltcorrList[nltindex][1],props,[0,0],@[mprops[propindex][0],mprops[propindex][1],mprops[propindex][0],mprops[propindex][1]],true,true)
  for propindex in 0..<mprops.len:
    for pionIdx in 0..<mesons.len:
      let combinedName = mesonname[pionIdx] & "_" & massname[propindex]
      corrJSON["0.00"][combinedName] = %* corrs[propindex][pionIdx]

    for nltindex in 0..<nltcorrList.len: 
      # need to get the name of the ith time-split operator, for now we just call it A0P since we only have one time-split operator which is A_0 g0g5 x g
      corrJSON["0.00"]["A0P_re_" & massname[propindex]] = %* corrs[propindex][mesons.len+nltindex]
  
for flow in flowInfo.keys():
  flowInfo[flow]["filename"] = %* (flow & "_" & cfg & ".log")
echo "Starting fermion flow"
u.staggeredFlow(props,mass,flowInfo):
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
      for propindex in 0..<mprops.len:
        for pionIdx in 0..<mesons.len:
          let combinedName = mesonname[pionIdx] & "_" & massname[propindex]
          corrJSON[tauStr][combinedName] = %* corrs[propindex][pionIdx]
    
        corrJSON[tauStr]["A0P_re_" & massname[propindex]] = %* u[3].buildMesonCorrelator(nltcorrList[propindex][0],nltcorrList[propindex][1],props,[0,0],@[mprops[propindex][0],mprops[propindex][1],mprops[propindex][0],mprops[propindex][1]],true,true)
        corrJSON[tauStr]["A0P_im_" & massname[propindex]] = %* u[3].buildMesonCorrelator(nltcorrList[propindex][0],nltcorrList[propindex][1],props,[0,0],@[mprops[propindex][0],mprops[propindex][1],mprops[propindex][0],mprops[propindex][1]],true,false)

echo "Saving correlators and gauge observables"
let gfobsStr = pretty(gfobsJSON)
let corrStr = pretty(corrJSON)
writeFile(gfobsFile, gfobsStr)
writeFile(corrFile, corrStr)


qexFinalize()
