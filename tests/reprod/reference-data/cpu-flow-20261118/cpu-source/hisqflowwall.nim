## CPU HISQ flow with N random walls or 3N color-diluted corner walls.
## Add --source=random-wall|corner-wall --sources=N --seed=S to normal inputs,
## or --source-times='[0,17,63]' (N is inferred, or must match --sources).
## Random times are uniform with replacement. Corner positions are also drawn
## from the seed; --source-corner=0..7 pins the corner for every source time.
## PP and forward thin-link/conserved A0P only; source operator always P.
import qex
#import examples/[hisqhmc_h]
import hisqflowutils #hisqtestutils
import cpuTimings
import json

import gauge/[gaugefix, hisqsmear]
import physics/[stagSolve]
import conserveA0rngw
import comms/qmp

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

proc drawSourceIndex(rng: var RngMilc6; extent: int): int =
  if extent <= 0 or extent > rng.numInts:
    qexError "source extent is outside the RNG integer range"
  let limit = rng.numInts - (rng.numInts mod extent)
  while true:
    let value = rng.integer
    if value < limit:
      return value mod extent

proc writeSourceRecord(path: string; record: JsonNode; initial = false) =
  let recordStart = cpuTimerStart()
  defer: cpuTimerStop(ctOutputIo, recordStart)
  if myRank == 0:
    # A root-only error must not call qexError's collective barrier.
    if initial and fileExists(path):
      qexFatal "source metadata already exists: " & path
    try:
      writeFile(path, pretty(record))
    except CatchableError as e:
      qexFatal "cannot write source metadata: " & e.msg

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
resetCpuTimings()

let cmd = readCMD()
var sourceMode = "corner-wall"
if cmd.hasKey("source"):
  if cmd["source"].kind != JString:
    qexError "source must be corner-wall or random-wall"
  sourceMode = cmd["source"].getStr()
if sourceMode notin ["corner-wall", "random-wall"]:
  qexError "source must be corner-wall or random-wall"
let colorsPerTime = if sourceMode == "random-wall": 1 else: 3
var sourceCount = 1
if cmd.hasKey("sources"):
  if cmd["sources"].kind != JInt or cmd["sources"].getBiggestInt() <= 0:
    qexError "sources must be a positive integer"
  sourceCount = cmd["sources"].getInt()
var explicitTimes: seq[int]
if cmd.hasKey("source-times"):
  var timesNode: JsonNode
  try:
    timesNode = parseJson(cmd["source-times"].getStr())
  except CatchableError:
    qexError "source-times must be a nonempty JSON array of integers"
  if timesNode.kind != JArray or timesNode.len == 0:
    qexError "source-times must be a nonempty JSON array of integers"
  for value in timesNode:
    if value.kind != JInt:
      qexError "source-times must be a nonempty JSON array of integers"
    explicitTimes.add(value.getInt())
  if cmd.hasKey("sources") and sourceCount != explicitTimes.len:
    qexError "sources must equal the length of source-times"
  sourceCount = explicitTimes.len
if sourceCount > high(int) div colorsPerTime:
  qexError "sources is too large"
if cmd.hasKey("source-corner"):
  if sourceMode != "corner-wall":
    qexError "source-corner requires source=corner-wall"
  if cmd["source-corner"].kind != JInt or
      cmd["source-corner"].getInt() notin 0..7:
    qexError "source-corner must be an integer in 0..7"
if cmd.hasKey("save-source"):
  if cmd["save-source"].kind != JString or cmd["save-source"].getStr().len == 0:
    qexError "save-source requires a nonempty filename"
if cmd.hasKey("source-time"):
  qexError "use source-times=[T] to specify a single source time"
if cmd.hasKey("seed"):
  if cmd["seed"].kind != JInt or cmd["seed"].getBiggestInt() < 0 or
      cmd["seed"].getBiggestInt() > high(uint32).int64:
    qexError "seed must fit in an unsigned 32-bit integer"

var
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
cpuTimed(ctGaugeInput): u.readGauge(filename & "." & cfg & ".ildg")
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
    # time extent
  nt = latInfo["lattice-geometry"].getIntSeq[^1]
# Separate scalar streams for times/corners and global-site streams for noise.
if latLayout.len != 4 or latLayout.anyIt(it <= 0 or (it and 1) != 0):
  qexError "wall sources require four positive even lattice extents"
for t in explicitTimes:
  if t < 0 or t >= nt:
    qexError "each source time must be in 0..<Nt"
var seed: uint64
if cmd.hasKey("seed"):
  let value = cmd["seed"].getBiggestInt()
  if value < 0 or value > high(uint32).int64:
    qexError "seed must fit in an unsigned 32-bit integer"
  seed = value.uint64
elif myRank == 0:
  seed = uint64(epochTime() * 1e6) and high(uint32).uint64
QMP_broadcast(seed.addr, sizeof(seed).csize_t)
var timeRng: RngMilc6
timeRng.seedIndep(seed, lo.physVol)
var cornerRng: RngMilc6
cornerRng.seedIndep(seed, lo.physVol + 1)
# Draw with replacement. Increasing sourceCount preserves the existing prefix.
var wallTimes = newSeq[int](sourceCount)
var wallCorners = newSeq[int](sourceCount)
for hit in 0..<sourceCount:
  let drawnTime = drawSourceIndex(timeRng, nt)
  wallTimes[hit] = if explicitTimes.len > 0: explicitTimes[hit] else: drawnTime
  if sourceMode == "corner-wall":
    let drawnCorner = drawSourceIndex(cornerRng, 8)
    wallCorners[hit] = if cmd.hasKey("source-corner"):
      cmd["source-corner"].getInt() else: drawnCorner
echo sourceMode, " seed: ", seed, " source times: ", wallTimes
var sourceInfo = %* {
  "source-kind": sourceMode, "seed": seed,
  "source-count": sourceCount, "source-times": wallTimes,
  "source-colors": 3, "inversions-per-time": colorsPerTime,
  "inversions-per-flavor": sourceCount * colorsPerTime,
  "source-average": "arithmetic mean after aligning each source time",
  "rng-scheme": "RngMilc6 wall v1; times=volume; corners=volume+1; noise=global sites, sequential Z4",
  "time-sampling": (if explicitTimes.len > 0: "explicit list" else: "uniform with replacement"),
  "lattice-geometry": latLayout, "source-operator": "P = epsilon * barChi * chi",
  "configuration": cfg, "gauge-file": filename & "." & cfg & ".ildg", "mass": mass,
  "time-index": "(sink time - source time + Nt) mod Nt",
  "current-component": "forward only; conserved current includes Naik"
}
if sourceMode == "random-wall":
  sourceInfo["source"] = %"Z4 full-color random wall"
  sourceInfo["normalization"] = %"divide by spatial volume; color trace is summed"
else:
  sourceInfo["source"] = %"color-diluted corner wall"
  sourceInfo["source-corners"] = %wallCorners
  sourceInfo["source-corner-fixed"] = %cmd.hasKey("source-corner")
  sourceInfo["normalization"] = %"legacy corner-wall color sum; no spatial-volume division"
  echo "source corners (x + 2*y + 4*z): ", wallCorners
let metadataFile = if cmd.hasKey("output"):
    cmd["output"].getStr() / ("metadata_" & flowname & "_" & cfg & ".json")
  else: "metadata_" & flowname & "_" & cfg & ".json"
writeSourceRecord(metadataFile, sourceInfo, initial = true)
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
cpuTimed(ctGaugeMeasurements): u.echoPlaq
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

#[ spectrum: P source, PP and thin-link/conserved A0P sinks ]#
let massname = ["ss"]
# initialize propagators and correlators
var 
  # initilize the propagators:
  props = newSeq[array[nsrcTime, array[Corners, array[Masses, typeof(lo.ColorVector)]]]](sourceCount * colorsPerTime)
  # specify what the flavor of the two quarks, 0,0 for ss in the case only one mass is considered
  mprops = [[0,0]] 
for slot in 0..<props.len:
  for srct in 0..<nsrcTime:
    for massIdx in 0..<Masses:
      for corner in 0..<Corners: props[slot][srct][corner][massIdx] = lo.ColorVector()
tic "HISQ Dirac inversion"
# One full-color inversion or three color-diluted inversions per time and mass.
var sourceRecords = newJArray()
block:
  # smear and rephase
  let solveLinksStart = cpuTimerStart()
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
  cpuTimerStop(ctHisqLinksSolve, solveLinksStart)
  # Seed once, then advance all site streams for every source (also off-wall sites).
  let sourceRngStart = cpuTimerStart()
  var rng = newRNGField(RngMilc6, lo, seed)
  cpuTimerStop(ctSourceGeneration, sourceRngStart)
  for slot in 0..<props.len:
    let hit = slot div colorsPerTime
    let sourceColor = slot mod colorsPerTime
    for srct in 0..<nsrcTime:
      for corner in 0..<Corners:
        # Flatten (hit, dilution color) so the existing CPU flow is unchanged.
        let sourceStart = cpuTimerStart()
        var src = lo.ColorVector()
        let wallTime = wallTimes[hit]
        var hitInfo = sourceInfo.copy()
        hitInfo["source-index"] = %hit
        hitInfo["source-time"] = %wallTime
        if sourceMode == "random-wall":
          threads:
            src.z4 rng
            threadBarrier()
            for i in src.sites:
              if lo.coords[3][i].int != wallTime:
                src{i} := 0
        else:
          let sourceCorner = wallCorners[hit]
          hitInfo["source-color"] = %sourceColor
          hitInfo["source-corner"] = %sourceCorner
          threads:
            src := 0
            threadBarrier()
            for i in src.sites:
              if lo.coords[3][i].int == wallTime and
                  (lo.coords[0][i].int and 1) == (sourceCorner and 1) and
                  (lo.coords[1][i].int and 1) == ((sourceCorner shr 1) and 1) and
                  (lo.coords[2][i].int and 1) == ((sourceCorner shr 2) and 1):
                src{i}[sourceColor] := 1.0
        hitInfo["source-norm2"] = %src.norm2
        let expectedNorm = if sourceMode == "random-wall":
            3 * (lo.physVol div nt) else: lo.physVol div (nt * 8)
        if src.norm2 != expectedNorm.float:
          qexError "wall source norm is inconsistent with its support"
        echo sourceMode, " source ", hit, " color slot ", sourceColor,
            " time: ", wallTime, " norm2: ", expectedNorm
        cpuTimerStop(ctSourceGeneration, sourceStart)
        if cmd.hasKey("save-source"):
          let sourceTimingStart = cpuTimerStart()
          let sourceIoStart = epochTime()
          let requestedFile = cmd["save-source"].getStr()
          let parts = splitFile(requestedFile)
          let suffix = ".source" & align($hit, 4, '0') &
              (if sourceMode == "corner-wall": ".color" & $sourceColor else: "")
          let sourceFile = if props.len == 1: requestedFile
            else: parts.dir / (parts.name & suffix & parts.ext)
          # All ranks must enter QIO together; only rank 0 checks existence.
          var sourceExists: cint = 0
          if myRank == 0:
            sourceExists = cint(requestedFile.len == 0 or fileExists(sourceFile))
          QMP_broadcast(sourceExists.addr, sizeof(sourceExists).csize_t)
          if sourceExists != 0:
            qexError "source filename is empty or already exists: " & sourceFile
          commsBarrier()
          hitInfo["source-field"] = %sourceFile
          let metadata = $hitInfo
          var writer = lo.newWriter(sourceFile, metadata)
          if writer.status != 0: qexError "cannot open source file"
          writer.write(src, metadata, "D")
          if writer.status != 0: qexError "cannot write source file"
          writer.close()
          if writer.status != 0: qexError "cannot close source file"
          # Verify the saved source using the not-yet-solved propagator field.
          var reader = lo.newReader(sourceFile)
          if reader.status != 0: qexError "cannot reopen source file"
          reader.read(props[slot][srct][corner][0])
          if reader.status != 0 or reader.recordMetadata != metadata:
            qexError "cannot read saved source or metadata"
          reader.close()
          if reader.status != 0: qexError "cannot close saved source file"
          threads: props[slot][srct][corner][0] -= src
          if props[slot][srct][corner][0].norm2 != 0:
            qexError "saved source differs from wall"
          hitInfo["source-field-verified"] = %true
          echo "Source QIO write/read seconds: ", epochTime() - sourceIoStart
          cpuTimerStop(ctSourceIo, sourceTimingStart)
        sourceRecords.add(hitInfo)
      # invert D to get propagator
        for massIdx in 0..<Masses:
          cpuTimed(ctSolve):
            threads: props[slot][srct][corner][massIdx] := 0
            stag.solve(props[slot][srct][corner][massIdx], src, mass[massIdx], sp)
sourceInfo["sources"] = sourceRecords
# Preserve the single-source metadata fields for existing consumers.
if props.len == 1:
  for key in ["source-time", "source-norm2", "source-field", "source-field-verified"]:
    if sourceRecords[0].hasKey(key): sourceInfo[key] = sourceRecords[0][key]
writeSourceRecord(metadataFile, sourceInfo)
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
var sourceCorrJSON = newSeq[JsonNode](sourceCount)
for hit in 0..<sourceCount: sourceCorrJSON[hit] = %* {}

template measureSources(fu, ful: untyped; tauStr: string) =
  let contractionStart = cpuTimerStart()
  # Contract each source separately at its own time origin before averaging.
  for hit in 0..<sourceCount:
    sourceCorrJSON[hit][tauStr] = %* {}
    for propindex in 0..<mprops.len:
      let flavor = mprops[propindex]
      var c: RandomWallCorrelators
      for sourceColor in 0..<colorsPerTime:
        let slot = hit * colorsPerTime + sourceColor
        let component = randomWallCorrelators(u, fu, ful,
            props[slot][0][0][flavor[0]], props[slot][0][0][flavor[1]], wallTimes[hit])
        if sourceColor == 0:
          c = component
        else:
          for t in 0..<nt:
            c.pp.re[t] += component.pp.re[t]
            c.a0pThin.re[t] += component.a0pThin.re[t]
            c.a0pThin.im[t] += component.a0pThin.im[t]
            c.a0pConserved.re[t] += component.a0pConserved.re[t]
            c.a0pConserved.im[t] += component.a0pConserved.im[t]
      # These P-source bilinears apply to either wall. Undo the helper's
      # 1/Vs factor for the original corner-wall normalization; sum colors.
      if sourceMode == "corner-wall":
        let volume = (lo.physVol div nt).float
        for t in 0..<nt:
          c.pp.re[t] *= volume
          c.a0pThin.re[t] *= volume
          c.a0pThin.im[t] *= volume
          c.a0pConserved.re[t] *= volume
          c.a0pConserved.im[t] *= volume
      let channels = sourceCorrJSON[hit][tauStr]
      channels["PP_" & massname[propindex]] = %* c.pp.re
      channels["P-P_" & massname[propindex]] = %* c.pp.re.mapIt(-it)
      channels["A0P_thin_re_" & massname[propindex]] = %* c.a0pThin.re
      channels["A0P_thin_im_" & massname[propindex]] = %* c.a0pThin.im
      channels["cA0P_re_" & massname[propindex]] = %* c.a0pConserved.re
      channels["cA0P_im_" & massname[propindex]] = %* c.a0pConserved.im
  for channel, values in sourceCorrJSON[0][tauStr]:
    var mean = newSeq[float](values.len)
    for hit in 0..<sourceCount:
      for t in 0..<values.len:
        mean[t] += sourceCorrJSON[hit][tauStr][channel][t].getFloat()
    for t in 0..<mean.len: mean[t] /= sourceCount.float
    corrJSON[tauStr][channel] = %mean
  cpuTimerStop(ctContractions, contractionStart)

if corrFile != "__NOOUTPUT__":
  let tauStr = formatFloat(0, ffDecimal, 2)
  corrJSON[tauStr] = %* {}
  qexLog "Measuring the propagators at tau=0.0"
  block:
    let contractionLinksStart = cpuTimerStart()
    u.rephase()
    var fu = lo.newGauge()
    var ful = lo.newGauge()
    tic "HISQ gauge smearing for time-split correlators at tau=" & tauStr
    discard hisq.smearGetForce(u, fu, ful)

    u.rephase()
    cpuTimerStop(ctHisqLinksContractions, contractionLinksStart)
    qexLog "HISQ gauge smearing for time-split correlators done at tau=",
        tauStr, ". Time: ", getElapsedTime()

    measureSources(fu, ful, tauStr)

  qexGC("after measuring correlators at tau=0.00")
  
for flow in flowInfo.keys():
  flowInfo[flow]["filename"] = %* (flow & "_" & cfg & ".log")
echo "Starting fermion flow"
let flowMass = newSeq[float](mass.len)
echo "flowMass: ", flowMass
u.staggeredFlow(props,flowMass,flowInfo):
  let flowPrintStart = cpuTimerStart()
  echo "Flow at tau= ", tau
  echo "mass of the flow ", flowMass
  echo props[0][0][0][0][0][0] # print the first element of the first propagator as a sanity check to make sure the flow is actually doing something to the propagators
  cpuTimerStop(ctFlowDiagnostics, flowPrintStart)
  if gfobsFile != "__NOOUTPUT__":
    echo "saving gauge observables at tau= ", tau
    let tauStr = formatFloat(tau, ffDecimal, 2)
    gfobsJSON[tauStr] = measurements
  if abs(tau.round(1) - tau) < 1e-10: # if tau is 0.1, 0.2 ...
    if corrFile != "__NOOUTPUT__":
      echo "Measuring the propagators at tau= ", tau
      let tauStr = formatFloat(tau, ffDecimal, 2)
      corrJSON[tauStr] = %* {}
      block:
        let contractionLinksStart = cpuTimerStart()
        u.rephase()
        var fu = lo.newGauge()
        var ful = lo.newGauge()
        tic "HISQ gauge smearing for time-split correlators at tau=" & tauStr
        discard hisq.smearGetForce(u, fu, ful)
        
        u.rephase()
        cpuTimerStop(ctHisqLinksContractions, contractionLinksStart)
        qexLog "HISQ gauge smearing for time-split correlators done at tau=",
            tauStr, ". Time: ", getElapsedTime()
        
        measureSources(fu, ful, tauStr)

      qexGC("after measuring correlators at tau=" & tauStr)

echo "Saving correlators and gauge observables"
let outputStart = cpuTimerStart()
let gfobsStr = pretty(gfobsJSON)
let corrStr = pretty(corrJSON)
if myRank == 0:
  if gfobsFile != "__NOOUTPUT__": writeFile(gfobsFile, gfobsStr)
  if corrFile != "__NOOUTPUT__": writeFile(corrFile, corrStr)
  if corrFile != "__NOOUTPUT__" and sourceCount > 1:
    var individual = newJArray()
    for hit in 0..<sourceCount:
      var item = %* {"source-index": hit, "source-time": wallTimes[hit],
                    "correlators": sourceCorrJSON[hit]}
      if sourceMode == "corner-wall": item["source-corner"] = %wallCorners[hit]
      individual.add(item)
    writeFile(corrFile & ".sources.json", pretty(individual))
cpuTimerStop(ctOutputIo, outputStart)
writeSourceRecord(metadataFile, sourceInfo)


let timingReport = cpuTimingReport()
timingReport["configuration"] = %cfg
timingReport["source_mode"] = %sourceMode
timingReport["source_count"] = %sourceCount
timingReport["solution_fields"] = %props.len
timingReport["seed"] = %seed
let timingFile = if cmd.hasKey("output"):
    cmd["output"].getStr() / ("timing_" & flowname & "_" & cfg & ".json")
  else: "timing_" & flowname & "_" & cfg & ".json"
if myRank == 0:
  writeFile(timingFile, pretty(timingReport))
  echo "CPU timing report: ", timingFile

qexFinalize()
