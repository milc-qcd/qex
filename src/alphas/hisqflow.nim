import qex
#import examples/[hisqhmc_h]
import hisqflowutils #hisqtestutils
import hisqsolver
import flowTimings
import quda/qudaWrapper
import json

import gauge/[gaugefix, hisqsmear]
import physics/[stagSolve]
import observables/[nlStagMeson]
import conservedA0#nltStagMeson
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
  ## Uniform integer without modulo bias. The scalar stream is rank independent.
  if extent <= 0 or extent > rng.numInts:
    qexError "source extent is outside the RNG integer range"
  let limit = rng.numInts - (rng.numInts mod extent)
  while true:
    let value = rng.integer
    if value < limit:
      return value mod extent

proc sourceRelative(values: seq[float]; sourceTime: int): seq[float] =
  ## Keep every output channel indexed by dt = (sink time - source time) mod Nt.
  result = newSeq[float](values.len)
  for dt in 0..<values.len:
    result[dt] = values[(sourceTime + dt) mod values.len]


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


proc selectSource(cmd: JsonNode): string =
  result = "corner-wall"
  if cmd.hasKey("source"):
    if cmd["source"].kind != JString:
      qexError "--source must be corner-wall or random-wall"
    result = cmd["source"].getStr()
  if result notin ["corner-wall", "random-wall"]:
    qexError "--source must be corner-wall or random-wall"
  if result == "random-wall" and cmd.hasKey("source-corner"):
    qexError "--source-corner applies only to corner-wall sources"
  if cmd.hasKey("save-source"):
    if cmd["save-source"].kind != JString or cmd["save-source"].getStr().len == 0:
      qexError "--save-source requires a filename"

proc writeSourceRecord(path: string; metadata: JsonNode; initial = false) =
  let recordStart = flowTimerStart()
  defer: flowTimerStop(ftOutputIo, recordStart)
  if myRank == 0:
    # qexError includes a collective barrier, so a rank-local I/O error
    # must use qexFatal to avoid leaving other ranks waiting in a reduction.
    try:
      let directory = parentDir(path)
      if directory.len > 0: createDir(directory)
      if initial and fileExists(path):
        qexFatal "source record already exists; use a new output directory: " & path
      writeFile(path, pretty(metadata) & "\n")
    except CatchableError as error:
      qexFatal "cannot write source record: " & error.msg

proc saveWallSource(src, check: auto; path: string; metadata: var JsonNode) =
  let sourceIoStart = flowTimerStart()
  defer: flowTimerStop(ftSourceIo, sourceIoStart)
  if myRank == 0:
    try:
      if fileExists(path):
        qexFatal "source field already exists: " & path
      let directory = parentDir(path)
      if directory.len > 0: createDir(directory)
    except CatchableError as error:
      qexFatal "cannot prepare source field output: " & error.msg
  commsBarrier()
  metadata["source-field"] = %path
  let record = $metadata
  var writer = src.l.newWriter(path, record)
  if writer.status != 0: qexFatal "cannot open source field"
  writer.write(src, record, "D")
  if writer.status != 0: qexFatal "cannot write source field"
  writer.close()
  if writer.status != 0: qexFatal "cannot close source field"
  var reader = src.l.newReader(path)
  if reader.status != 0: qexFatal "cannot reopen source field"
  reader.read(check)
  if reader.status != 0 or reader.recordMetadata != record:
    qexFatal "cannot read saved source or metadata"
  reader.close()
  if reader.status != 0: qexFatal "cannot close saved source field"
  threads: check -= src
  if check.norm2 != 0:
    qexError "saved source differs from the wall"
  metadata["source-field-verified"] = %true

proc randomWallMeasurements(u, fu, ful, props: auto; sourceTime: int): JsonNode =
  # Host contractions; all placements share the same smeared gauge links.
  let c = randomWallCorrelatorsFromProps(u, fu, ful, props, sourceTime)
  result = %* {
    "PP_ss": c.pp.re,
    "P-P_ss": c.pp.re.mapIt(-it),
    "A0P_thin_re_ss": c.a0pThin.re,
    "A0P_thin_im_ss": c.a0pThin.im,
    "cA0P_re_ss": c.a0pConserved.re,
    "cA0P_im_ss": c.a0pConserved.im
  }

qexInit()
resetFlowTimings()

var
  #reads command line arguments into a json object
  cmd = readCMD()
  solverBackend = selectHisqSolver(
    if cmd.hasKey("solver"): cmd["solver"].getStr() else: "qex")
  sourceMode = selectSource(cmd)
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
flowTimed(ftGaugeInput): u.readGauge(filename & "." & cfg & ".ildg")
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
    # time extent
  nt = latLayout[^1]

# Match the CPU hisqflowwall source scheme. Time/corner draws use separate
# scalar streams; site-noise streams are independent of MPI decomposition.
if latLayout.len != 4 or latLayout.anyIt(it <= 0 or (it and 1) != 0):
  qexError "HISQ wall sources require four positive even lattice extents"
for option in ["seed", "sources", "source-time", "source-corner"]:
  if cmd.hasKey(option) and cmd[option].kind != JInt:
    qexError "--" & option & " requires an integer"
var sourceCount = if cmd.hasKey("sources"): cmd["sources"].getInt() else: 1
if sourceCount <= 0 or sourceCount > high(int) div 3:
  qexError "--sources must be a positive integer with a representable field count"
var explicitTimes: seq[int]
if cmd.hasKey("source-times"):
  if cmd.hasKey("source-time"):
    qexError "use only one of --source-time and --source-times"
  var values: JsonNode
  try:
    values = parseJson(cmd["source-times"].getStr())
  except CatchableError:
    qexError "--source-times must be a nonempty JSON array of integers"
  if values.kind != JArray or values.len == 0:
    qexError "--source-times must be a nonempty JSON array of integers"
  for value in values:
    if value.kind != JInt:
      qexError "--source-times must be a nonempty JSON array of integers"
    explicitTimes.add value.getInt()
  if cmd.hasKey("sources") and sourceCount != explicitTimes.len:
    qexError "--sources must equal the length of --source-times"
  sourceCount = explicitTimes.len
elif cmd.hasKey("source-time"):
  if sourceCount != 1:
    qexError "--source-time requires one source; use --source-times for a list"
  explicitTimes = @[cmd["source-time"].getInt()]
for t in explicitTimes:
  if t < 0 or t >= nt:
    qexError "--source-time / --source-times entries must be in 0..<Nt"
if cmd.hasKey("source-corner") and cmd["source-corner"].getInt() notin 0..7:
  qexError "--source-corner must be in 0..7 (x + 2*y + 4*z)"
var sourceSeed: uint64
if cmd.hasKey("seed"):
  let value = cmd["seed"].getInt()
  if value < 0 or value.uint64 > high(uint32).uint64:
    qexError "--seed must be an integer in 0..4294967295"
  sourceSeed = value.uint64
elif myRank == 0:
  sourceSeed = uint64(epochTime() * 1e6) and high(uint32).uint64
QMP_broadcast(sourceSeed.addr, sizeof(sourceSeed).csize_t)
var timeRng, cornerRng: RngMilc6
timeRng.seedIndep(sourceSeed, lo.physVol)
cornerRng.seedIndep(sourceSeed, lo.physVol + 1)
var wallTimes, wallCorners: seq[int]
for hit in 0..<sourceCount:
  let drawnTime = drawSourceIndex(timeRng, nt)
  wallTimes.add(if explicitTimes.len > 0: explicitTimes[hit] else: drawnTime)
  if sourceMode == "corner-wall":
    let drawnCorner = drawSourceIndex(cornerRng, 8)
    wallCorners.add(if cmd.hasKey("source-corner"):
      cmd["source-corner"].getInt() else: drawnCorner)
let sourceRecord = if corrFile != "__NOOUTPUT__": corrFile & ".source.json"
                   else: "source_" & flowname & "_" & cfg & ".json"
let fieldsPerTime = if sourceMode == "random-wall": 1 else: 3
var sourceMetadata = %* {
  "source-kind": sourceMode,
  "seed": sourceSeed,
  "source-count": sourceCount,
  "source-times": wallTimes,
  "source-time-fixed": explicitTimes.len > 0,
  "source-time-range": [0, nt-1],
  "time-sampling": (if explicitTimes.len > 0: "explicit list" else: "uniform with replacement"),
  "source-average": "arithmetic mean after aligning each source time",
  "rng-scheme": "RngMilc6 wall v1; times=volume; corners=volume+1; noise=global sites, sequential Z4",
  "lattice-geometry": latLayout,
  "configuration": cfg,
  "gauge-file": filename & "." & cfg & ".ildg",
  "mass": mass,
  "solver": (if solverBackend == sbQuda: "quda" else: "qex"),
  "source-colors": 3,
  "inversions-per-time": fieldsPerTime,
  "inversions-per-flavor": fieldsPerTime * sourceCount,
  "source-operator": "P = epsilon * barChi * chi",
  "source-draw": "RngMilc6 stream lattice-volume; uniform rejection sampling",
  "time-index": "dt = (sink time - source time + Nt) mod Nt",
  "A0P-time-location": "forward current at dt + 1/2"
}
if sourceCount == 1: sourceMetadata["source-time"] = %wallTimes[0]
if sourceMode == "random-wall":
  sourceMetadata["source"] = %"Z4 full-color random wall"
  sourceMetadata["noise-draw"] = %"RngMilc6 global-site streams; seed once, advance three Z4 draws per site per hit, including off-wall sites"
  sourceMetadata["source-hits"] = %sourceCount
  sourceMetadata["source-norm2"] = %(3 * (lo.physVol div nt))
  sourceMetadata["normalization"] = %"divide by spatial volume; color trace is summed"
  sourceMetadata["current-component"] = %"forward only; conserved current includes Naik"
else:
  sourceMetadata["source"] = %"color-diluted corner wall"
  sourceMetadata["source-corners"] = %wallCorners
  sourceMetadata["source-corner-fixed"] = %cmd.hasKey("source-corner")
  sourceMetadata["source-corner-range"] = %*[0, 7]
  sourceMetadata["corner-draw"] = %"RngMilc6 stream lattice-volume+1; uniform rejection sampling"
  sourceMetadata["source-norm2-per-color"] = %(lo.physVol div (nt * 8))
  sourceMetadata["normalization"] = %"legacy color-summed corner-wall normalization"
  if sourceCount == 1:
    sourceMetadata["source-corner"] = %wallCorners[0]
    sourceMetadata["source-origin"] = %*[px(wallCorners[0]), py(wallCorners[0]),
                                       pz(wallCorners[0]), wallTimes[0]]
echo sourceMode, " seed=", sourceSeed, " times=", wallTimes
if sourceMode == "corner-wall": echo "source corners=", wallCorners
writeSourceRecord(sourceRecord, sourceMetadata, initial = true)

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
flowTimed(ftGaugeMeasurements): u.echoPlaq
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
sp.backend = solverBackend
qexLog "HISQ solver backend: ", (if sp.backend == sbQuda: "quda" else: "qex")

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
proc cornerWallMeasurements(u, fu, ful, props: auto;
    sourceTime, srcCorner: int): JsonNode =
  let
    lo = u[0].l
    nt = lo.physGeom[3]
    colors = props.len
    mprops = [[0, 0]]
  var corrs = newSeqWith(mprops.len,
      newSeqWith(mesons.len + nltcorrList.len, newSeq[float](nt)))
  result = newJObject()
  for color in 0..<colors:
    for pionIdx in 0..<mesons.len:
      for termIdx, term in mesons[pionIdx]:
        for forward in 0..<Corners:
          # echo "forward ", forward, "term.delta ", term.delta
          let
            backward = shiftIdx(forward, term.delta)
            base = [px(srcCorner xor forward), py(srcCorner xor forward),
                pz(srcCorner xor forward), sourceTime]
          # echo "forward ", forward, " backward ", backward
          var tmp = lo.ColorVector()

              # Build shifted + rephased copy of `props[backward]`.
          for propindex in 0..<mprops.len:
            shiftAndRephase(tmp, props[color][0][backward][mprops[propindex][0]], term.delta, 15 - term.phase, base)

                # Correlator for this pair, binned by timeslice
            let corr = dotByTimeslice(props[color][0][forward][mprops[propindex][1]], tmp)
            for t in 0..<nt: corrs[propindex][pionIdx][t] -= term.factor.float * corr[t]
  # su/sul already include APBC. Add it once to the unphased thin link.
  var thinTemporal = lo.ColorMatrix()
  threads:
    thinTemporal := u[3]
    for i in thinTemporal.sites:
      if lo.coords[3][i].int == nt-1:
        thinTemporal{i} *= -1

  for propindex in 0..<mprops.len:
    for pionIdx in 0..<mesons.len:
      let combinedName = mesonname[pionIdx] & "_" & massname[propindex]
      result[combinedName] = %* sourceRelative(corrs[propindex][pionIdx], sourceTime)

    # The A0-P helpers use stored source slot 0, independent of the sampled
    # position. For this local P source their contraction is -S^dagger U S+;
    # its source phases cancel. Physical time enters sourceRelative below.
    for nltindex in 0..<nltcorrList.len:
      let
        flavor = @[mprops[propindex][0], mprops[propindex][1],
            mprops[propindex][0], mprops[propindex][1]]
        thinRe = thinTemporal.buildMesonCorrelator(nltcorrList[nltindex][0],
            nltcorrList[nltindex][1], props, [0, 0], flavor, true, true)
        thinIm = thinTemporal.buildMesonCorrelator(nltcorrList[nltindex][0],
            nltcorrList[nltindex][1], props, [0, 0], flavor, true, false)
        consRe = buildConservedA0PCorrelator(u, fu, ful, props, flavor, true)
        consIm = buildConservedA0PCorrelator(u, fu, ful, props, flavor, false)
      result["A0P_thin_re_" & massname[propindex]] = %* sourceRelative(thinRe, sourceTime)
      result["A0P_thin_im_" & massname[propindex]] = %* sourceRelative(thinIm, sourceTime)
      result["cA0P_re_" & massname[propindex]] = %* sourceRelative(consRe, sourceTime)
      result["cA0P_im_" & massname[propindex]] = %* sourceRelative(consIm, sourceTime)

  # Keep the legacy mesons and add the same positive-PP aliases as the CPU
  # wall driver. pi_5 includes the relative temporal phase (-1)^dt.
  var pp = newSeq[float](nt)
  for dt in 0..<nt:
    pp[dt] = (if (dt and 1) == 0: -1.0 else: 1.0) *
        result["pi_5_ss"][dt].getFloat()
  result["PP_ss"] = %pp
  result["P-P_ss"] = %pp.mapIt(-it)

proc hisq2WallMeasurements(fu2, props: auto; sourceTime: int): JsonNode =
  let mprops = [[0, 0]]
  result = newJObject()
  for propindex in 0..<mprops.len:
    for nltindex in 0..<nltcorrList.len:
      let
        flavor = @[mprops[propindex][0], mprops[propindex][1],
            mprops[propindex][0], mprops[propindex][1]]
        hisqRe = fu2[3].buildMesonCorrelator(nltcorrList[nltindex][0],
            nltcorrList[nltindex][1], props, [0, 0], flavor, true, true)
        hisqIm = fu2[3].buildMesonCorrelator(nltcorrList[nltindex][0],
            nltcorrList[nltindex][1], props, [0, 0], flavor, true, false)
      result["A0P_hisq_re_" & massname[propindex]] = %* sourceRelative(hisqRe, sourceTime)
      result["A0P_hisq_im_" & massname[propindex]] = %* sourceRelative(hisqIm, sourceTime)
      result["A0P_re_" & massname[propindex]] = %* sourceRelative(hisqRe, sourceTime)
      result["A0P_im_" & massname[propindex]] = %* sourceRelative(hisqIm, sourceTime)

template runSpectrumAndFlow(colors: static[int]) =
  block:
    # initialize propagators and correlators
    var
      # initilize the propagators:
      props = newSeq[array[nsrcTime, array[Corners, array[Masses, typeof(lo.ColorVector)]]]](sourceCount * colors)
    for slot in 0..<props.len:
      for srct in 0..<nsrcTime:
        for massIdx in 0..<Masses:
          for corner in 0..<Corners: props[slot][srct][corner][massIdx] = lo.ColorVector()
    var sourceRecords = newJArray()
    tic "HISQ Dirac inversion"
    # get spectrum and do it at each color
    block:
      # smear and rephase
      let solveLinksStart = flowTimerStart()
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
      flowTimerStop(ftHisqLinksSolve, solveLinksStart)
      when colors == 1:
        # Advance the same site streams once per hit, including off-wall sites.
        let sourceRngStart = flowTimerStart()
        var rng = newRNGField(RngMilc6, lo, sourceSeed)
        flowTimerStop(ftSourceGeneration, sourceRngStart)
      for slot in 0..<props.len:
        let hit = slot div colors
        let color = slot mod colors
        let sourceTime = wallTimes[hit]
        let srcCorner = if colors == 3: wallCorners[hit] else: 0
        for srct in 0..<nsrcTime:
          for corner in 0..<Corners:
            let sourceStart = flowTimerStart()
            var src = lo.ColorVector()
            var fieldMetadata = sourceMetadata.copy()
            fieldMetadata["source-index"] = %hit
            fieldMetadata["source-time"] = %sourceTime
            when colors == 3:
              fieldMetadata["source-color"] = %color
              fieldMetadata["source-corner"] = %srcCorner
            when colors == 1:
              threads:
                src.z4 rng
                threadBarrier()
                for i in src.sites:
                  if lo.coords[3][i].int != sourceTime:
                    src{i} := 0
              let sourceNorm2 = src.norm2
              if sourceNorm2 != float(3 * (lo.physVol div nt)):
                qexError "random-wall source norm is inconsistent with its support"
              echo "Random-wall norm2=", sourceNorm2
            else:
              threads:
                let cc = lo.coords
                src := 0
                threadBarrier()
                for i in src.sites:
                  if cc[3][i].int == sourceTime and
                    (cc[0][i].int and 1) == px(srcCorner xor corner) and
                    (cc[1][i].int and 1) == py(srcCorner xor corner) and
                    (cc[2][i].int and 1) == pz(srcCorner xor corner):
                    src{i}[color] := 1.0
              let sourceNorm2 = src.norm2
              let expectedNorm2 = float(lo.physVol div (nt * 8))
              if sourceNorm2 != expectedNorm2:
                qexError "corner-wall source norm is inconsistent with its support"
              echo "Corner-wall color=", color, " norm2=", sourceNorm2
            fieldMetadata["source-norm2"] = %src.norm2
            flowTimerStop(ftSourceGeneration, sourceStart)
            if cmd.hasKey("save-source"):
              let requested = absolutePath(cmd["save-source"].getStr())
              let parts = splitFile(requested)
              let suffix = ".source" & align($hit, 4, '0') &
                  (if colors == 3: ".color" & $color else: "")
              let path = if props.len == 1: requested
                         else: parts.dir / (parts.name & suffix & parts.ext)
              saveWallSource(src, props[slot][srct][corner][0], path, fieldMetadata)
            sourceRecords.add(fieldMetadata)
          # invert D to get propagator
            for massIdx in 0..<Masses:
              flowTimed(ftSolve):
                threads: props[slot][srct][corner][massIdx] := 0
                stag.solve(props[slot][srct][corner][massIdx], src, mass[massIdx], sp)
              if sp.backend == sbQuda:
                let verificationStart = flowTimerStart()
                # Verify the original, unpreconditioned equation with QEX's D.
                # This also catches a failed inner solve or a normalization mismatch.
                var check = src.newOneOf
                threads:
                  stag.D(check, props[slot][srct][corner][massIdx], mass[massIdx])
                  threadBarrier()
                  check := src - check
                let trueR2 = check.norm2 / src.norm2
                qexLog "QUDA true relative residual squared: ", trueR2
                if (cast[uint64](trueR2) and 0x7ff0000000000000'u64) == 0x7ff0000000000000'u64 or
                    trueR2 > 1.01*sp.r2req:
                  qexError "QUDA HISQ solve did not satisfy the requested true residual"
                flowTimerStop(ftSolveVerification, verificationStart)
    sourceMetadata["sources"] = sourceRecords
    if props.len == 1:
      for key in ["source-field", "source-field-verified"]:
        if sourceRecords[0].hasKey(key):
          sourceMetadata[key] = sourceRecords[0][key]
    writeSourceRecord(sourceRecord, sourceMetadata)
    if sp.backend == sbQuda:
      flowTimed(ftSolverCleanup): qudaShutdown()
    qexGC("after HISQ Dirac inversion")
    u.rephase() #phase out

    var gfobsJSON = newJObject()
    if gfobsFile != "__NOOUTPUT__":
      for flow, flowin in flowInfo:
        let loops = if flowin.hasKey("loops"): flowin["loops"].getInt() else: 1
        gfobsJSON["0.00"] = u.flowMeasurements(loops, 0)
        break
    var corrJSON = newJObject()
    var sourceCorrJSON = newSeq[JsonNode](sourceCount)
    for hit in 0..<sourceCount: sourceCorrJSON[hit] = newJObject()

    template measureSources(tauStr: string) =
      block:
        # Smear the shared gauge once per measurement, not once per source.
        let contractionLinksStart = flowTimerStart()
        var fu = lo.newGauge()
        var ful = lo.newGauge()
        u.rephase()
        discard hisq.smearGetForce(u, fu, ful)
        u.rephase()
        flowTimerStop(ftHisqLinksContractions, contractionLinksStart)
        let contractionStart = flowTimerStart()
        for hit in 0..<sourceCount:
          var wallProps: array[colors, typeof(props[0])]
          for color in 0..<colors: wallProps[color] = props[hit * colors + color]
          when colors == 1:
            sourceCorrJSON[hit][tauStr] =
                randomWallMeasurements(u, fu, ful, wallProps, wallTimes[hit])
          else:
            sourceCorrJSON[hit][tauStr] =
                cornerWallMeasurements(u, fu, ful, wallProps, wallTimes[hit], wallCorners[hit])
        flowTimerStop(ftContractions, contractionStart)
      when colors == 3:
        block:
          let contractionLinksStart = flowTimerStart()
          var fu2 = lo.newGauge()
          var ful2 = lo.newGauge()
          u.rephase()
          discard hisq2.smearGetForce(u, fu2, ful2)
          u.rephase()
          flowTimerStop(ftHisqLinksContractions, contractionLinksStart)
          let contractionStart = flowTimerStart()
          for hit in 0..<sourceCount:
            var wallProps: array[colors, typeof(props[0])]
            for color in 0..<colors: wallProps[color] = props[hit * colors + color]
            let extra = hisq2WallMeasurements(fu2, wallProps, wallTimes[hit])
            for channel, values in extra: sourceCorrJSON[hit][tauStr][channel] = values
          flowTimerStop(ftContractions, contractionStart)
      let averagingStart = flowTimerStart()
      corrJSON[tauStr] = newJObject()
      for channel, values in sourceCorrJSON[0][tauStr]:
        var mean = newSeq[float](nt)
        for hit in 0..<sourceCount:
          for t in 0..<nt:
            mean[t] += sourceCorrJSON[hit][tauStr][channel][t].getFloat()
        for t in 0..<nt: mean[t] /= sourceCount.float
        corrJSON[tauStr][channel] = %mean
      flowTimerStop(ftContractions, averagingStart)
      qexGC("after wall measurements at tau=" & tauStr)

    if corrFile != "__NOOUTPUT__": measureSources("0.00")
    for flow in flowInfo.keys():
      flowInfo[flow]["filename"] = %* (flow & "_" & cfg & ".log")
    let flowMass = newSeq[float](mass.len)
    flowTimed(ftFlowDiagnostics): echo "Starting fermion flow for ", props.len, " source fields"
    u.staggeredFlow(props, flowMass, flowInfo):
      let tauStr = formatFloat(tau, ffDecimal, 2)
      if gfobsFile != "__NOOUTPUT__": gfobsJSON[tauStr] = measurements
      if corrFile != "__NOOUTPUT__" and abs(tau.round(1) - tau) < 1e-10:
        measureSources(tauStr)

    let outputStart = flowTimerStart()
    if myRank == 0:
      if gfobsFile != "__NOOUTPUT__": writeFile(gfobsFile, pretty(gfobsJSON))
      if corrFile != "__NOOUTPUT__":
        writeFile(corrFile, pretty(corrJSON))
        if sourceCount > 1:
          var individual = newJArray()
          for hit in 0..<sourceCount:
            var record = %* {"source-index": hit, "source-time": wallTimes[hit],
                             "correlators": sourceCorrJSON[hit]}
            when colors == 3: record["source-corner"] = %wallCorners[hit]
            individual.add(record)
          writeFile(corrFile & ".sources.json", pretty(individual))
    flowTimerStop(ftOutputIo, outputStart)

if sourceMode == "random-wall":
  runSpectrumAndFlow(1)
else:
  runSpectrumAndFlow(3)

let timingReport = flowTimingReport()
timingReport["configuration"] = %cfg
timingReport["source_mode"] = %sourceMode
timingReport["source_count"] = %sourceCount
timingReport["solution_fields"] = %(sourceCount * fieldsPerTime)
timingReport["seed"] = %sourceSeed
timingReport["solver"] = %(if solverBackend == sbQuda: "quda" else: "qex")
let timingFile = if cmd.hasKey("output"):
    cmd["output"].getStr() / ("timing_" & flowname & "_" & cfg & ".json")
  else: "timing_" & flowname & "_" & cfg & ".json"
if myRank == 0:
  try:
    writeFile(timingFile, pretty(timingReport) & "\n")
    echo "Flow timing report: ", timingFile
  except CatchableError as error:
    qexFatal "cannot write timing report: " & error.msg

qexFinalize()
