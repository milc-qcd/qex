## CPU wall-clock accounting. Call timers outside OpenMP regions, on every rank.
## Regions must not overlap. No synchronization is added inside the calculation;
## only the final report reduces accumulated times across ranks.
import qex
import std/[monotimes, times, json, strutils]

type CpuPhase* = enum
  ctGaugeFlow, ctFermionFlow, ctContractions, ctSolve,
  ctHisqLinksFlow, ctHisqLinksSolve, ctHisqLinksContractions,
  ctGaugeMeasurements, ctFlowDiagnostics, ctSourceGeneration,
  ctSourceIo, ctGaugeInput, ctOutputIo

const phaseNames: array[CpuPhase, string] = [
  "gauge_flow", "fermion_flow", "contractions", "solve",
  "hisq_links_flow", "hisq_links_solve", "hisq_links_contractions",
  "gauge_measurements", "flow_diagnostics", "source_generation",
  "source_io", "gauge_input", "output_io"]
const phaseDescriptions: array[CpuPhase, string] = [
  "Gauge force, RK matrix exponentials/link updates, gauge work buffers, flow reunitarization; excludes HISQ construction and measurements",
  "All source fields: directional HISQ Laplacians, RK vector updates, and fermion work buffers; excludes HISQ construction",
  "All source/color contractions, normalization, time alignment, source averaging and correlator JSON assembly; excludes HISQ construction",
  "Zeroing each solution and stag.solve, including solver setup and communication; excludes HISQ construction and source generation/I/O",
  "HISQ construction and staggered-operator setup at all three RK stages, shared by all source fields",
  "HISQ construction and staggered-operator setup before the solves, shared by all source fields",
  "HISQ construction at each correlator measurement, shared by all source fields",
  "Gauge observables, including flowMeasurements and diagnostic plaquettes",
  "Diagnostic propagator/work-vector norms and flow-state printing",
  "Noise RNG field initialization and construction/validation of wall sources",
  "Source QIO existence check, write, readback and exact field verification",
  "Input gauge-file loading",
  "Observable/metadata serialization and file writes; excludes timing-report output"]

var phaseSeconds: array[CpuPhase, float64]
var phaseSegments: array[CpuPhase, int]
var calculationStart: MonoTime

proc resetCpuTimings*() =
  phaseSeconds = default(typeof(phaseSeconds))
  phaseSegments = default(typeof(phaseSegments))
  calculationStart = getMonoTime()

proc cpuTimerStart*(): MonoTime {.inline.} = getMonoTime()

proc cpuTimerStop*(phase: CpuPhase; started: MonoTime) {.inline.} =
  phaseSeconds[phase] += (getMonoTime() - started).inNanoseconds.float64 * 1e-9
  inc phaseSegments[phase]

template cpuTimed*(phase: CpuPhase; body: untyped) =
  block:
    let started = cpuTimerStart()
    body
    cpuTimerStop(phase, started)

proc rankTiming(seconds: float64; segments: int): JsonNode =
  let comm = getComm()
  var lowSeconds = seconds
  var highSeconds = seconds
  var sumSeconds = seconds
  var lowSegments = segments.float64
  var highSegments = segments.float64
  comm.allReduceMin(lowSeconds)
  comm.allReduceMax(highSeconds)
  comm.allReduce(sumSeconds)
  comm.allReduceMin(lowSegments)
  comm.allReduceMax(highSegments)
  result = %* {"rank_min_seconds": lowSeconds,
               "rank_mean_seconds": sumSeconds / nRanks.float64,
               "rank_max_seconds": highSeconds,
               "rank_min_segments": lowSegments.int,
               "rank_max_segments": highSegments.int}

proc cpuTimingReport*(): JsonNode =
  ## Capture every local interval before entering any reporting collectives.
  let totalSeconds = (getMonoTime() - calculationStart).inNanoseconds.float64 * 1e-9
  var measuredSeconds = 0.0
  for phase in CpuPhase: measuredSeconds += phaseSeconds[phase]
  let otherSeconds = totalSeconds - measuredSeconds
  if otherSeconds < -1e-6:
    qexError "CPU timing regions overlap"
  result = %* {
    "schema_version": 1, "units": "seconds", "clock": "monotonic wall clock",
    "mpi_ranks": nRanks,
    "scope": "after qexInit through ordinary output; excludes timing report and qexFinalize",
    "aggregation": "min/mean/max across ranks of accumulated local elapsed time; includes communication within each region; no per-region barriers",
    "accounting": "exclusive phases; sum of rank means including other equals total rank mean; maxima can belong to different ranks and must not be summed as wall time",
    "phases": {}, "total": rankTiming(totalSeconds, 1)}
  for phase in CpuPhase:
    let entry = rankTiming(phaseSeconds[phase], phaseSegments[phase])
    entry["description"] = %phaseDescriptions[phase]
    result["phases"][phaseNames[phase]] = entry
  result["phases"]["other"] = rankTiming(max(0.0, otherSeconds), 1)
  result["phases"]["other"]["description"] =
      %"Other setup, bookkeeping, logging, garbage collection and timer overhead"
  if myRank == 0:
    echo "CPU timings: accumulated wall seconds per rank (exclusive regions)"
    for name, entry in result["phases"]:
      echo "CPU_TIMING ", name, " mean_seconds=",
          formatFloat(entry["rank_mean_seconds"].getFloat(), ffDecimal, 6),
          " max_seconds=", formatFloat(entry["rank_max_seconds"].getFloat(), ffDecimal, 6),
          " segments=", entry["rank_max_segments"].getInt()
    echo "CPU_TIMING total mean_seconds=",
        formatFloat(result["total"]["rank_mean_seconds"].getFloat(), ffDecimal, 6),
        " max_seconds=", formatFloat(result["total"]["rank_max_seconds"].getFloat(), ffDecimal, 6)
