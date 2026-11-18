## CPU/GPU wall-clock accounting. Call timers outside OpenMP regions, on every rank.
## Regions must not overlap. No synchronization is added inside the calculation;
## only the final report reduces accumulated times across ranks.
import qex
import std/[monotimes, times, json, strutils]

const Backend {.strdefine.} = "CPU"
# SYCL onGpuQ waits before returning, as do both gpuMemCpy directions.
# QUDA returns its host solution before stag.solve returns. Therefore these
# intervals include completed work without additional queue/MPI barriers.

type FlowPhase* = enum
  ftGaugeFlow, ftFermionFlow, ftContractions, ftSolve,
  ftHisqLinksFlow, ftHisqLinksSolve, ftHisqLinksContractions,
  ftGaugeMeasurements, ftFlowDiagnostics, ftSourceGeneration,
  ftSourceIo, ftGaugeInput, ftOutputIo,
  ftFlowSetup, ftFlowDownload, ftSolveVerification, ftSolverCleanup

const phaseNames: array[FlowPhase, string] = [
  "gauge_flow", "fermion_flow", "contractions", "solve",
  "hisq_links_flow", "hisq_links_solve", "hisq_links_contractions",
  "gauge_measurements", "flow_diagnostics", "source_generation",
  "source_io", "gauge_input", "output_io",
  "flow_setup", "flow_download", "solve_verification", "solver_cleanup"]
const phaseDescriptions: array[FlowPhase, string] = [
  "Gauge force, RK matrix exponentials/link updates and flow reunitarization; excludes HISQ construction, measurements and persistent coupled-flow setup",
  "All source fields: directional HISQ Laplacians and RK vector updates, including communication; excludes HISQ construction and persistent coupled-flow setup",
  "All source/color contractions, normalization, time alignment, source averaging and correlator JSON assembly; excludes HISQ construction",
  "Zeroing each solution and stag.solve, including solver setup and communication; excludes HISQ construction and source generation/I/O",
  "HISQ construction and staggered-link preparation at all three RK stages, shared by all source fields; includes required transfers and communication",
  "HISQ construction and staggered-operator setup before the solves, shared by all source fields",
  "HISQ construction at each correlator measurement, shared by all source fields",
  "Gauge observables, including flowMeasurements and diagnostic plaquettes",
  "Diagnostic propagator/work-vector norms and flow-state printing",
  "Noise RNG field initialization and construction/validation of wall sources",
  "Source QIO existence check, write, readback and exact field verification",
  "Input gauge-file loading",
  "Observable/metadata serialization and file writes; excludes timing-report output",
  "Persistent coupled-flow workspaces, halo plans, allocation, initial host copies and cleanup; separate from the CPU fallback's per-step work buffers",
  "Explicit gauge/fermion downloads for measurements, including copying to caller fields; halo transfers and lazy uploads remain in the phase that requests them",
  "QEX check of the unpreconditioned true residual after each QUDA solve, including reductions",
  "QUDA backend shutdown after all solves"]

var phaseSeconds: array[FlowPhase, float64]
var phaseSegments: array[FlowPhase, int]
var calculationStart: MonoTime

proc resetFlowTimings*() =
  phaseSeconds = default(typeof(phaseSeconds))
  phaseSegments = default(typeof(phaseSegments))
  calculationStart = getMonoTime()

proc flowTimerStart*(): MonoTime {.inline.} = getMonoTime()

proc flowTimerStop*(phase: FlowPhase; started: MonoTime) {.inline.} =
  phaseSeconds[phase] += (getMonoTime() - started).inNanoseconds.float64 * 1e-9
  inc phaseSegments[phase]

template flowTimed*(phase: FlowPhase; body: untyped) =
  block:
    let started = flowTimerStart()
    body
    flowTimerStop(phase, started)

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

proc flowTimingReport*(): JsonNode =
  ## Capture every local interval before entering any reporting collectives.
  let totalSeconds = (getMonoTime() - calculationStart).inNanoseconds.float64 * 1e-9
  var measuredSeconds = 0.0
  for phase in FlowPhase: measuredSeconds += phaseSeconds[phase]
  let otherSeconds = totalSeconds - measuredSeconds
  if otherSeconds < -1e-6:
    qexFatal "Flow timing regions overlap"
  result = %* {
    "schema_version": 1, "units": "seconds", "clock": "monotonic wall clock",
    "mpi_ranks": nRanks, "backend": Backend,
    "flow_implementation": (when defined(hisqFlowGpu): "coupled" else: "cpu"),
    "device_completion": (if Backend == "SYCL":
      "onGpuQ and gpuMemCpy wait before return; QUDA host solution complete at solve return; no extra synchronization"
      else: "synchronous CPU backend; no extra synchronization"),
    "scope": "after qexInit through ordinary output; excludes timing report and qexFinalize",
    "aggregation": "min/mean/max across ranks of accumulated local elapsed time; includes communication within each region; no per-region barriers",
    "accounting": "exclusive phases; sum of rank means including other equals total rank mean; maxima can belong to different ranks and must not be summed as wall time",
    "phases": {}, "total": rankTiming(totalSeconds, 1)}
  for phase in FlowPhase:
    let entry = rankTiming(phaseSeconds[phase], phaseSegments[phase])
    entry["description"] = %phaseDescriptions[phase]
    result["phases"][phaseNames[phase]] = entry
  result["phases"]["other"] = rankTiming(max(0.0, otherSeconds), 1)
  result["phases"]["other"]["description"] =
      %"Other setup, bookkeeping, logging, garbage collection and timer overhead"
  if myRank == 0:
    echo "Flow timings: accumulated wall seconds per rank (exclusive regions)"
    for name, entry in result["phases"]:
      echo "FLOW_TIMING ", name, " mean_seconds=",
          formatFloat(entry["rank_mean_seconds"].getFloat(), ffDecimal, 6),
          " max_seconds=", formatFloat(entry["rank_max_seconds"].getFloat(), ffDecimal, 6),
          " segments=", entry["rank_max_segments"].getInt()
    echo "FLOW_TIMING total mean_seconds=",
        formatFloat(result["total"]["rank_mean_seconds"].getFloat(), ffDecimal, 6),
        " max_seconds=", formatFloat(result["total"]["rank_max_seconds"].getFloat(), ffDecimal, 6)
