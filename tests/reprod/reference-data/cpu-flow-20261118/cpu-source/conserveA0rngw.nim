## Random-wall PP, thin-link A0P, and conserved HISQ A0P contractions.
##
## When imported, the main driver supplies:
##   * left/right propagator fields from the SAME random wall (masses may differ);
##   * ordinary thin gauge links u (without fermion boundary signs by default);
##   * coefficient-weighted HISQ one-link su and three-link sul, including BC;
##   * sourceTime, the actual Euclidean time of that wall, in 0..<Nt.
## Already-flowed propagators can be passed at each measurement flow time.
##
## The source operator is always P(y) = epsilon(y) barChi(y) chi(y).
## Epsilon hermiticity cancels the source epsilon and the epsilon at the
## unshifted current endpoint. No corner-dependent rephasing or extra sink
## epsilon is applied to the propagator inner product. This also applies to
## the parity-preserving HISQ D_mu^dagger D_mu fermion flow.
##
## PP uses the positive norm convention. Both A0P outputs retain
## conservedA0.nim's Wick minus sign and FORWARD current only, with W = 1.
## The backward-current contribution is not included. The older spatial-phase
## P source (mask 7) differs from the epsilon P source by (-1)^sourceTime.
##
## Returned arrays are divided by the spatial volume and indexed by
## dt = (sinkTime - sourceTime + Nt) mod Nt. Current bins denote dt + 1/2.
## sourceTime changes these bins, not the contraction's epsilon factors.
##
## Single wall:
##   let c = randomWallCorrelators(u, su, sul, prop1, prop2, sourceTime)
##   # c.pp, c.a0pThin, c.a0pConserved each have .re and .im time arrays.
## Existing main-driver storage:
##   let c = randomWallCorrelatorsFromProps(u, su, sul, props, sourceTime,
##                                          flavor = [0, 0], sourceSlot = 0)
## The leading props index enumerates disjoint dilution components to SUM,
## not independent noise hits to average. For one undiluted wall its length
## is one. Store the full spatial wall in corner slot zero.
## Set thinLinksHaveBC=true if u already includes fermion boundary signs.
## antiperiodicTime defaults to true and controls adding those signs to u;
## su/sul must already carry the chosen boundary conditions in either case.

## Compiled directly, this file runs a standalone random-wall measurement:
##   conserveA0rngw --unit-gauge --lattice-json:lattice.json --seed:2 \
##     --mass:0.2 --output:randomwall.json
## Replace --unit-gauge with --configuration:FILE to read a gauge field.
## --help lists solver/JSON options. One full-color Z4 wall is drawn uniformly
## from all Nt source times, and one inversion measures PP and both A0P forms.
## The seed, source time, normalization, and checked residual are saved in JSON.
## --save-source:FILE also saves the actual Z4 field in double-precision QIO
## format and checks an exact read-back before solving.

import qex

type
  WallCorrelator* = object
    re*, im*: seq[float]

proc checkWallInputs(left, right: auto; sourceTime: int) =
  if left.l != right.l:
    qexError "random-wall propagators must use the same layout"
  if left.l.physGeom.len != 4:
    qexError "random-wall correlators require a four-dimensional lattice"
  if sourceTime < 0 or sourceTime >= left.l.physGeom[3]:
    qexError "sourceTime must be the actual source time in 0..<Nt"

proc dotByWallTime(left, right: auto): WallCorrelator =
  ## Call outside a threads block; visit all local sites, then reduce ranks.
  let nt = left.l.physGeom[^1]
  result.re = newSeq[float](nt)
  result.im = newSeq[float](nt)
  for i in 0..<left.l.nSites:
    let t = left.l.coords[3][i].int
    # Materialize scalar components: dot() on the old QEX complex proxies
    # can retain adjoint temporaries and silently return zero.
    for color in 0..<left[0].len:
      var ar, ai, br, bi: float
      ar := left{i}[color].re
      ai := left{i}[color].im
      br := right{i}[color].re
      bi := right{i}[color].im
      result.re[t] += ar*br + ai*bi
      result.im[t] += ar*bi - ai*br
  result.re.rankSum
  result.im.rankSum

proc relativeWallTime(c: WallCorrelator; sourceTime: int;
                      normalization: float): WallCorrelator =
  let nt = c.re.len
  result.re = newSeq[float](nt)
  result.im = newSeq[float](nt)
  for dt in 0..<nt:
    let t = (sourceTime + dt) mod nt
    result.re[dt] = c.re[t] * normalization
    result.im[dt] = c.im[t] * normalization

proc wallSpatialVolume(f: auto): int =
  f.l.physGeom[0] * f.l.physGeom[1] * f.l.physGeom[2]

proc randomWallPP*(left, right: auto; sourceTime: int): WallCorrelator =
  ## Positive PP norm convention; same P-source wall on both legs.
  checkWallInputs(left, right, sourceTime)
  result = relativeWallTime(dotByWallTime(left, right), sourceTime,
                            1.0 / wallSpatialVolume(left).float)

proc randomWallA0PThin*(u0: auto; left, right: auto; sourceTime: int;
                       thinLinksHaveBC: bool = false;
                       antiperiodicTime: bool = true): WallCorrelator =
  ## Thin-link <A0_forward P>. u0 is the temporal gauge field u[3].
  ## Ordinary gauge links do not contain the fermion boundary sign.
  checkWallInputs(left, right, sourceTime)
  if u0.l != left.l:
    qexError "thin link and random-wall propagators must use the same layout"
  let
    lo = left.l
    nt = lo.physGeom[3]
    addBoundarySign = antiperiodicTime and not thinLinksHaveBC
  var linked = lo.ColorVector()
  var sh = newShifter(right, 3, 1)
  discard sh ^* right
  threads:
    for i in linked.sites:
      linked{i} := u0{i} * sh.field{i}
      if addBoundarySign and lo.coords[3][i].int == nt - 1:
        linked{i} *= -1
  result = relativeWallTime(dotByWallTime(left, linked), sourceTime,
                            -1.0 / wallSpatialVolume(left).float)

proc randomWallA0PForward*(su, sul: auto; left, right: auto;
                          sourceTime: int): WallCorrelator =
  ## <A0_forward P>, including the HISQ Naik term. No additional epsilon.
  ## su and sul are the phased, coefficient-weighted solver links.
  checkWallInputs(left, right, sourceTime)
  if su.len != 4 or sul.len != 4:
    qexError "supply the four HISQ one-link and four HISQ long-link fields"
  for mu in 0..<4:
    if su[mu].l != left.l or sul[mu].l != left.l:
      qexError "HISQ links and random-wall propagators must use the same layout"
  let
    lo = left.l
    nt = lo.physGeom[^1]
  var
    shifted = lo.ColorVector()
    linked = lo.ColorVector()
    absolute = WallCorrelator(re: newSeq[float](nt),
                              im: newSeq[float](nt))

  for hop in [1, 3]:
    var sh = newShifter(right, 3, hop)
    discard sh ^* right
    threads:
      shifted := sh.field
      threadBarrier()
      for site in linked:
        if hop == 1:
          linked[site] := su[3][site] * shifted[site]
        else:
          linked[site] := sul[3][site] * shifted[site]
    let c = dotByWallTime(left, linked)
    # An h-link hop crosses h elementary temporal links.
    for t in 0..<nt:
      for offset in 0..<hop:
        let bin = (t + offset) mod nt
        absolute.re[bin] -= c.re[t]
        absolute.im[bin] -= c.im[t]

  result = relativeWallTime(absolute, sourceTime,
                            1.0 / wallSpatialVolume(left).float)


type
  RandomWallCorrelators* = object
    pp*, a0pThin*, a0pConserved*: WallCorrelator

proc randomWallCorrelators*(u, su, sul: auto; left, right: auto;
    sourceTime: int; thinLinksHaveBC: bool = false;
    antiperiodicTime: bool = true): RandomWallCorrelators =
  ## Contract one common random wall; either propagator may be flowed.
  if u.len != 4:
    qexError "supply the four thin gauge-link fields"
  result.pp = randomWallPP(left, right, sourceTime)
  result.a0pThin = randomWallA0PThin(u[3], left, right, sourceTime,
                                   thinLinksHaveBC, antiperiodicTime)
  result.a0pConserved = randomWallA0PForward(su, sul, left, right, sourceTime)

proc randomWallCorrelatorsFromProps*(u, su, sul: auto; props: auto;
    sourceTime: int; flavor: array[2, int] = [0, 0];
    sourceSlot: int = 0; thinLinksHaveBC: bool = false;
    antiperiodicTime: bool = true): RandomWallCorrelators =
  ## Adapter for props[dilution][sourceSlot][0][mass] in hisqflow.nim.
  ## sourceSlot is an array index; sourceTime is the actual lattice time.
  if props.len == 0:
    qexError "at least one random-wall propagator is required"
  for dilution in 0..<props.len:
    if sourceSlot < 0 or sourceSlot >= props[dilution].len:
      qexError "sourceSlot is outside the propagator array"
    if props[dilution][sourceSlot].len != 1:
      qexError "use one full-wall slot, not the corner-wall array"
    for mass in flavor:
      if mass < 0 or mass >= props[dilution][sourceSlot][0].len:
        qexError "flavor is outside the propagator mass array"
    let c = randomWallCorrelators(u, su, sul,
        props[dilution][sourceSlot][0][flavor[0]],
        props[dilution][sourceSlot][0][flavor[1]], sourceTime,
        thinLinksHaveBC, antiperiodicTime)
    if dilution == 0:
      result = c
    else:
      for t in 0..<c.pp.re.len:
        result.pp.re[t] += c.pp.re[t]
        result.pp.im[t] += c.pp.im[t]
        result.a0pThin.re[t] += c.a0pThin.re[t]
        result.a0pThin.im[t] += c.a0pThin.im[t]
        result.a0pConserved.re[t] += c.a0pConserved.re[t]
        result.a0pConserved.im[t] += c.a0pConserved.im[t]


when isMainModule:
  import gauge/hisqsmear
  import physics/stagSolve
  import comms/qmp
  import std/[json, os, parseopt, strutils, tables, times, math]

  proc drawWallTime(seed: uint64; nt, stream: int): int =
    ## Collective; all ranks draw the same time. Rejection avoids modulo bias.
    var r: RngMilc6
    r.seed(seed, stream)
    if nt <= 0 or nt > r.numInts:
      qexError "time extent must be in 1.." & $r.numInts
    let limit = r.numInts - (r.numInts mod nt)
    while true:
      let value = r.integer
      if value < limit:
        return value mod nt

  proc randomWallSource(src: Field; seed: uint64): int =
    ## Scalar time RNG uses a stream outside the per-site noise stream range.
    let lo = src.l
    result = drawWallTime(seed, lo.physGeom[^1], lo.physVol)
    let sourceTime = result
    var rng = newRNGField(RngMilc6, lo, seed)
    threads:
      src.z4 rng
      threadBarrier()
      for i in src.sites:
        if lo.coords[3][i].int != sourceTime:
          src{i} := 0

  qexInit()
  var options = initTable[string, string]()
  var parser = initOptParser()
  while true:
    parser.next()
    case parser.kind
    of cmdEnd: break
    of cmdLongOption, cmdShortOption:
      if parser.key notin ["configuration", "unit-gauge", "lattice-json",
          "json", "output", "save-source", "seed", "mass", "r2req", "maxits",
          "help", "h"]:
        qexError "unknown option: " & parser.key
      options[parser.key] = parser.val
    of cmdArgument:
      qexError "unexpected argument: " & parser.key

  if options.hasKey("help") or options.hasKey("h"):
    echo """conserveA0rngw --configuration:FILE --lattice-json:FILE [options]
  --unit-gauge          use unit links instead of --configuration
  --json:FILE           spectrum.mass and solver settings
  --mass:FLOAT          override spectrum.mass (default 0.013)
  --seed:UINT32         reproduce both wall and randomly drawn source time
                        (default: clock seed, broadcast from rank zero)
  --r2req:FLOAT         relative squared residual (default 1e-20)
  --maxits:INT          maximum solver iterations (default 10000)
  --output:FILE         write correlators and source metadata as JSON
  --save-source:FILE    save the Z4 source as double-precision QIO and verify it
Exactly one full-color Z4 wall and one inversion; all Nt times are eligible.
The current is the forward component, matching conservedA0.nim."""
    qexFinalize()
    quit(0)

  if options.hasKey("configuration") == options.hasKey("unit-gauge"):
    qexError "specify exactly one of --configuration:FILE or --unit-gauge"

  let info = if options.hasKey("json"): parseFile(options["json"])
             else: newJObject()
  let lattice = if options.hasKey("lattice-json"):
                  parseFile(options["lattice-json"])
                elif info.hasKey("lattice-geometry"): info
                else: %* {"lattice-geometry": [16, 16, 16, 48]}
  proc ints(node: JsonNode): seq[int] =
    for x in node: result.add x.getInt()
  let geometry = ints(lattice["lattice-geometry"])
  if geometry.len != 4:
    qexError "lattice-geometry must have four dimensions"
  for extent in geometry:
    if extent <= 0 or (extent mod 2) != 0:
      qexError "staggered epsilon hermiticity requires positive even extents"
  let lo = if lattice.hasKey("rank-geometry"):
             if lattice.hasKey("simd-geometry"):
               newLayout(geometry, VLEN, ints(lattice["rank-geometry"]),
                         ints(lattice["simd-geometry"]))
             else: newLayout(geometry, ints(lattice["rank-geometry"]))
           else: newLayout(geometry)

  var
    mass = 0.013
    r2req = 1e-20
    maxits = 10000
  if info.hasKey("spectrum") and info["spectrum"].hasKey("mass"):
    mass = info["spectrum"]["mass"].getFloat()
  if info.hasKey("solver"):
    let solver = info["solver"]
    if solver.hasKey("minimum-squared-residual"):
      r2req = solver["minimum-squared-residual"].getFloat()
    if solver.hasKey("maximum-iterations"):
      maxits = solver["maximum-iterations"].getInt()
  if options.hasKey("mass"): mass = parseFloat(options["mass"])
  if options.hasKey("r2req"): r2req = parseFloat(options["r2req"])
  if options.hasKey("maxits"): maxits = parseInt(options["maxits"])
  if not (mass > 0 and mass < Inf and r2req > 0 and r2req < 1 and maxits > 0):
    qexError "require finite mass > 0, 0 < r2req < 1, and maxits > 0"

  var seed: uint64
  if options.hasKey("seed"):
    seed = parseBiggestUInt(options["seed"]).uint64
    if seed > high(uint32).uint64:
      qexError "seed must fit in 32 bits for RngMilc6"
  elif myRank == 0:
    seed = uint64(epochTime() * 1e6) and high(uint32).uint64
  QMP_broadcast(seed.addr, sizeof(seed).csize_t)

  var
    g = lo.newGauge()
    su = lo.newGauge()
    sul = lo.newGauge()
    src = lo.ColorVector()
    prop = lo.ColorVector()
    residual = lo.ColorVector()
  if options.hasKey("configuration"):
    let filename = options["configuration"]
    if not fileExists(filename):
      qexError "configuration does not exist: " & filename
    if g.loadGauge(filename) != 0:
      qexError "unable to read configuration: " & filename
  else:
    threads: g.unit()
  threads:
    g.projectSU
    threadBarrier()
    g.setBC()
    threadBarrier()
    g.stagPhase()
  let hisq = newHISQ(0.0, 1.0)
  discard hisq.smearGetForce(g, su, sul)
  let stag = newStag3(su, sul)
  # Restore ordinary thin gauge links for the thin-link insertion.
  threads:
    g.stagPhase()
    threadBarrier()
    g.setBC()

  let sourceTime = randomWallSource(src, seed)
  let sourceNorm2 = src.norm2
  let expectedNorm2 = float(src[0].len * (geometry[0] * geometry[1] * geometry[2]))
  if abs(sourceNorm2 - expectedNorm2) > 1e-12 * expectedNorm2:
    qexError "random wall source norm does not equal Nc * spatial volume"
  echo "random-wall seed: ", seed, " source time: ", sourceTime,
       " source norm2: ", sourceNorm2

  let sourceFile = options.getOrDefault("save-source", "")
  if options.hasKey("save-source"):
    if sourceFile.len == 0:
      qexError "--save-source requires a filename"
    if fileExists(sourceFile):
      qexError "refusing to overwrite source field: " & sourceFile
    let sourceMetadata = $(%* {
      "source": "Z4 full-color random wall",
      "seed": seed,
      "source-time": sourceTime,
      "lattice-geometry": geometry,
      "source-norm2": sourceNorm2
    })
    var writer = lo.newWriter(sourceFile, sourceMetadata)
    if writer.status != 0:
      qexError "unable to open source field for writing: " & sourceFile
    writer.write(src, sourceMetadata, "D")
    if writer.status != 0:
      qexError "unable to write source field: " & sourceFile
    writer.close()
    if writer.status != 0:
      qexError "unable to close source field: " & sourceFile
    var reader = lo.newReader(sourceFile)
    if reader.status != 0:
      qexError "unable to open saved source field: " & sourceFile
    reader.read(residual)
    if reader.status != 0 or reader.recordMetadata != sourceMetadata:
      qexError "unable to read saved source field or metadata: " & sourceFile
    reader.close()
    if reader.status != 0:
      qexError "unable to close saved source field: " & sourceFile
    threads: residual -= src
    if residual.norm2 != 0.0:
      qexError "saved random-wall field does not exactly reproduce the source"
    echo "saved source field verified exactly: ", sourceFile

  var sp = initSolverParams()
  sp.r2req = r2req
  sp.maxits = maxits
  threads: prop := 0
  stag.solve(prop, src, mass, sp)
  threads:
    stag.D(residual, prop, mass)
    threadBarrier()
    residual -= src
  let relativeResidual2 = residual.norm2 / sourceNorm2
  if not (relativeResidual2 <= r2req):
    qexError "solve failed requested relative squared residual: " &
             $relativeResidual2 & " > " & $r2req
  echo "verified relative squared residual: ", relativeResidual2

  # Exercise the adapter used by a main driver with nested propagator storage.
  var props: array[1, array[1, array[1, array[1, typeof(prop)]]]]
  props[0][0][0][0] = prop
  let
    correlators = randomWallCorrelatorsFromProps(g, su, sul, props, sourceTime)
    pp = correlators.pp
    thin = correlators.a0pThin
    ap = correlators.a0pConserved
  var rawPP = newSeq[float](pp.re.len)
  for t in 0..<rawPP.len: rawPP[t] = -pp.re[t]
  let output = %* {
    "metadata": {
      "source": "Z4 full-color random wall",
      "noise-hits": 1,
      "inversions": 1,
      "seed": seed,
      "source-time": sourceTime,
      "source-time-range": [0, geometry[3] - 1],
      "source-time-draw": "uniform integer with rejection sampling",
      "source-norm2": sourceNorm2,
      "source-field": sourceFile,
      "source-field-format": "QIO double-precision color vector",
      "source-field-verified": sourceFile.len > 0,
      "lattice-geometry": geometry,
      "configuration": options.getOrDefault("configuration", "unit"),
      "mass": mass,
      "relative-residual2": relativeResidual2,
      "normalization": "divide by spatial volume; color trace is summed",
      "time-index": "(sink time - source time + Nt) mod Nt",
      "current-location": "dt + 1/2",
      "current-component": "forward only; one-link plus Naik on crossed links",
      "source-operator": "P = epsilon * barChi * chi",
      "PP-convention": "PP = positive norm; P-P = raw <P P>",
      "A0P-convention": "cA0P = <A0_forward P>; Wick minus sign"
    },
    "PP": pp.re,
    "P-P": rawPP,
    "A0Pre": thin.re,
    "A0Pim": thin.im,
    "cA0Pre": ap.re,
    "cA0Pim": ap.im
  }
  if myRank == 0:
    if options.hasKey("output"):
      writeFile(options["output"], pretty(output) & "\n")
    else:
      echo pretty(output)
  qexFinalize()
