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
