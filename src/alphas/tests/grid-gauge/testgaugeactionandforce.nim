import qex
import grid/[Grid]
import gauge/[gaugeAction]
import gauge/[symanzik1loopAction]
import gauge/[symanzik1loopForce]

{.pragma: grid, header: "<Grid/Grid.h>".}

letParam:
  lat = @[8, 8, 8, 16]

const GRIDGAUGEACTIONSTR = "Grid::PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction<Grid::PeriodicGimplR>"

type GridGaugeAction* {.importcpp: GRIDGAUGEACTIONSTR, grid.} = object

proc newGridGaugeAction(grid: ptr GridCartesian; beta, cp, cr, cpg: float): GridGaugeAction
  {.importcpp: GRIDGAUGEACTIONSTR & "(#, #, #, #)", constructor, grid.}

proc action(a: GridGaugeAction; u: GridLatticeGaugeField): float {.importcpp: "#.S(#)", grid.}

proc deriv(a: GridGaugeAction; u: GridLatticeGaugeField; force: GridLatticeGaugeField) {.importcpp: "#.deriv(#, #)", grid.}

proc checkerboard(x: ptr GridLatticeGaugeField): int 
  {.importcpp: "#->Checkerboard()", grid.}

proc `:=`(r0: seq[Field], x0: GridLatticeGaugeField) =
  type GridScalarObject = GridLatticeGaugeField.scalarObj
  let 
    r = addr r0
    x = addr x0
  let
    lo = r0[0].l
    nd = lo.nDim
    nc = r0[0][0].ncols
    nSites = lo.nSites
  var subset = lo.getSubset("all")
  let glSites = x0.Grid.lSites
  var c0 = newSeq[cint](nd)

  lo.coord(c0, lo.myrank, 0)
  if glSites != nSites:
    subset = case x.checkerboard == 0
      of true: lo.getSubset("even")
      of false: lo.getSubset("odd")

  block:
    {.emit: "using namespace Grid;".}
    {.emit: "Coordinate c(`nd`);".}
    {.emit: ["autoView(dst, ", x[], ", CpuRead);"].}
    
    var t: GridScalarObject
    
    for s in subset.singleSites:
      # set coordinate
      for mu in 0..<nd: {.emit: ["c[", mu, "] = ", lo.coords[mu][s].cint - c0[mu], ";"].}
      {.emit: "peekLocalSite(`t`, dst, c);".}

      # set link values
      for mu in 0..<nd:
        for a in 0..<nc:
          for b in 0..<nc:
            var tr, ti {.noinit.}: float
            {.emit: "`tr` = `t`._internal[`mu`]._internal._internal[`a`][`b`].real();".}
            {.emit: "`ti` = `t`._internal[`mu`]._internal._internal[`a`][`b`].imag();".}
            r[][mu]{s}[a,b] := newComplex(tr, ti)

proc testAction =
  let lo = lat.newLayout()
  var qu = lo.newGauge()
  var unit = lo.newGauge()
  var grid = qu[0].getGridPtr()
  var gu = grid[].gauge()
  var info: PerfInfo

  qu.random()
  gu := qu
  unit.unit()

  let (beta, cp, cr, cpg) = (6.0, 1.0, 1.0, 1.0)

  let qa = GaugeActionCoeffs(plaq: beta*cp, rect: beta*cr, pgm: beta*cpg)
  var ga = grid.newGridGaugeAction(beta, cp, cr, cpg)

  let (space, time) = qa.symanzik1loopAction(qu, info)
  let (snorm, tnorm) = qa.symanzik1loopAction(unit, info)
  let qav = space - snorm + time - tnorm
  let gav = ga.action(gu)

  echo "QEX: ", qav, " Grid: ", gav, " diff: ", qav-gav, " ratio: ", qav/gav

  var qf = lo.newGauge()
  var gf = grid[].gauge()
  var qgf = lo.newGauge()

  threads:
    for mu in 0..<qf.len: qf[mu] := 0

  qa.symanzik1loopDeriv(qu, qf, info)
  #qa.gaugeForce(qu, qf)
  ga.deriv(gu, gf)
  qgf := gf
  contractProjectTAH(qu, qf)
  threads:
    for mu in 0..<qf.len:
      for s in qf[mu]:
        qf[mu][s] *= -1.0
        qgf[mu][s] *= 2.0

  var df = 0.0

  threads:
    var dft = 0.0
    for mu in 0..<qf.len:
      for s in qf[mu]:
        dft += redot(qf[mu][s] - qgf[mu][s], qf[mu][s] - qgf[mu][s]).simdSum()
    threadBarrier()
    threadSum(dft)
    threadBarrier()
    threadMaster:
      df = dft

  rankSum(df)

  echo "df: ", sqrt(df)/4.0/3.0/lo.physVol



when isMainModule:
  qexInit()
  testAction()
  qexFinalize()