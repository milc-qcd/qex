import qex
import grid/[Grid]
import gauge/[gaugeAction]

import gridutilities

{.pragma: grid, header: "<Grid/Grid.h>".}

const GRIDONELOOPACTION = "Grid::PeriodicSymanzikOneLoopGaugeAction<Grid::PeriodicGimplR>"

type GridOneLoopGaugeAction* {.importcpp: GRIDONELOOPACTION, grid.} = object

proc gaugeAction2(
  action: GridOneLoopGaugeAction; 
  u: GridLatticeGaugeField
): cdouble {.importcpp: "#.S(#)", grid.}

proc gaugeForce2(
  action: GridOneLoopGaugeAction; 
  u: GridLatticeGaugeField; 
  force: GridLatticeGaugeField
) {.importcpp: "#.deriv(#, #)", grid.}

proc newGridOneLoopGaugeAction(
  grid: ptr GridCartesian; 
  cp: cdouble,
  u0: cdouble,
  nf: cint
): GridOneLoopGaugeAction 
  {.importcpp: GRIDONELOOPACTION & "(#, #, #, #)", constructor, grid.}

proc gaugeActionOneLoopHISQ*[U](gc: GaugeActionCoeffs; u: openArray[U]): cdouble =
  var info: PerfInfo
  
  # prepare grid
  let lo = u[0].l
  let
    lat = lo.physGeom
    latSize = newCoordinate(lat)
  let
    simdLayout = GridDefaultSimd(len(lat), Nsimd(GridVComplex))
    mpiLayout = newCoordinate(lo.rankGeom)
  let grid = latSize.newGridCartesian(simdLayout, mpiLayout)

  # get grid link and action object
  var g = grid.gauge()
  let action = newGridOneLoopGaugeAction(addr grid, gc.plaq, 1.0, 0)

  # do action calculation and return
  g := u
  tic("GridGaugeAction")
  result = action.gaugeAction2(g)
  info.secs = getElapsedTime()
  echo &"gaugeAction: {info.secs:.5f}s"
  info.clear()

proc gaugeForceOneLoopHISQ*[U](gc: GaugeActionCoeffs; u, force: seq[U]) =
  var info: PerfInfo

  # prepare grid
  let lo = u[0].l
  let
    lat = lo.physGeom
    latSize = newCoordinate(lat)
  let
    simdLayout = GridDefaultSimd(len(lat), Nsimd(GridVComplex))
    mpiLayout = newCoordinate(lo.rankGeom)
  let grid = latSize.newGridCartesian(simdLayout, mpiLayout)

  # get grid link and action object
  var g = grid.gauge()
  var f = grid.gauge()
  let action = newGridOneLoopGaugeAction(addr grid, gc.plaq, 1.0, 0)

  # calculate force using Grid
  g := u
  tic("GridGaugeForce")
  action.gaugeForce2(g, f)
  info.secs = getElapsedTime()
  echo &"gaugeForce: {info.secs:.5f}s"
  info.clear()

  # convert force back to QEX format & set proper normalization
  force.toQEX(f)
  #threads:
  #  for mu in 0..<force.len:
  #    force[mu] *= 2.0

#[
{.pragma: gauge, header: "<Grid/qcd/action/gauge/PlaqPlusRectangleAction.h>".}

const GAUGEACTION = "Grid::PlaqPlusRectangleAction<Grid::PeriodicGimplR>"

type GridPlaqPlusRectangleAction* {.importcpp: GAUGEACTION, gauge.} = object

proc gaugeAction2(
  action: GridPlaqPlusRectangleAction; 
  u: GridLatticeGaugeField
): cdouble {.importcpp: "#.S(#)", gauge.}

proc gaugeForce2(
  action: GridPlaqPlusRectangleAction; 
  u: GridLatticeGaugeField; 
  force: GridLatticeGaugeField
) {.importcpp: "#.deriv(#, #)", gauge.}

proc newGridPlaqPlusRectangleAction(cp, cr: cdouble): GridPlaqPlusRectangleAction
  {.importcpp: GAUGEACTION & "(#, #)", constructor, gauge.}

template newRectangleGaugeAction(gc: GaugeActionCoeffs): untyped = 
  newGridPlaqPlusRectangleAction(gc.plaq, gc.rect)

proc gaugeActionOneLoopHISQ*[U](gc: GaugeActionCoeffs; u: openArray[U]): cdouble =
  var info: PerfInfo
  
  # prepare grid
  let lo = u[0].l
  let
    lat = lo.physGeom
    latSize = newCoordinate(lat)
  let
    simdLayout = GridDefaultSimd(len(lat), Nsimd(GridVComplex))
    mpiLayout = newCoordinate(lo.rankGeom)
  let grid = latSize.newGridCartesian(simdLayout, mpiLayout)

  # get grid link and action object
  var g = grid.gauge()
  let action = gc.newRectangleGaugeAction()

  # do action calculation and return
  g := u
  tic("GridGaugeAction")
  result = action.gaugeAction2(g)
  info.secs = getElapsedTime()
  echo &"gaugeAction: {info.secs:.5f}s"
  info.clear()

proc gaugeForceOneLoopHISQ*[U](gc: GaugeActionCoeffs; u, force: seq[U]) =
  var info: PerfInfo

  # prepare grid
  let lo = u[0].l
  let
    lat = lo.physGeom
    latSize = newCoordinate(lat)
  let
    simdLayout = GridDefaultSimd(len(lat), Nsimd(GridVComplex))
    mpiLayout = newCoordinate(lo.rankGeom)
  let grid = latSize.newGridCartesian(simdLayout, mpiLayout)

  # get grid link and action object
  var g = grid.gauge()
  var f = grid.gauge()
  let action = gc.newRectangleGaugeAction()

  # calculate force using Grid
  g := u
  tic("GridGaugeForce")
  action.gaugeForce2(g, f)
  info.secs = getElapsedTime()
  echo &"gaugeForce: {info.secs:.5f}s"
  info.clear()

  # convert force back to QEX format & set proper normalization
  force.toQEX(f)
  threads:
    for mu in 0..<force.len:
      force[mu] *= 2.0
]#

#[
{.pragma: gauge, header: "<Grid/qcd/action/gauge/OneLoopGaugeAction.h>".}

const ONELOOPACTION = "Grid::OneLoopGaugeAction<Grid::PeriodicGimplR>"

type OneLoopGaugeAction* {.importcpp: ONELOOPACTION, gauge.} = object

proc newOneLoopGaugeAction(
  grid: ptr GridCartesian, 
  beta, cp, cr, cpr: cdouble
): OneLoopGaugeAction 
  {.importcpp: ONELOOPACTION & "(#, #, #, #, #)", constructor, gauge.}

template newOneLoopGaugeAction*(
  gc: GaugeActionCoeffs; 
  grid: ptr GridCartesian
): untyped =
  newOneLoopGaugeAction(grid, gc.plaq, gc.plaq, gc.rect, gc.pgm)

proc S(action: OneLoopGaugeAction; u: GridLatticeGaugeField): cdouble 
  {.importcpp: "#.S(#)", gauge.}

proc gaugeActionOneLoopHISQ*[U](gc: GaugeActionCoeffs; u: openArray[U]): cdouble =
  # prepare grid
  let lo = u[0].l
  let
    lat = lo.physGeom
    latSize = newCoordinate(lat)
  let
    simdLayout = GridDefaultSimd(len(lat), Nsimd(GridVComplex))
    mpiLayout = newCoordinate(lo.rankGeom)
  let grid = latSize.newGridCartesian(simdLayout, mpiLayout)

  # get grid link and action object
  var g = grid.gauge()
  let action = gc.newOneLoopGaugeAction(addr grid)

  # do action calculation and return
  g := u
  return action.S(g)
]#
  
