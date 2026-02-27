import qex
import grid/[Grid]

{.pragma: grid, header: "<Grid/Grid.h>".}

# std::vector wraper

type vector*[T] {.importcpp: "std::vector<'*0>", header:"<vector>".} = object

proc newVector*[T](x,y: ptr T): vector[T] 
  {.importcpp: "std::vector<'*0>(#, #)", constructor.}

proc toVector*[T](x: openArray[T]): vector[T] =
  let x0 = unsafeAddr x[0]
  return newVector(x0, x0 + x.len)

# Grid ---> QEX conversion of gauge field

proc checkerboard(x: ptr GridLatticeGaugeField): int 
  {.importcpp: "#->Checkerboard()", grid.}

proc toQEX*(r0: seq[Field], x0: GridLatticeGaugeField) =
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

  block: # should be block, not threads; causes problems w/ Grid's memory manager (cpuLock != 0)
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

proc toGrid*(r: var GridLatticeGaugeField, x0: openArray[Field]) = r := x0