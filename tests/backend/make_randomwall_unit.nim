import qex
import std/os

qexInit()
let lo = newLayout(@[8, 8, 8, 8], VLEN, @[1, 1, 1, 1], @[2, 2, 2, 1])
var u = lo.newGauge()
threads: u.unit()
let filename = paramStr(1)
if filename.len == 0 or fileExists(filename):
  qexError "supply a new gauge filename"
if u.saveGauge(filename) != 0:
  qexError "cannot save unit gauge"
qexFinalize()
