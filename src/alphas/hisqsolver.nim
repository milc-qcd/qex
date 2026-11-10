## Explicit solver selection for HISQ flow. Keep the CPU reference as default
## even when the optional QUDA library is linked into the application.
import physics/stagSolve
import quda/qudaWrapper
import std/strutils

proc selectHisqSolver*(name: string): SolverBackend =
  case name.toLowerAscii()
  of "qex", "cpu": result = sbQex
  of "quda":
    when haveQuda():
      result = sbQuda
    else:
      raise newException(ValueError,
        "--solver=quda requires rebuilding with QUDA_DIR set; use --solver=qex for this binary")
  else:
    raise newException(ValueError, "Unknown HISQ solver '" & name & "'; choose qex or quda")
