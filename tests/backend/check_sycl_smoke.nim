import qex
import backend/accel
import std/math
proc main =
  qexInit()
  var a = @[1.0, 2.0, 3.0, 4.0]
  onGpu(4):
    for i in gpuRange(4):
      a[i] = 2.0 * a[i]
  doAssert a == @[2.0, 4.0, 6.0, 8.0]
  # Nim 2's imported math routines carry documentation bodies. The device
  # inliner must preserve the foreign call instead of expanding that body.
  onGpu(4):
    for i in gpuRange(4):
      let x = a[i]
      a[i] = sqrt(x) + sin(x) + cos(x)
  for i in 0..<4:
    let x = 2.0*(i+1).float
    let error = abs(a[i] - (sqrt(x)+sin(x)+cos(x)))
    doAssert (cast[uint64](error) and 0x7ff0000000000000'u64) != 0x7ff0000000000000'u64
    doAssert error < 1e-13
  echo "PASS: SYCL kernel, device transfers and imported math calls"
  freeAllGpuMem()
  qexFinalize()

main()
