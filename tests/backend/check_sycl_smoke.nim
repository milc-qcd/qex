import qex
import backend/accel
proc main =
  qexInit()
  var a = @[1.0, 2.0, 3.0, 4.0]
  onGpu(4):
    for i in gpuRange(4):
      a[i] = 2.0 * a[i]
  doAssert a == @[2.0, 4.0, 6.0, 8.0]
  echo "PASS: SYCL kernel and device transfers"
  freeAllGpuMem()
  qexFinalize()

main()
