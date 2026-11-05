# Growing the metadata table must not invalidate a retained parent pointer.
# Nested halo/sequence conversions retain such pointers while adding children.
import qex
import backend/accel
qexInit()
var buffers = newSeq[seq[float]](512)
for k in 0..<buffers.len:
  buffers[k] = @[k.float]
let parent = getGpuMem(addr buffers[0][0], sizeof(float))
let allocation = parent.p
for k in 1..<buffers.len:
  discard getGpuMem(addr buffers[k][0], sizeof(float))
doAssert parent == getGpuMem(addr buffers[0][0])
doAssert parent.p == allocation and parent.bytes == sizeof(float)
for k in 0..<buffers.len:
  freeGpuMem(addr buffers[k][0])
echo "PASS: GPU metadata pointers survive table growth"
qexFinalize()
