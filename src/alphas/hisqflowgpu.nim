## Native accelerator implementation of the directional, massless HISQ Laplacian.
## Device-resident stencil with device halo packing and host-staged MPI.
## Only compact boundary messages cross host memory during an application.
##
## Mathematical operation (mu is one of the four spacetime directions):
##   D_mu psi(x) = 1/2 [F_mu(x) psi(x+mu) - F_mu(x-mu)^dagger psi(x-mu)
##                      +N_mu(x) psi(x+3mu) - N_mu(x-3mu)^dagger psi(x-3mu)]
##   L psi = sum_mu D_mu(D_mu psi) = -sum_mu D_mu^dagger D_mu psi.
## F is the fat link; N is the Naik link, which connects sites three steps apart.
## The second equality uses D_mu^dagger = -D_mu for this massless operator.
## Applying each direction twice separately avoids mixed terms D_mu D_nu.
##
## Typical use: create workspace -> updateLinks at each RK stage -> laplacian
## for each source -> free when finished. RK means Runge-Kutta time stepping.
## The constructor allocates host fields/maps; device buffers are obtained
## lazily by onGpu. Backend=SYCL uses the selected SYCL device (which can also
## be a CPU); Backend=CPU runs the same calculation through the host fallback.
##
## Nim notation used below:
##   * after a declaration exports it; auto lets Nim infer a concrete type.
##   ref object is a shared reference to an object, not a value-copy object.
##   = binds/assigns a variable; QEX's := copies values into an existing field.
##   newOneOf allocates a field with the same type/layout, without copying data.
##   0..<4 visits 0,1,2,3; @[...] constructs a sequence; .adj means adjoint.
##   A procedure's final expression can be its return value.
##
## Transfer flags are bookkeeping, not hardware read/write permissions:
##   gmGpuRead + (gmCpuWrite or gmCpuWriteOnce) requests host-to-device copying.
##   gmCpuRead + gmGpuWrite requests device-to-host copying.
##   gmCpuWriteOnce is cleared after one upload, so unchanged data stay resident.
##   gpuFlagsIncl adds flags; gpuFlagsExcl removes them. Neither copies data
##   immediately: transfers occur through toGpu/fromGpu around the kernel.
##   These rules also track whether a copy already happened in this kernel call.
##   A new entry initially allows CPU/GPU reads and writes; exclusions below
##   remove unnecessary transfers. They do not detect CPU writes automatically.
##
## A Halo has TWO data buffers: .field holds this rank's local sites, while
## .halo holds extra neighbor values gathered from local or remote sites.
## h.gpuFlags... changes ONLY the halo-buffer flags; h.field.gpuFlags... changes
## the local-field flags. h[index] chooses either buffer using the halo layout.
# Core QEX types, field arithmetic, threading, layouts and communicator access.
import qex
# Accelerator dispatch/memory flags plus Field <-> GpuField conversion.
import backend/[accel, cgfield]
# Host halo construction/gathering plus device halo types and neighbor indexing.
import comms/[halo, halogpu, halodevice]

# Expose these adapters to modules instantiating the generic procedures below.
export accel, cgfield, halo, halogpu, halodevice

# Reusable workspace. Generic types are: halo layout, halo map, gauge halo,
# fermion halo, ordinary fermion field, and fermion/gauge exchange plans.
type HisqFlowGpu*[HL, HM, GH, FH, F, E, GE] = ref object
  # Shared geometry and neighbor-index tables for local sites plus halos.
  layout: HL
  # Eight host gather plans: one backward-link halo for each fat/Naik field.
  linkMaps: seq[HM]
  # Host gather plan covering source offsets +/-1 and +/-3 in all directions.
  fermionMap: HM
  # Four smaller plans: intermediate neighbors needed for one direction only.
  directionMaps: seq[HM]
  # Owned link copies: [fat_0, Naik_0, fat_1, Naik_1, ..., fat_3, Naik_3].
  links: seq[GH]
  # Source psi and scratch D_mu psi, each with a separate neighbor buffer.
  source, intermediate: FH
  # Device accumulator for the host-input convenience API.
  output: F
  # Initially false: no links have been loaded, and the workspace is not freed.
  linksReady, closed: bool
  # Persistent device packing plans and compact MPI send/receive buffers.
  sourceExchange: E
  directionExchanges: seq[E]
  linkExchanges: seq[GE]

# Construct storage and communication plans. g supplies a gauge-field type;
# psi supplies the fermion type/layout. Actual link values are loaded later.
proc newHisqFlowGpu*(g: auto, psi: auto): auto =
  # Read the lattice layout attached to the caller's fermion field.
  let lo = psi.l
  # The fixed-size offsets and direction loops below assume four dimensions.
  doAssert lo.nDim == 4
  # Three layers are needed for a Naik hop. The second application exchanges
  # a fresh intermediate, so a six-layer halo is not required.
  # Reserve three outer-site layers on both sides of every dimension and
  # build the neighbor tables. Only requested halo entries will be gathered.
  let hl = lo.makeHaloLayout([3,3,3,3], [3,3,3,3])
  # Name the gather-map type specialized to this lattice layout.
  type HM = HaloMap[typeof(lo)]
  # Infer a gauge halo's type; typeof here does not allocate a halo at runtime.
  type GH = typeof(makeHalo(hl, g[0]))
  # Infer the corresponding fermion halo type, also without runtime allocation.
  type FH = typeof(makeHalo(hl, psi))
  # Declare a reference with all five workspace type parameters fixed.
  type E = typeof(newDeviceHaloExchange(default(FH), default(HM)))
  type GE = typeof(newDeviceHaloExchange(default(GH), default(HM)))
  var w: HisqFlowGpu[typeof(hl), HM, GH, FH, typeof(psi), E, GE]
  # Allocate the workspace object; fields start at their default values.
  new w
  # Keep the shared halo geometry for all subsequent applications.
  w.layout = hl
  # Allocate an independent local source field and its halo storage.
  w.source = makeHalo(hl, psi.newOneOf)
  # Allocate the local D_mu psi scratch field and its halo storage.
  w.intermediate = makeHalo(hl, psi.newOneOf)
  # Allocate the result field; it does not need its own halo.
  w.output = psi.newOneOf
  # Allocate eight halo-reference slots, not the link data buffers yet.
  w.links = newSeq[GH](8)
  # Allocate eight slots for the link gather plans.
  w.linkMaps = newSeq[HM](8)
  # Allocate one intermediate-fermion gather-plan slot per direction.
  w.directionMaps = newSeq[HM](4)
  # Use QEX's current communicator for host halo exchanges between ranks.
  let comm = getDefaultComm()
  # Start an empty list of four-component displacement vectors for psi.
  var offsets: seq[array[4,int32]]
  # Visit each of the four lattice directions.
  for mu in 0..<4:
    # Start the displacement list for this direction alone.
    var directionOffsets: seq[array[4,int32]]
    # Handle the one-link fat term, then the three-link Naik term.
    for hop in [1,3]:
      # Map (direction, hop) to the interleaved link slot: 2*mu or 2*mu+1.
      let k = 2*mu + (if hop == 1: 0 else: 1)
      # Create a displacement vector initially equal to (0,0,0,0).
      var backward: array[4,int32]
      # Select x-mu for a fat link, or x-3mu for a Naik link.
      backward[mu] = int32(-hop)
      # Precompute how to gather those backward links. Forward terms use links
      # at the local site x, so no forward-link halo is needed.
      w.linkMaps[k] = hl.makeHaloMap(comm, @[backward])
      # Allocate this link's owned local matrix field and halo; g[0] is a type/layout
      # prototype only, and updateLinks will fill the correct direction's values.
      w.links[k] = makeHalo(hl, g[0].newOneOf)
      # The fermion stencil needs both backward and forward neighbors.
      for sign in [-1,1]:
        # Start another all-zero four-dimensional displacement.
        var offset: array[4,int32]
        # Set its only nonzero component to -1, +1, -3 or +3.
        offset[mu] = int32(sign*hop)
        # Add the displacement to the all-directions source gather plan.
        offsets.add offset
        # Also add it to this direction's intermediate gather plan.
        directionOffsets.add offset
    # Finish the gather plan for +/-1 and +/-3 along this mu only.
    w.directionMaps[mu] = hl.makeHaloMap(comm, directionOffsets)
  # Finish the source plan with all sixteen displacements (four per mu).
  w.fermionMap = hl.makeHaloMap(comm, offsets)
  w.sourceExchange = newDeviceHaloExchange(w.source, w.fermionMap)
  for hm in w.directionMaps:
    w.directionExchanges.add newDeviceHaloExchange(w.intermediate, hm)
  for k in 0..<8:
    w.linkExchanges.add newDeviceHaloExchange(w.links[k], w.linkMaps[k])
  # These fields are owned by the workspace: changing their flags cannot
  # invalidate the caller's CPU fields or another solver's device storage.
  # Source interior is read-only on device. Disable repeated CPU uploads;
  # loadSource will explicitly mark each new source with gmCpuWriteOnce.
  w.source.field.gpuFlagsExcl {gmGpuWrite, gmCpuWrite}
  # Source halo is filled on device by sourceExchange before each application.
  w.source.gpuFlagsExcl {gmGpuWrite, gmCpuWrite}
  # The intermediate interior is produced on device, never overwritten on CPU.
  # The resident policy below disables full-field downloads for halo exchange.
  w.intermediate.field.gpuFlagsExcl {gmCpuWrite}
  # Intermediate halos are written by device exchange and read by the stencil.
  # exchange explicitly enables gmGpuWrite for its packing/copy kernels.
  w.intermediate.gpuFlagsExcl {gmGpuWrite, gmCpuWrite}
  # Keep the accumulating output on device: no upload of old host contents,
  # and no automatic download until downloadOutput temporarily enables it.
  w.output.gpuFlagsExcl {gmCpuRead, gmCpuWrite}
  # Apply the read-only device policy to all eight owned link halos.
  for link in w.links:
    # Local links are read by kernels; updateLinks requests uploads when changed.
    link.field.gpuFlagsExcl {gmGpuWrite, gmCpuWrite}
    # Apply the same policy separately to this link's halo buffer.
    link.gpuFlagsExcl {gmGpuWrite, gmCpuWrite}
  # Every local buffer now stays resident, including D_mu psi. Halo exchange
  # downloads only the compact send buffer, not the intermediate field.
  for f in [w.source.field, w.intermediate.field, w.output]: f.resident()
  for link in w.links: link.field.resident()
  # Return the new workspace through Nim's implicit result variable.
  result = w

# Copy fresh, already smeared/phased links into the workspace. Call again
# whenever the RK stage changes the gauge links; this routine does no smearing.
proc updateLinks*(w: HisqFlowGpu, links: auto) =
  ## Host-input compatibility API; copy, upload once, then exchange on device.
  doAssert not w.closed and links.len == 8
  for k in 0..<8:
    threads: w.links[k].field := links[k]
    w.links[k].field.hostUpdated()
    w.linkExchanges[k].exchange(w.links[k])
  w.linksReady = true

proc updateDeviceLinks*(w: HisqFlowGpu, links: auto) =
  ## Accept already resident GPU HISQ links without a host download/re-upload.
  doAssert not w.closed and links.len == 8
  for k in 0..<8:
    deviceCopy(w.links[k].field, links[k])
    w.linkExchanges[k].exchange(w.links[k])
  w.linksReady = true

# Convenience overload accepting QEX's staggered-operator object.
proc updateLinks*(w: HisqFlowGpu, stag: Staggered) =
  # Extract its link sequence and use the copying/gathering overload above.
  w.updateLinks(stag.g)

# Private kernel wrapper: compute dst = D_mu src, or dst += D_mu src when
# accumulate is true. src must have the required device halo ready beforehand.
proc applyDirection(w: HisqFlowGpu, dst: auto, src: auto,
                    mu: int, accumulate: bool) =
  # Use the destination's lattice layout to size the kernel launch.
  let lo = dst.l
  # Capture the shared neighbor tables for conversion to their device form.
  let hl = w.layout
  # Capture this direction's fat-link halo reference, without copying matrices.
  var fat = w.links[2*mu]
  # Capture the matching Naik-link halo reference.
  var naik = w.links[2*mu+1]
  # Convert false/true to a captured integer switch, 0/1.
  let add = ord(accumulate)
  # Make a mutable local alias for the accelerator conversion/finalization hooks.
  # This shadows the parameter name; it does not allocate or copy a whole field.
  var dst = dst
  # Likewise make a mutable local alias to the source Halo object.
  var src = src
  # Declare that this kernel writes dst. In particular, restore this flag when
  # the intermediate becomes a destination again after being a read-only source.
  dst.gpuFlagsIncl {gmGpuWrite}
  # Declare the source interior read-only. When src is the intermediate, this
  # prevents downloading it again after the second, read-only application.
  src.field.gpuFlagsExcl {gmGpuWrite}
  # Convert captured fields/halos to device views, upload as requested by flags,
  # launch the kernel, wait, then perform only the flagged downloads. The body
  # below runs on the selected device; the surrounding procedure runs on host.
  onGpu(lo):
    # Distribute local destination sites among device threads. This iterator
    # handles QEX's SIMD-packed storage; the GPU path processes individual sites.
    for s in gpuSites(dst):
      # Find x+mu in the extended layout; the result may point into halo storage.
      let f1 = hl.nbrFwd(mu, s)
      # Find x-mu, also allowing a halo index at a local boundary.
      let b1 = hl.nbrBck(mu, s)
      # Follow two more forward neighbors from f1 to reach x+3mu.
      let f3 = hl.nbrFwd(mu, hl.nbrFwd(mu, f1))
      # Follow two more backward neighbors from b1 to reach x-3mu.
      let b3 = hl.nbrBck(mu, hl.nbrBck(mu, b1))
      # Links already contain HISQ coefficients, staggered phases and BCs.
      # Do not multiply the Naik coefficient or eta_mu a second time.
      # Overwrite mode starts a fresh result; accumulate mode preserves prior terms.
      if add == 0:
        # Zero the color vector at this site before adding the four stencil terms.
        dst[s] := 0
      # +1-hop term: F_mu(x) times psi(x+mu), with the overall D_mu factor 1/2.
      dst[s] += 0.5 * (fat.field[s] * src[f1])
      # -1-hop term: subtract F_mu(x-mu)^dagger psi(x-mu)/2. .adj conjugates
      # and transposes the color matrix; halo indexing supplies remote values.
      dst[s] -= 0.5 * (fat[b1].adj * src[b1])
      # +3-hop term: N_mu(x) times psi(x+3mu)/2. N already contains its coefficient.
      dst[s] += 0.5 * (naik.field[s] * src[f3])
      # -3-hop term: subtract N_mu(x-3mu)^dagger psi(x-3mu)/2.
      dst[s] -= 0.5 * (naik[b3].adj * src[b3])

# Prepare a fresh caller-supplied fermion source in owned host storage.
proc loadSource(w: HisqFlowGpu, src: auto) =
  ## Host-input API: callers do not need to manage device flags themselves.
  threads: w.source.field := src
  w.source.field.hostUpdated()
  w.sourceExchange.exchange(w.source)

# Make the convenience API accumulator available to CPU callers.
proc downloadOutput(w: HisqFlowGpu) =
  # Temporarily enable the CPU-read half of the device-to-host copy condition.
  w.output.gpuFlagsIncl {gmCpuRead}
  # Copy device output to its host field (a no-op for the shared-memory CPU backend).
  w.output.fromGpu()
  # Restore device residency for the next sequence of kernel accumulations.
  w.output.gpuFlagsExcl {gmCpuRead}

# Public one-direction D_mu application, useful for checking the stencil alone.
proc directional*(w: HisqFlowGpu, src: auto, mu: int): auto =
  ## Result aliases workspace storage; copy it before the next application.
  # Require a live workspace with link values already loaded.
  doAssert not w.closed and w.linksReady
  # Reject an invalid direction before indexing the link/map sequences.
  doAssert mu >= 0 and mu < 4
  # Copy/upload the new source and exchange its boundary values on device.
  w.loadSource(src)
  # Compute just D_mu src, overwriting any previous output.
  w.applyDirection(w.output, w.source, mu, false)
  # Download once so the returned ordinary Field can be consumed on CPU.
  w.downloadOutput()
  # Return the owned output field itself, not a newly allocated copy.
  w.output

proc laplacianInto*(w: HisqFlowGpu, dst, src: Field, mass = 0.0) =
  ## Device-input/device-output API for the RK integrator. dst may alias src:
  ## the input is preserved in owned scratch before any output is written.
  doAssert not w.closed and w.linksReady
  doAssert mass == 0.0, "HISQ fermion flow requires zero flow mass"
  if (addr src[0]) != (addr w.source.field[0]):
    deviceCopy(w.source.field, src)
  w.sourceExchange.exchange(w.source)
  for mu in 0..<4:
    w.applyDirection(w.intermediate.field, w.source, mu, false)
    # The first pass stays on device. Pack only the neighbors needed by mu.
    w.directionExchanges[mu].exchange(w.intermediate)
    w.applyDirection(dst, w.intermediate, mu, mu != 0)

proc laplacian*(w: HisqFlowGpu, src: auto, mass = 0.0): auto =
  ## Compatibility API returning host-visible workspace-owned output.
  ## Flow uses laplacianInto instead so all RK vectors remain on device.
  doAssert not w.closed and w.linksReady
  doAssert mass == 0.0, "HISQ fermion flow requires zero flow mass"
  threads: w.source.field := src
  w.source.field.hostUpdated()
  w.laplacianInto(w.output, w.source.field, mass)
  w.downloadOutput()
  w.output

# Explicit cleanup; host fields/maps remain ordinary Nim-managed objects.
proc free*(w: HisqFlowGpu) =
  ## Release only this workspace's device allocations and metadata.
  # Make repeated cleanup harmless: return immediately if already freed.
  if w.closed: return
  w.sourceExchange.free()
  for ex in w.directionExchanges: ex.free()
  for ex in w.linkExchanges: ex.free()
  # Visit the eight owned link fields and their halo buffers.
  for link in w.links:
    # Use the host field address as the lookup key to release its device allocation
    # and bookkeeping entry; this does not free the host field itself.
    freeGpuMem(addr link.field[0])
    # Release the separate device copy/bookkeeping for this link's halo buffer.
    freeGpuMem(addr link.halo[0])
  # Clean up both fermion halos owned by this workspace.
  for fieldHalo in [w.source, w.intermediate]:
    # Release the device allocation associated with the local fermion field.
    freeGpuMem(addr fieldHalo.field[0])
    # Release the device allocation associated with its neighbor buffer.
    freeGpuMem(addr fieldHalo.halo[0])
  # Release the device accumulator and its memory-table entry.
  freeGpuMem(addr w.output[0])
  # Call the halo-layout overload to release device neighbor tables and pointers.
  freeGpuMem(w.layout)
  # Record closure so future calculations fail their assertions and free is idempotent.
  w.closed = true
