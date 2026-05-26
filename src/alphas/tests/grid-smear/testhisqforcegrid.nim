import qex
import ../../[alphashisq]
import ../../extra/[qexsmear]
import ../../extra/[gridsmear]

import gauge/[gaugeAction]

letParam:
  lat = @[8, 8, 8, 16]

proc testForce = 
  let lo = lat.newLayout()

  var u = lo.newGauge()
  var unit = lo.newGauge()
  var (qsu, qsul) = (lo.newGauge(), lo.newGauge())
  var (gsu, gsul) = (lo.newGauge(), lo.newGauge())

  u.random()
  #u.unit()
  unit.unit()
  
  var hisq = newHISQ(0.0, 1.0)#, reunitEps = 5e-5)

  var qforce = qexsmear.smearGetForce(hisq, u, qsu, qsul, regulate = false)
  var gforce = gridsmear.smearGetForce(hisq, u, gsu, gsul, regulate = false)

  var (dsdsu, dsdsul) = (lo.newGauge(), lo.newGauge())

  dsdsu.random()
  dsdsul.random()

  var (qdsdu, gdsdu) = (lo.newGauge(), lo.newGauge())

  threads:
    for mu in 0..<qdsdu.len: qdsdu[mu] := 0
    for mu in 0..<gdsdu.len: gdsdu[mu] := 0

  qforce(qdsdu, dsdsu, dsdsul)
  gforce(gdsdu, dsdsu, dsdsul)

  # test agreement in links

  var (dsu, dsul) = (0.0, 0.0)

  threads:
    var (dsut, dsult) = (0.0, 0.0)
    for mu in 0..<u.len:
      for s in u[mu]:
        dsut += redot(qsu[mu][s] - gsu[mu][s], unit[mu][s]).simdSum()
        dsult += redot(qsul[mu][s] - gsul[mu][s], unit[mu][s]).simdSum()
    threadBarrier()
    threadSum(dsut)
    threadSum(dsult)
    threadBarrier()
    threadMaster:
      dsu = dsut
      dsul = dsult

  rankSum(dsu)
  rankSum(dsul)

  echo "dsu: ", dsu/4.0/3.0/lo.physVol, " dsul: ", dsul/4.0/3.0/lo.physVol

  # test agreement in force

  var df = 0.0

  threads:
    var dft = 0.0
    for mu in 0..<u.len:
      for s in u[mu]:
        dft += redot(qdsdu[mu][s] - gdsdu[mu][s], unit[mu][s]).simdSum()
    threadBarrier()
    threadSum(dft)
    threadBarrier()
    threadMaster:
      df = dft

  rankSum(df)

  echo "df: ", df/4.0/3.0/lo.physVol

when isMainModule:
  qexInit()
  testForce()
  qexFinalize()