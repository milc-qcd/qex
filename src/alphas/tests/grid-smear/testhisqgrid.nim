import qex
import ../../[alphashisq]
import ../../extra/[qexsmear]
import ../../extra/[gridsmear]

letParam:
  lat = @[8, 8, 8, 16]

proc testSmear = 
  let lo = lat.newLayout()

  var u = lo.newGauge()
  var unit = lo.newGauge()
  var (qsu, qsul) = (lo.newGauge(), lo.newGauge())
  var (gsu, gsul) = (lo.newGauge(), lo.newGauge())

  u.random()
  unit.unit()
  
  var hisq = newHISQ(0.0, 1.0)

  discard qexsmear.smearGetForce(hisq, u, qsu, qsul)
  discard gridsmear.smearGetForce(hisq, u, gsu, gsul)

  var (dsu, dsul) = (0.0, 0.0)

  threads:
    for mu in 0..<u.len:
      for s in u[mu]:
        dsu += redot(qsu[mu][s] - gsu[mu][s], unit[mu][s]).simdSum()
        dsul += redot(qsul[mu][s] - gsul[mu][s], unit[mu][s]).simdSum()

  echo "dsu: ", dsu/3.0/lo.physVol, " dsul: ", dsul/3.0/lo.physVol

when isMainModule:
  qexInit()
  testSmear()
  qexFinalize()