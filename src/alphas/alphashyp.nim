import gauge/[hypsmear]
import extra/[qexgauge]

export qexgauge
export hypsmear

template gridBackend*(work: untyped): untyped = 
  when defined(gridDir): work

template qexBackend*(work: untyped): untyped = 
  when not defined(gridDir): work

proc newHYP*(
  alpha1: float,
  alpha2: float,
  alpha3: float
): HypCoefs = HypCoefs(alpha1: alpha1, alpha2: alpha2, alpha3: alpha3)