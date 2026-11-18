"""Validate exclusive timing reports and unchanged numerical results."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess

CPU_PHASES = {
    "gauge_flow", "fermion_flow", "contractions", "solve", "hisq_links_flow",
    "hisq_links_solve", "hisq_links_contractions", "gauge_measurements",
    "flow_diagnostics", "source_generation", "source_io", "gauge_input", "output_io",
}
GPU_PHASES = {"flow_setup", "flow_download", "solve_verification", "solver_cleanup"}


def read(path):
    return json.loads(path.read_text())


def inspect_report(work, backend, solver, *, steps=5, intervals=1):
    metadata_path, = work.glob("correlator_*.source.json")
    metadata = read(metadata_path)
    cfg = metadata["configuration"]
    path = work / f"timing_C0p0_{cfg}.json"
    report = read(path)
    assert report["schema_version"] == 1 and report["units"] == "seconds"
    assert report["clock"] == "monotonic wall clock"
    assert report["backend"] == backend and report["solver"] == solver
    assert report["flow_implementation"] == "coupled"
    mode, count = metadata["source-kind"], metadata["source-count"]
    fields = count * (1 if mode == "random-wall" else 3)
    assert report["source_mode"] == mode and report["source_count"] == count
    assert report["solution_fields"] == fields
    assert report["seed"] == metadata["seed"] and report["configuration"] == cfg
    lattice = read(work/"lattice.json")
    ranks = math.prod(lattice["rank-geometry"])
    assert report["mpi_ranks"] == ranks
    phases = report["phases"]
    assert set(phases) == CPU_PHASES | GPU_PHASES | {"other"}
    for name, entry in {**phases, "total":report["total"]}.items():
        low, mean, high = [entry[f"rank_{stat}_seconds"] for stat in ["min","mean","max"]]
        assert all(math.isfinite(v) and v >= 0 for v in [low,mean,high]), (name,entry)
        assert low <= mean + 1e-10 and mean <= high + 1e-10, (name,entry)
        assert 0 <= entry["rank_min_segments"] <= entry["rank_max_segments"]
        assert entry["rank_min_segments"] == entry["rank_max_segments"], (name,entry)
        if name != "total":
            assert entry["description"]
    total = report["total"]["rank_mean_seconds"]
    accounted = sum(entry["rank_mean_seconds"] for entry in phases.values())
    assert abs(total-accounted) <= max(1e-9, total*1e-12), (total,accounted)
    source_saved = all("source-field" in record for record in metadata["sources"])
    # Five RK steps, measured at tau=0 and tau=.1. Three directional
    # Laplacian stages and three shared HISQ constructions per step.
    expected = {
        "gauge_flow": 6*steps+1, "fermion_flow": 3*steps,
        "hisq_links_flow": 3*steps, "hisq_links_solve": 1,
        "hisq_links_contractions": 2 if mode == "random-wall" else 4,
        "contractions": 4 if mode == "random-wall" else 6,
        "solve": fields, "solve_verification": fields if solver=="quda" else 0,
        "solver_cleanup": 1 if solver=="quda" else 0,
        "flow_setup": 2*intervals, "flow_download": steps,
        "gauge_measurements": steps+2, "source_io": fields if source_saved else 0,
        "source_generation": fields+(1 if mode=="random-wall" else 0),
        "gauge_input": 1, "output_io": 3, "flow_diagnostics": 1, "other": 1,
    }
    for phase, segments in expected.items():
        entry = phases[phase]
        assert entry["rank_max_segments"] == segments, (work.name,phase,entry,segments)
        if segments:
            assert entry["rank_max_seconds"] > 0, (work.name,phase)
        else:
            assert entry["rank_max_seconds"] == 0, (work.name,phase)
    text = (work/"run.log").read_text()
    assert "FLOW_TIMING total mean_seconds=" in text
    assert "Flow timing report:" in text
    if backend == "SYCL":
        assert "SYCL device: NVIDIA" in text
        assert "onGpuQ" in report["device_completion"]
    return {"passed":True, "report":str(path), "mpi_ranks":ranks,
            "source_mode":mode, "solution_fields":fields,
            "accounting_error_seconds":abs(total-accounted),
            "phase_count":len(phases)}


def payload(path):
    data = path.read_bytes()
    pos = 0
    while pos < len(data):
        magic, version, flags, size = struct.unpack_from(">IHHQ",data,pos)
        assert magic == 0x456789ab and version == 1
        kind = data[pos+16:pos+144].split(b"\0")[0]
        pos += 144
        if kind == b"scidac-binary-data":
            return data[pos:pos+size]
        pos += size+(-size)%8
    raise AssertionError("missing QIO data")


def compare(a,b):
    maximum = 0.
    def walk(x,y):
        nonlocal maximum
        if isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:
                walk(x[k],y[k])
        elif isinstance(x,list):
            assert len(x)==len(y)
            for u,v in zip(x,y):
                walk(u,v)
        elif isinstance(x,(int,float)):
            assert math.isfinite(x) and math.isfinite(y)
            maximum=max(maximum,abs(x-y))
            assert abs(x-y)<=2e-11+2e-9*abs(y),(x,y)
        else:
            assert x==y
    walk(a,b)
    return maximum


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,
                        help="Check reports in a completed check_wall_sources.py run")
    parser.add_argument("--backend",choices=["CPU","SYCL"],default="CPU")
    parser.add_argument("--solver",choices=["qex","quda"],default="qex")
    parser.add_argument("--executable",type=Path)
    parser.add_argument("--baseline-executable",type=Path)
    parser.add_argument("--unit-gauge",type=Path)
    parser.add_argument("--interacting-gauge",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.results:
        checks=read(args.results/"checks.json")
        assert checks["status"]=="passed"
        cases=[key[:-len("-solves")] for key in checks["checks"] if key.endswith("-solves")]
        reports={name:inspect_report(args.results/name,args.backend,args.solver) for name in cases}
        output=args.results/"timing-checks.json"
        output.write_text(json.dumps({"status":"passed","reports":reports},indent=2)+"\n")
        print(f"PASS {len(reports)} phase reports: {output}",flush=True)
        return
    assert all([args.executable,args.baseline_executable,args.unit_gauge,
                args.interacting_gauge,args.output])
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    report={"status":"running","checks":{}}

    def run(name,binary,mode,times,gauge,ranks,threads,save,intervals):
        work=args.output/name
        work.mkdir()
        stem,cfg,ext=gauge.name.rsplit(".",2)
        flow={"C0p0":{"action":"Wilson","time-increments":[.02]*intervals,
                       "maximum-flow-times":[.1] if intervals==1 else [.04,.1],
                       "loops":1}}
        lattice={"lattice-geometry":[8]*4,"rank-geometry":[ranks,1,1,1],
                 "simd-geometry":[1,2,2,2]}
        (work/"flow.json").write_text(json.dumps(flow))
        (work/"lattice.json").write_text(json.dumps(lattice))
        cmd=["mpirun","-n",str(ranks),str(binary.resolve()),
             f"--flow-json={work}/flow.json",f"--lattice-json={work}/lattice.json",
             f"--configuration={cfg}",f"--base-filename={gauge.resolve().parent/stem}",
             f"--output={work}",f"--source={mode}",f"--solver={args.solver}",
             "--seed=457046288","--source-times="+json.dumps(times,separators=(",",":"))]
        if save:
            cmd.append(f"--save-source={work}/wall.qio")
        (work/"command.json").write_text(json.dumps(cmd,indent=2))
        with (work/"run.log").open("w") as log:
            process=subprocess.Popen(cmd,cwd=work,env=dict(os.environ,OMP_NUM_THREADS=str(threads)),
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                status=process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
                raise AssertionError(f"timeout: {name}")
        assert status==0,(name,(work/"run.log").read_text()[-3000:])
        return work,cfg

    cases=[
        ("random-many","random-wall",[7,7,0],args.unit_gauge,2,2,True,1),
        ("corner-many","corner-wall",[7,0],args.interacting_gauge,2,2,True,1),
        ("random-one","random-wall",[0],args.interacting_gauge,1,1,False,1),
        ("corner-two-intervals","corner-wall",[7],args.unit_gauge,1,1,False,2),
    ]
    for name,mode,times,gauge,ranks,threads,save,intervals in cases:
        old,cfg=run(name+"-untimed",args.baseline_executable,mode,times,gauge,ranks,threads,save,intervals)
        new,_=run(name,args.executable,mode,times,gauge,ranks,threads,save,intervals)
        stats=inspect_report(new,args.backend,args.solver,intervals=intervals)
        stats["correlator_max_absolute_difference"]=compare(
            read(new/f"correlator_C0p0_{cfg}"),read(old/f"correlator_C0p0_{cfg}"))
        stats["gauge_max_absolute_difference"]=compare(
            read(new/f"gaugeobs_C0p0_{cfg}"),read(old/f"gaugeobs_C0p0_{cfg}"))
        if len(times)>1:
            stats["individual_max_absolute_difference"]=compare(
                read(new/f"correlator_C0p0_{cfg}.sources.json"),
                read(old/f"correlator_C0p0_{cfg}.sources.json"))
        a=read(new/f"correlator_C0p0_{cfg}.source.json")
        b=read(old/f"correlator_C0p0_{cfg}.source.json")
        for key in ["seed","source-times","source-count","rng-scheme"]:
            assert a[key]==b[key]
        if save:
            for left,right in zip(a["sources"],b["sources"]):
                assert payload(Path(left["source-field"]))==payload(Path(right["source-field"]))
            stats["source_bytes_identical"]=True
        report["checks"][name]=stats
        print("PASS",name,stats,flush=True)
        (args.output/"checks.json").write_text(json.dumps(report,indent=2)+"\n")
    report["status"]="passed"
    (args.output/"checks.json").write_text(json.dumps(report,indent=2)+"\n")
    print("PASS: timing accounting and untimed regression",flush=True)


if __name__=="__main__":
    main()
