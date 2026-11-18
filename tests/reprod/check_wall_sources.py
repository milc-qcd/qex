"""CPU/GPU wall-source integration: exact fields, CPU agreement, and MPI replay."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess


class Milc6:
    def __init__(self, seed, stream):
        self.r = []
        for _ in range(7):
            seed = ((69607 + 8*stream)*seed + 12345) & 0xffffffff
            self.r.append((seed >> 8) & 0xffffff)
        self.state = ((69607 + 8*stream)*seed + 12345) & 0xffffffff
        self.multiplier = (100005 + 8*stream) & 0xffffffff

    def integer(self):
        r = self.r
        v = (((r[5] >> 7) | (r[6] << 17)) ^
             ((r[4] >> 1) | (r[5] << 23))) & 0xffffff
        self.r = [v] + r[:6]
        self.state = (self.state*self.multiplier + 12345) & 0xffffffff
        return v ^ ((self.state >> 8) & 0xffffff)


def draw(rng, extent):
    limit = 2**24 - 2**24 % extent
    while True:
        value = rng.integer()
        if value < limit:
            return value % extent


def read(path):
    return json.loads(path.read_text())


def payload(path):
    data = path.read_bytes()
    pos = 0
    while pos < len(data):
        magic, version, flags, size = struct.unpack_from(">IHHQ", data, pos)
        assert magic == 0x456789ab and version == 1
        kind = data[pos+16:pos+144].split(b"\0")[0]
        pos += 144
        if kind == b"scidac-binary-data":
            return data[pos:pos+size]
        pos += size + (-size) % 8
    raise AssertionError("no QIO source payload")


def execute(command, work, threads=2, success=True):
    (work/"command.json").write_text(json.dumps(command, indent=2))
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    with (work/"run.log").open("w") as log:
        process = subprocess.Popen(command, cwd=work, stdout=log,
                                   stderr=subprocess.STDOUT, env=env,
                                   start_new_session=True)
        try:
            status = process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise AssertionError(f"timeout: {work.name}")
    text = (work/"run.log").read_text()
    assert (status == 0) == success, (work.name, status, text[-2500:])
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--unit-gauge", required=True, type=Path)
    parser.add_argument("--interacting-gauge", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--solver", choices=["qex", "quda"], default="qex")
    parser.add_argument("--reference-executable", type=Path)
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument("--old-corner-executable", type=Path)
    parser.add_argument("--expect-gpu", action="store_true")
    args = parser.parse_args()
    assert bool(args.reference_executable) != bool(args.reference_results)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    refs = args.reference_results or (args.output/"cpu-reference")
    if args.reference_executable:
        refs.mkdir()
    checks = {}
    report = {"status": "running", "solver": args.solver, "checks": checks,
              "executable": str(args.executable.resolve()),
              "executable_sha256": hashlib.sha256(args.executable.read_bytes()).hexdigest()}

    def record():
        (args.output/"checks.json").write_text(json.dumps(report, indent=2)+"\n")

    def compare(a, b, label, common=False):
        maximum = 0.
        def walk(x,y):
            nonlocal maximum
            if isinstance(x, dict):
                if not common:
                    assert x.keys() == y.keys(), (label,x.keys(),y.keys())
                else:
                    assert y.keys() <= x.keys(), (label,x.keys(),y.keys())
                for key in y:
                    walk(x[key],y[key])
            elif isinstance(x,list):
                assert len(x) == len(y)
                for u,v in zip(x,y):
                    walk(u,v)
            else:
                assert math.isfinite(x) and math.isfinite(y), label
                maximum = max(maximum,abs(x-y))
                assert abs(x-y) <= 2e-10 + 2e-8*abs(y), (label,x,y)
        walk(a,b)
        checks[label] = {"max_absolute_difference": maximum}
        record()

    def inputs(work, gauge, ranks):
        stem, cfg, ext = gauge.name.rsplit(".",2)
        assert ext == "ildg"
        (work/"flow.json").write_text(json.dumps({"C0p0": {
            "action":"Wilson", "time-increments":[0.02],
            "maximum-flow-times":[0.1], "loops":1}}))
        (work/"lattice.json").write_text(json.dumps({
            "lattice-geometry":[8]*4, "rank-geometry":[ranks,1,1,1],
            "simd-geometry":[1,2,2,2]}))
        return cfg, [
            f"--flow-json={work}/flow.json", f"--lattice-json={work}/lattice.json",
            f"--base-filename={gauge.parent/stem}", f"--configuration={cfg}",
            f"--output={work}"]

    def validate_fields(work, metadata, mode, times, corner):
        seed = metadata["seed"]
        n = metadata["source-count"]
        time_rng = Milc6(seed,8**4)
        expected_times = [draw(time_rng,8) for _ in range(n)]
        if times is not None:
            expected_times = times
        assert metadata["source-times"] == expected_times
        fields = []
        if mode == "random-wall":
            fields = [bytearray() for _ in range(n)]
            values = [(1.,0.),(0.,1.),(-1.,0.),(0.,-1.)]
            for site in range(8**4):
                rng = Milc6(seed,site)
                for hit in range(n):
                    for color in range(3):
                        value = values[rng.integer() >> 22]
                        if site//512 != expected_times[hit]:
                            value = (0.,0.)
                        fields[hit].extend(struct.pack(">dd",*value))
        else:
            rng = Milc6(seed,8**4+1)
            corners = [draw(rng,8) for _ in range(n)]
            if corner is not None:
                corners = [corner]*n
            assert metadata["source-corners"] == corners
            for hit in range(n):
                for color in range(3):
                    field = bytearray()
                    c = corners[hit]
                    for site in range(8**4):
                        x,y,z,t = site%8,(site//8)%8,(site//64)%8,site//512
                        active = (t == expected_times[hit] and x%2 == c%2
                                  and y%2 == (c//2)%2 and z%2 == c//4)
                        for k in range(3):
                            field.extend(struct.pack(">dd",float(active and k == color),0.))
                    fields.append(field)
        assert len(metadata["sources"]) == len(fields)
        for slot,expected in enumerate(fields):
            info = metadata["sources"][slot]
            assert info["source-field-verified"]
            assert info["source-norm2"] == (1536 if mode == "random-wall" else 64)
            assert payload(Path(info["source-field"])) == expected, (work.name,slot)
        checks[work.name+"-exact-fields"] = {"times":expected_times,"count":len(fields)}
        record()

    def run(name, mode="random-wall", n=4, seed=457046288, times=None,
            corner=None, ranks=2, threads=2, gauge=None, reference=True):
        gauge = (gauge or args.unit_gauge).resolve()
        work = args.output/name
        work.mkdir()
        cfg, options = inputs(work,gauge,ranks)
        source_options = [f"--source={mode}"]
        if n is not None:
            source_options.append(f"--sources={n}")
        if seed is not None:
            source_options.append(f"--seed={seed}")
        if times is not None:
            source_options.append("--source-times="+json.dumps(times,separators=(",",":")))
        if corner is not None:
            source_options.append(f"--source-corner={corner}")
        text = execute(["mpirun","-n",str(ranks),str(args.executable.resolve()),
                        *options,*source_options,f"--solver={args.solver}",
                        f"--save-source={work}/wall.qio"],work,threads)
        metadata = read(work/f"correlator_C0p0_{cfg}.source.json")
        count = len(times) if n is None and times is not None else n
        assert metadata["source-count"] == count
        assert metadata["inversions-per-flavor"] == count*(1 if mode=="random-wall" else 3)
        if seed is not None:
            assert metadata["seed"] == seed
        validate_fields(work,metadata,mode,times,corner)
        if args.solver == "quda":
            residuals = [float(x) for x in re.findall(
                r"QUDA true relative residual squared: (\S+)",text)]
        else:
            residuals = [float(x) for x in re.findall(
                r"^stagSolve:.*?([0-9.eE+-]+)\s*$",text,re.M)]
        assert len(residuals) == metadata["inversions-per-flavor"], (name,residuals)
        assert all(math.isfinite(x) and 0<=x<=1.01e-20 for x in residuals)
        if args.expect_gpu:
            assert "SYCL device: NVIDIA" in text, "GPU device was not selected"
        checks[name+"-solves"] = {"count":len(residuals),"max_r2":max(residuals)}
        data = read(work/f"correlator_C0p0_{cfg}")
        assert set(data) == {"0.00","0.10"}
        individual = (read(work/f"correlator_C0p0_{cfg}.sources.json") if count>1
                      else [{"correlators":data}])
        if count>1:
            mean = {tau:{ch:[sum(hit["correlators"][tau][ch][t] for hit in individual)/count
                            for t in range(8)] for ch in data[tau]} for tau in data}
            compare(data,mean,name+"-mean")
        for hit in individual:
            assert hit["correlators"]["0.00"] != hit["correlators"]["0.10"]
        if reference:
            ref = refs/name
            signature = {"options":source_options,"gauge_sha256":hashlib.sha256(gauge.read_bytes()).hexdigest()}
            if args.reference_executable:
                ref.mkdir()
                ref_cfg,ref_options = inputs(ref,gauge,2)
                execute(["mpirun","-n","2",str(args.reference_executable.resolve()),
                         *ref_options,*source_options],ref)
                (ref/"params.json").write_text(json.dumps(signature,indent=2))
            else:
                assert read(ref/"params.json") == signature
            expected = read(ref/f"correlator_C0p0_{cfg}")
            compare(data,expected,name+"-CPU-correlators",common=True)
            if count>1:
                ref_hits = read(ref/f"correlator_C0p0_{cfg}.sources.json")
                for hit in range(count):
                    compare(individual[hit]["correlators"],ref_hits[hit]["correlators"],
                            name+f"-CPU-hit{hit}",common=True)
            compare(read(work/f"gaugeobs_C0p0_{cfg}"),
                    read(ref/f"gaugeobs_C0p0_{cfg}"),name+"-CPU-gauge")
        print("PASS",name,metadata["source-times"],flush=True)
        record()
        return data,metadata,work

    for mode in ["random-wall","corner-wall"]:
        first,metadata,work = run(mode+"-random",mode=mode,n=4)
        replay,_,_ = run(mode+"-rank1",mode=mode,n=4,ranks=1,threads=1,reference=False)
        compare(replay,first,mode+"-MPI-thread-replay")
        listed,_,_ = run(mode+"-listed",mode=mode,n=None,times=metadata["source-times"],reference=False)
        compare(listed,first,mode+"-explicit-list-replay")
        run(mode+"-boundary",mode=mode,n=3,seed=0,times=[7,7,0])
        if args.interacting_gauge:
            run(mode+"-interacting",mode=mode,n=2,seed=2,times=[7,0],gauge=args.interacting_gauge)

    run("corner-fixed",mode="corner-wall",n=2,times=[0,7],corner=5)
    clock,meta,_ = run("clock",n=2,seed=None,reference=False)
    replay,_,_ = run("clock-replay",n=2,seed=meta["seed"],reference=False)
    compare(replay,clock,"clock-seed-replay")

    if args.old_corner_executable:
        new,_,_ = run("corner-legacy",mode="corner-wall",n=1,times=[7],corner=5,reference=False)
        old = args.output/"corner-old"
        old.mkdir()
        cfg,options = inputs(old,args.unit_gauge.resolve(),2)
        execute(["mpirun","-n","2",str(args.old_corner_executable.resolve()),
                 *options,"--seed=457046288","--source=corner-wall",
                 "--source-time=7","--source-corner=5"],old)
        compare(new,read(old/f"correlator_C0p0_{cfg}"),"all-legacy-corner-channels",common=True)

    for index,option in enumerate([
        "--sources=0","--sources=-1","--sources=1.5","--source-times=[]",
        "--source-times=[0,1.5]","--sources=3 --source-times=[0,1]",
        "--source-times=[8]","--source-times=[-1]","--source-corner=0",
        "--seed=-1","--seed=4294967296",
        "--source-time=1 --source-times=[1]"]):
        work = args.output/f"invalid-{index}"
        work.mkdir()
        cfg,options = inputs(work,args.unit_gauge.resolve(),2)
        execute(["mpirun","-n","2",str(args.executable.resolve()),*options,
                 "--source=random-wall",*option.split()],work,success=False)
        assert not (work/f"correlator_C0p0_{cfg}.source.json").exists()
    work = args.output/"no-overwrite"
    work.mkdir()
    cfg,options = inputs(work,args.unit_gauge.resolve(),2)
    original = args.output/"random-wall-random/wall.source0000.qio"
    before = original.read_bytes()
    execute(["mpirun","-n","2",str(args.executable.resolve()),*options,
             "--source=random-wall","--sources=4","--seed=457046288",
             f"--save-source={original.parent}/wall.qio"],work,success=False)
    assert original.read_bytes() == before
    checks["invalid-options-and-QIO-preservation"] = True
    report["status"] = "passed"
    record()
    print("PASS: all multi-source wall checks",flush=True)


if __name__ == "__main__":
    main()
