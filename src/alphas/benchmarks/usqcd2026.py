import json

BETAS: dict[str, str] = {
    '200': '20.0',
    '180': '18.0',
    '160': '16.0',
    '140': '14.0',
    '120': '12.0',
    '100': '10.0',
    '950': '9.50',
    '900': '9.00',
    '850': '8.50',
    '800': '8.00',
    '750': '7.50',
    '725': '7.25',
    '700': '7.00',
}
MASSES: dict[str, str] = {
    '000':  '0.0e-0',
    '0005': '5.0e-3',
    '00025':'2.5e-3',
    '0001': '1.0e-3',
}
REFERENCES: dict[str, dict[str, str]] = {
    '20.0': {'0.0e-0': 'f4l48t96b200m000_HISQ_pppa'},
    '18.0': {'0.0e-0': 'f4l48t96b180m000_HISQ_pppa'},
    '16.0': {'0.0e-0': 'f4l48t96b160m000_HISQ_pppa'},
    '14.0': {'0.0e-0': 'f4l32t64b140m000_HISQ_pppa'},
    '12.0': {'0.0e-0': 'f4l48t96b120m000_HISQ_pppa'},
    '10.0': {'0.0e-0': 'f4l48t96b100m000_HISQ_pppa'},
    '9.00': {'0.0e-0': 'f4l48t96b900m000_HISQ_pppa'},
    '8.50': {'0.0e-0': 'f4l48t96b850m000_HISQ_pppa'},
    '8.00': {'0.0e-0': 'f4l40t80b800m000_HISQ_pppa'},
    '7.50': {
        '1.0e-3': 'f4l32t64b750m0001_HISQ_pppa',
        '2.5e-3': 'f4l32t64b750m00025_HISQ_pppa',
        '5.0e-3': 'f4l32t64b750m0005_HISQ_pppa'
    },
    '7.25': {
        '1.0e-3': 'f4l32t64b725m0001_HISQ_pppa',
        '2.5e-3': 'f4l32t64b725m00025_HISQ_pppa',
        '5.0e-3': 'f4l32t64b725m0005_HISQ_pppa'
    },
    '7.00': {
        '1.0e-3': 'f4l32t64b700m0001_HISQ_pppa',
        '2.5e-3': 'f4l32t64b700m00025_HISQ_pppa',
        '5.0e-3': 'f4l32t64b700m0005_HISQ_pppa'
    }
}
TRAJECTORIES: dict[str, dict[str, int]] = {
    '20.0': {'0.0e-0': 10},
    '18.0': {'0.0e-0': 10},
    '16.0': {'0.0e-0': 10},
    '14.0': {'0.0e-0': 10},
    '12.0': {'0.0e-0': 10},
    '10.0': {'0.0e-0': 10},
    '9.00': {'0.0e-0': 10},
    '8.50': {'0.0e-0': 10},
    '8.00': {'0.0e-0': 10},
    '7.50': {
        '1.0e-3': 1,
        '2.5e-3': 1,
        '5.0e-3': 10  # last benchmark run covered 10 trajectories
    },
    '7.25': {
        '1.0e-3': 1,
        '2.5e-3': 10,
        '5.0e-3': 1
    },
    '7.00': {
        '1.0e-3': 1,
        '2.5e-3': 1,
        '5.0e-3': 1
    }
}
GB: dict[str, float] = {
    '20': 0.169,
    '24': 0.419,
    '32': 1.100,
    '40': 2.700,
    '48': 5.500,
    '64': 5.500 * (64./48.)**4,  # scaled from '48'
}
DATA: dict[str, dict[str, list[str]]] = {
    '20.20.20.40': {
        '200': ['000'],
        '180': ['000'],
        '160': ['000'],
        '140': ['000'],
        '120': ['000'],
        '100': ['000'],
        '900': ['000'],
        '850': ['000'],
        '800': ['000']
    },
    '24.24.24.48': {
        '200': ['000'],
        '180': ['000'],
        '160': ['000'],
        '140': ['000'],
        '120': ['000'],
        '100': ['000'],
        '900': ['000'],
        '850': ['000'],
        '800': ['000']
    },
    '32.32.32.64': {
        '200': ['000'],
        '180': ['000'],
        '160': ['000'],
        '140': ['000'],
        '120': ['000'],
        '100': ['000'],
        '900': ['000'],
        '850': ['000'],
        '800': ['000']
    },
    '40.40.40.80': {
        '200': ['000'],
        '160': ['000'],
        '180': ['000'],
        '160': ['000'],
        '140': ['000'],
        '120': ['000'],
        '100': ['000'],
        '900': ['000'],
        '850': ['000'],
        '800': ['000']
    },
    '48.48.48.96': {
        '200': ['000'],
        '180': ['000'],
        '160': ['000'],
        '120': ['000'],
        '100': ['000'],
        '900': ['000']
    }
}

class BenchmarkReference:
    def __init__(self, ensemble):
        # ensemble information
        self.m: str = MASSES[ensemble.split('m')[-1].split('_')[0]]
        self.b: str = BETAS[ensemble.split('b')[-1].split('m')[0]]
        self.l: int = int(ensemble.split('l')[-1].split('t')[0])
        self.t: int = int(ensemble.split('t')[-1].split('b')[0])

        # volume (for scaling times)
        self.v: float = float(self.l)**3 * float(self.t)

        # benchmark information
        with open('../data/' + ensemble + '-info.json', 'r') as in_file: 
            data: dict[str, list] = json.load(in_file)
        self.ensemble: str = ensemble
        self.nodes: int = data['nodes'][-1]
        self.tasks_per_node: int = data['tasks-per-node'][-1]
        self.threads_per_task: int = data['threads-per-task'][-1]
        self.cores: int = self.nodes * self.tasks_per_node * self.threads_per_task
        self.hours: float = data['total-time'][-1] / 3600.
        self.host: str = data['hosts'][-1]
        
        # core-hour & service unit computation; see 2026 CfP for service unit info
        # 24s (JLab) is 2.5x faster than LQ1 (the SU reference), so 1 24s node-hour = 2.5 SU
        self.node_hours: float = self.nodes * self.hours / TRAJECTORIES[self.b][self.m]
        if 'jlab' in self.host: self.service_unit: float = self.node_hours * 2.5
        else: self.service_unit: float = self.node_hours

        # storage in GB
        self.gb = GB[str(self.l)]
    
    def volume_service_unit_cost(self, other_v):
        result = self.service_unit * (other_v / self.v)**(5./4.)
        return result

if __name__ == "__main__":
    benchmarks: dict[str, dict[str, BenchmarkReference]] = {}
    want: dict[str, dict[str, dict[str, int]]] = {
        '7.00': {
            '1.0e-3': {'24': 200, '32': 200, '40': 200, '48': 200},
            '2.5e-3': {'24': 200, '32': 200, '40': 200, '48': 200},
            '5.0e-3': {'24': 200, '32': 200, '40': 200, '48': 200}
        },
        '7.125': {
            '1.0e-3': {'24': 200, '32': 200, '40': 200, '48': 200},
            '2.5e-3': {'24': 200, '32': 200, '40': 200, '48': 200},
            '5.0e-3': {'24': 200, '32': 200, '40': 200, '48': 200}
        },
        '7.25': {
            '1.0e-3': {'24': 200, '32': 200, '40': 200, '48': 200},
            '2.5e-3': {'24': 200, '32': 200, '40': 200, '48': 200},
            '5.0e-3': {'24': 200, '32': 200, '40': 200, '48': 200}
        },
        '7.50': {
            '1.0e-3': {'24': 200, '32': 200, '40': 200, '48': 200},
            '2.5e-3': {'24': 200, '32': 200, '40': 200, '48': 200},
            '5.0e-3': {'24': 200, '32': 200, '40': 200, '48': 200}
        },
        '8.25': {'0.0e-0': {'24': 200, '32': 200, '40': 200, '48': 200}},
        '9.00': {'0.0e-0': {'64': 200}},
        '9.50': {'0.0e-0': {'24': 200, '32': 200, '40': 200, '48': 200, '64': 200}},
        '10.0': {'0.0e-0': {'64': 200}},
        '11.0': {'0.0e-0': {'24': 200, '32': 200, '40': 200, '48': 200, '64': 200}},
        '12.0': {'0.0e-0': {'64': 200}},
        '14.0': {'0.0e-0': {'64': 200}},
        '16.0': {'0.0e-0': {'64': 200}},
        '18.0': {'0.0e-0': {'64': 200}},
        '20.0': {'0.0e-0': {'64': 200}}
    }

    # gather reference information from benchmarks
    for beta in REFERENCES.keys():
        benchmarks[beta] = {}
        for mass in REFERENCES[beta].keys():
            benchmarks[beta][mass] = BenchmarkReference(REFERENCES[beta][mass])
    
    # Helper to find nearest betas for interpolation
    # Returns the original string keys (not re-stringified floats) so dict lookups work.
    def find_nearest_betas(beta, beta_list):
        beta_f = float(beta)
        keyed = sorted(beta_list, key=lambda x: float(x))
        lower_str = max((b for b in keyed if float(b) <= beta_f), key=lambda x: float(x), default=keyed[0])
        upper_str = min((b for b in keyed if float(b) >= beta_f), key=lambda x: float(x), default=keyed[-1])
        return lower_str, upper_str

    # Helper to interpolate service units between two betas
    def interpolate_su(beta, beta1, su1, beta2, su2):
        beta = float(beta)
        beta1 = float(beta1)
        beta2 = float(beta2)
        if beta1 == beta2: return su1
        return su1 + (su2 - su1) * (beta - beta1) / (beta2 - beta1)

    # Define volumes to show as columns (union of all volumes in want)
    volumes = ["24", "32", "40", "48", "64"]

    # GB is purely a function of volume; scale from nearest known entry if needed
    def volume_gb(vol_str):
        if vol_str in GB: return GB[vol_str]
        v = int(vol_str)
        nearest = min(GB.keys(), key=lambda k: abs(int(k) - v))
        return GB[nearest] * (v / int(nearest))**4

    # Prepare data: {(beta, mass): {vol: (su, gb)}}
    table_data = {}
    for beta in sorted(want.keys(), key=lambda x: float(x)):
        for mass in sorted(want[beta].keys(), key=lambda x: float(x.replace('e-0','')) if 'e' in x else float(x)):
            row = {}
            for vol in volumes:
                traj = want[beta][mass].get(vol, None)
                if traj is None:
                    row[vol] = (None, None)
                    continue
                su = None
                # GB is always purely from the target volume
                gb = volume_gb(vol) * traj
                # Exact match on beta and mass
                if beta in benchmarks and mass in benchmarks[beta]:
                    ref = benchmarks[beta][mass]
                    ref_vol = str(ref.l)
                    if ref_vol == vol: su = ref.service_unit * traj
                    else:
                        v_ref = float(ref.l)**3 * float(ref.t)
                        v_target = float(vol)**3 * (2.*float(vol))
                        su = ref.service_unit * (v_target / v_ref)**(5./4.) * traj
                else:
                    # Interpolate SU between nearest betas
                    beta1, beta2 = find_nearest_betas(beta, benchmarks.keys())
                    su1 = su2 = None
                    for b, su_var in [(beta1, 'su1'), (beta2, 'su2')]:
                        if b in benchmarks and mass in benchmarks[b]:
                            ref = benchmarks[b][mass]
                            ref_vol = str(ref.l)
                            v_ref = float(ref.l)**3 * float(ref.t)
                            v_target = float(vol)**3 * (2.*float(vol))
                            val_su = ref.service_unit * traj if ref_vol == vol else \
                                     ref.service_unit * (v_target / v_ref)**(5./4.) * traj
                            if su_var == 'su1': su1 = val_su
                            else: su2 = val_su
                    if su1 is not None and su2 is not None:
                        su = interpolate_su(beta, beta1, su1, beta2, su2)
                if su is None: row[vol] = (None, None)
                else: row[vol] = (su, gb)
            table_data[(beta, mass)] = row

    # Generate LaTeX table in requested format
    print("\\begin{table*}[t!]")
    print("\\resizebox{\\textwidth}{!}{%")
    print("\\centering")
    print("\\begin{tabular}{c@{\\extracolsep{4pt}~~~~}cc@{~~~}cc@{~~~}cc@{~~~}cc@{~~~}cc@{~~~}cc}")
    print("\\hline\\hline")
    print("& \\multicolumn{10}{c}{$N_{\\mathrm{s}}$} & \\multicolumn{2}{c}{} \\\\")
    print("\\cline{2-11}")
    print(" & \\multicolumn{2}{c}{24} & \\multicolumn{2}{c}{32} & \\multicolumn{2}{c}{40} & \\multicolumn{2}{c}{48} & \\multicolumn{2}{c}{64} & \\multicolumn{2}{c}{Total} \\\\")
    print("\\cline{2-3} \\cline{4-5} \\cline{6-7} \\cline{8-9} \\cline{10-11} \\cline{12-13}")
    print("$(\\beta_b,am_{f})$ & SU & GB & SU & GB & SU & GB & SU & GB & SU & GB & SU & GB \\\\")
    print(" \\hline ")
    col_totals = {vol: [0.0, 0.0] for vol in volumes}  # [su_total, gb_total]
    grand_total_su = 0.0
    grand_total_gb = 0.0
    for (beta, mass), row in table_data.items():
        label = f"$({beta},\ {float(mass):.4f})$"
        line = [label]
        row_su = 0.0
        row_gb = 0.0
        row_has_data = False
        for vol in volumes:
            su, gb = row[vol]
            if su is None or gb is None:
                line.extend(["--", "--"])
            else:
                line.extend([f"{int(round(su))}", f"{int(round(gb))}"])
                col_totals[vol][0] += su
                col_totals[vol][1] += gb
                row_su += su
                row_gb += gb
                row_has_data = True
        if row_has_data:
            line.extend([f"{int(round(row_su))}", f"{int(round(row_gb))}"])
            grand_total_su += row_su
            grand_total_gb += row_gb
        else:
            line.extend(["--", "--"])
        print(" & ".join(line) + r" \\")
    print("\\hline\\hline")
    total_line = ["Total"]
    for vol in volumes:
        su_t, gb_t = col_totals[vol]
        if su_t == 0.0 and gb_t == 0.0:
            total_line.extend(["--", "--"])
        else:
            total_line.extend([f"{int(round(su_t))}", f"{int(round(gb_t))}"])
    total_line.extend([f"{int(round(grand_total_su))}", f"{int(round(grand_total_gb))}"])
    print(" & ".join(total_line) + r" \\")
    print("\\hline\\hline")
    print("\\end{tabular}}")
    print("\\caption{Summary of our resource request. Save for the last two columns, each row represents a fixed $(\\beta_{b},am_{f})$ pair and each column represents a fixed volume $N_{\\mathrm{s}}^3 \\times 2N_{\\mathrm{s}}$. We show both the service units (SU) and gigabytes (GB) needed to obtain 200 thermalized configurations on each $(\\beta_{b},am_{f},N_{\\mathrm{s}})$ triple. The Total column and bottom row display sums over volumes and $(\\beta_b,am_f)$ pairs, respectively.}\\label{table:request}")
    print("\\end{table*}")

    # ── Second table: regeneration cost vs. storage cost ──────────────
    # 1 TB tape = 30 SU  (see 2026 CfP)
    import glob, os, re as _re

    SU_PER_TB = 30.0
    MIN_CONFIGS = 200      # minimum stored configurations to project to
    TRAJ_PER_CFG = 10      # trajectories between saved configurations

    regen_rows = []
    for filepath in sorted(glob.glob('../data/f4l*-info.json')):
        basename = os.path.basename(filepath).replace('-info.json', '')
        m_ens = _re.match(r'f4l(\d+)t(\d+)b(\d+)m(\w+)_HISQ_pppa', basename)
        if not m_ens:
            continue
        ns, nt = int(m_ens.group(1)), int(m_ens.group(2))
        beta_key, mass_key = m_ens.group(3), m_ens.group(4)

        # Only volumes >= 32
        if ns < 32:
            continue

        beta_str = BETAS.get(beta_key, beta_key)
        mass_str = MASSES.get(mass_key, mass_key)

        with open(filepath) as fh:
            info = json.load(fh)

        pfc = info.get('per-file-cut', [])
        if not pfc:
            continue

        # Thermalized (post-cut) configurations: entries with per-file-cut == 0
        actual_cfgs = sum(1 for p in pfc if p == 0)
        # Project to at least MIN_CONFIGS stored configurations
        store_cfgs  = max(actual_cfgs, MIN_CONFIGS)
        projected   = store_cfgs > actual_cfgs

        # Target volume for scaling benchmark costs
        v_target = float(ns)**3 * float(nt)

        # ── SU per trajectory at target volume from benchmark references ──
        su_per_traj = None
        if beta_str in benchmarks and mass_str in benchmarks[beta_str]:
            ref = benchmarks[beta_str][mass_str]
            su_per_traj = ref.volume_service_unit_cost(v_target)
        else:
            # Interpolate between nearest betas that have this mass
            avail = [b for b in benchmarks if mass_str in benchmarks[b]]
            if avail:
                beta1, beta2 = find_nearest_betas(beta_str, avail)
                su1 = su2 = None
                if beta1 in benchmarks and mass_str in benchmarks[beta1]:
                    su1 = benchmarks[beta1][mass_str].volume_service_unit_cost(v_target)
                if beta2 in benchmarks and mass_str in benchmarks[beta2]:
                    su2 = benchmarks[beta2][mass_str].volume_service_unit_cost(v_target)
                if su1 is not None and su2 is not None:
                    su_per_traj = interpolate_su(beta_str, beta1, su1, beta2, su2)
                elif su1 is not None:
                    su_per_traj = su1
                elif su2 is not None:
                    su_per_traj = su2

        if su_per_traj is None:
            continue

        # One stored configuration = TRAJ_PER_CFG trajectories
        su_per_cfg = su_per_traj * TRAJ_PER_CFG
        regen_su   = store_cfgs * su_per_cfg

        gb_per_cfg = GB.get(str(ns), 0)
        store_gb   = store_cfgs * gb_per_cfg
        store_su   = store_gb / 1000.0 * SU_PER_TB

        regen_rows.append({
            'ns': ns, 'nt': nt,
            'beta': beta_str, 'mass': mass_str,
            'actual_cfgs': actual_cfgs,
            'store_cfgs': store_cfgs,
            'projected': projected,
            'regen_su': regen_su,
            'store_gb': store_gb,
            'store_su': store_su,
        })

    # ── Also include volumes from 'want' that have no data file yet ──
    existing_keys = {(r['ns'], r['beta'], r['mass']) for r in regen_rows}
    for beta in want:
        for mass in want[beta]:
            for vol_str in want[beta][mass]:
                ns = int(vol_str)
                nt = 2 * ns
                if ns < 32:
                    continue
                if (ns, beta, mass) in existing_keys:
                    continue
                v_target = float(ns)**3 * float(nt)
                su_per_traj = None
                if beta in benchmarks and mass in benchmarks[beta]:
                    ref = benchmarks[beta][mass]
                    su_per_traj = ref.volume_service_unit_cost(v_target)
                else:
                    avail = [b for b in benchmarks if mass in benchmarks[b]]
                    if avail:
                        beta1, beta2 = find_nearest_betas(beta, avail)
                        su1 = su2 = None
                        if beta1 in benchmarks and mass in benchmarks[beta1]:
                            su1 = benchmarks[beta1][mass].volume_service_unit_cost(v_target)
                        if beta2 in benchmarks and mass in benchmarks[beta2]:
                            su2 = benchmarks[beta2][mass].volume_service_unit_cost(v_target)
                        if su1 is not None and su2 is not None:
                            su_per_traj = interpolate_su(beta, beta1, su1, beta2, su2)
                        elif su1 is not None:
                            su_per_traj = su1
                        elif su2 is not None:
                            su_per_traj = su2
                if su_per_traj is None:
                    continue
                store_cfgs = MIN_CONFIGS
                su_per_cfg = su_per_traj * TRAJ_PER_CFG
                regen_su   = store_cfgs * su_per_cfg
                gb_per_cfg = volume_gb(vol_str)
                store_gb   = store_cfgs * gb_per_cfg
                store_su   = store_gb / 1000.0 * SU_PER_TB
                regen_rows.append({
                    'ns': ns, 'nt': nt,
                    'beta': beta, 'mass': mass,
                    'actual_cfgs': 0,
                    'store_cfgs': store_cfgs,
                    'projected': True,
                    'regen_su': regen_su,
                    'store_gb': store_gb,
                    'store_su': store_su,
                })

    # Sort: volume ascending, beta descending, mass ascending
    regen_rows.sort(key=lambda r: (r['ns'], -float(r['beta']), float(r['mass'])))

    # ── Aggregate by volume ─────────────────────────────────────────
    from collections import defaultdict
    vol_agg = defaultdict(lambda: {
        'n_ens': 0, 'n_projected': 0,
        'total_cfgs': 0, 'regen_su': 0.0,
        'store_gb': 0.0, 'store_su': 0.0,
    })
    for row in regen_rows:
        a = vol_agg[row['ns']]
        a['n_ens']       += 1
        a['n_projected'] += int(row['projected'])
        a['total_cfgs']  += row['store_cfgs']
        a['regen_su']    += row['regen_su']
        a['store_gb']    += row['store_gb']
        a['store_su']    += row['store_su']

    # ── Print second LaTeX table (compact, one row per volume) ────────
    print()
    print("\\begin{table}[t!]")
    print("\\centering")
    print("\\begin{tabular}{ccccccc}")
    print("\\hline\\hline")
    print("$N_{\\mathrm{s}}$ & Ens. & Cfgs"
          " & Regen (SU) & Store (GB) & Store (SU)"
          " & Regen/Store \\\\")
    print("\\hline")

    total_regen_su = 0.0
    total_store_gb = 0.0
    total_store_su = 0.0
    total_ens = 0
    total_cfgs = 0
    for ns in sorted(vol_agg.keys()):
        a = vol_agg[ns]
        ratio = (a['regen_su'] / a['store_su']
                 if a['store_su'] > 0 else float('inf'))
        proj = '$^\\dagger$' if a['n_projected'] > 0 else ''
        line = [
            str(ns),
            str(a['n_ens']),
            f"{a['total_cfgs']}{proj}",
            f"{int(round(a['regen_su']))}",
            f"{int(round(a['store_gb']))}",
            f"{a['store_su']:.1f}",
            f"{ratio:.0f}",
        ]
        print(" & ".join(line) + r" \\")
        total_regen_su += a['regen_su']
        total_store_gb += a['store_gb']
        total_store_su += a['store_su']
        total_ens      += a['n_ens']
        total_cfgs     += a['total_cfgs']

    print("\\hline\\hline")
    total_ratio = (total_regen_su / total_store_su
                   if total_store_su > 0 else float('inf'))
    total_line = [
        "Total",
        str(total_ens),
        str(total_cfgs),
        f"{int(round(total_regen_su))}",
        f"{int(round(total_store_gb))}",
        f"{total_store_su:.1f}",
        f"{total_ratio:.0f}",
    ]
    print(" & ".join(total_line) + r" \\")
    print("\\hline\\hline")
    print("\\end{tabular}")
    print("\\caption{Cost to regenerate vs.\\ store existing and"
          " planned ensembles ($N_{\\mathrm{s}} \\geq 32$),"
          " aggregated by spatial volume (1~TB $= 30$~SU)."
          "  ``Ens.'' is the number of $(\\beta_b, am_f)$"
          " ensembles at each $N_{\\mathrm{s}}$."
          "  ``Cfgs'' is the total number of stored configurations"
          " (each separated by 10 trajectories);"
          " a $^\\dagger$ indicates that at least one ensemble"
          " in the group has been projected to a minimum of"
          " 200 stored configurations."
          "  Regen~(SU) is the benchmark-derived service-unit"
          " cost to reproduce these configurations."
          "  Store~(SU) converts the tape footprint at the"
          " 30~SU/TB rate."
          "}\\label{table:regen-vs-store}")
    print("\\end{table}")

    # ── Third table: thermalized configuration census ─────────────────
    import glob, os, re

    census_rows = []
    for filepath in sorted(glob.glob('../data/f4l*-info.json')):
        basename = os.path.basename(filepath).replace('-info.json', '')
        # parse ensemble name: f4l{Ns}t{Nt}b{beta_key}m{mass_key}_HISQ_pppa
        m = re.match(r'f4l(\d+)t(\d+)b(\d+)m(\w+)_HISQ_pppa', basename)
        if not m:
            continue
        ns, nt = int(m.group(1)), int(m.group(2))
        beta_key, mass_key = m.group(3), m.group(4)
        beta_str = BETAS.get(beta_key, beta_key)
        mass_str = MASSES.get(mass_key, mass_key)

        # Skip non-zero mass ensembles
        if mass_key != '000':
            continue

        with open(filepath) as fh:
            info = json.load(fh)

        pfc      = info.get('per-file-cut', [])
        traj_arr = info.get('trajectories', [])
        n_entries = len(pfc)

        total_traj   = sum(traj_arr)
        n_therm      = sum(1 for p in pfc if p == 0)
        n_cut        = sum(pfc)

        census_rows.append({
            'ns': ns, 'nt': nt,
            'beta': beta_str, 'mass': mass_str,
            'total_entries': n_entries,
            'total_traj': total_traj,
            'therm_cfgs': n_therm,
            'cut_cfgs': n_cut,
        })

    # Pivot census: rows = Ns, columns = beta (descending)
    all_ns    = sorted(set(r['ns'] for r in census_rows))
    all_betas = sorted(set(r['beta'] for r in census_rows),
                       key=lambda x: -float(x))

    # Build lookup: (ns, beta) -> therm_cfgs
    census_lookup = {}
    for r in census_rows:
        census_lookup[(r['ns'], r['beta'])] = r['therm_cfgs']

    print()
    print("\\begin{table*}[t!]")
    print("\\centering")
    col_spec = "c" * (len(all_betas) + 1)   # Ns + beta columns
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\hline\\hline")

    header = ["$N_{\\mathrm{s}}$"]
    for beta in all_betas:
        header.append(f"${beta}$")
    print("& \\multicolumn{" + str(len(all_betas))
          + "}{c}{$\\beta_b$} \\\\")
    print("\\cline{2-" + str(len(all_betas) + 1) + "}")
    print(" & ".join(header) + r" \\")
    print("\\hline")

    for ns in all_ns:
        line = [f"${ns}$"]
        for beta in all_betas:
            val = census_lookup.get((ns, beta), None)
            if val is not None:
                line.append(str(val))
            else:
                line.append("--")
        print(" & ".join(line) + r" \\")
    print("\\hline\\hline")
    print("\\end{tabular}")
    print("\\caption{Census of thermalized configurations"
          " across all existing pure-gauge ensembles ($am_f = 0$)."
          "  Each column is a bare coupling $\\beta_b$"
          " and each row is a spatial extent $N_{\\mathrm{s}}$."
          "  Entries are the number of post-cut configurations"
          " (\\texttt{per-file-cut}${}=0$)."
          "}\\label{table:census}")
    print("\\end{table*}")
