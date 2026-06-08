"""
Extracts HMC information from all ensembles and stores in nice little JSON files
"""

import os
import sys
import subprocess
import typing
import re
import json
import statistics

LOCATION = 'JLab'
FLAVOR = 'f4'
SMEAR_BCS = ['HISQ_pppa', 'HYP_pppa']
CPATH = os.getcwd()

READCG = False
READPBP = False
HASENBUSCH = True
START_NEW_TRAJECTORY = True
MEASPLAQ = False

COUPLING_CONVERT = {
    '200': '20.0',
    '180': '18.0',
    '160': '16.0',
    '140': '14.0',
    '120': '12.0',
    '100': '10.0',
    '900': '9.00',
    '850': '8.50',
    '800': '8.00',
    '750': '7.50',
    '725': '7.25',
    '700': '7.00',
    '675': '6.75',
    '638': '6.38'
}
MASS_CONVERT = {
    '000'  : '0.000',
    '0005' : '0.005',
    '00025': '0.0025',
    '0001' : '0.001'
}
#DATA = {'48.48.48.96': {'850': ['000']}} #{'20.20.20.40': {'700': ['0001']}}
#"""
DATA = {
    '16.16.16.32': {
        '200': ['000'],
        '160': ['000'],
        '120': ['000'],
        '800': ['000']
    },
    '20.20.20.40': {
        '200': ['000'],
        '180': ['000'],
        '160': ['000'],
        '140': ['000'],
        '120': ['000'],
        '100': ['000'],
        '900': ['000'],
        '850': ['000'],
        '800': ['000'],
        '750': ['000', '0005', '00025', '0001'],
        '725': ['0005', '00025', '0001'],
        '700': ['000', '0005', '00025', '0001'],
        '675': ['000'],
        '638': ['000', '0005', '00025', '0001']
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
        '800': ['000'],
        '750': ['000', '0005', '00025', '0001'],
        '725': ['0005', '00025', '0001'],
        '700': ['000', '0005', '00025', '0001'],
        '675': ['000'],
        '638': ['000', '0005', '00025', '0001']
    },
    '32.32.32.64': {
        '200': ['000'],
	'180': ['000'],
	'160': ['000'],
	'140': ['000'],
	'120': ['000'],
        '110': ['000'],
	'100': ['000'],
        '950': ['000'],
	'900': ['000'],
	'850': ['000'],
	'800': ['000'],
	'750': ['000', '0005', '00025', '0001'],
        '725': ['0005',	'00025', '0001'],
        '700': ['000', '0005', '00025', '0001'],
        '675': ['000'],
        '638': ['000', '0005', '00025', '0001']
    },
    '40.40.40.80': {
        '200': ['000'],
	'160': ['000'],
	'180': ['000'],
	'160': ['000'],
	'140': ['000'],
	'120': ['000'],
        '110': ['000'],
	'100': ['000'],
        '950': ['000'],
	'900': ['000'],
	'850': ['000'],
	'800': ['000'],
	'750': ['000', '0005', '00025', '0001'],
        '725': ['0005',	'00025', '0001'],
        '700': ['000', '0005', '00025', '0001'],
        '675': ['000'],
        '638': ['000', '0005', '00025', '0001'],
    },
    '48.48.48.96': {
        '200': ['000'],
	'180': ['000'],
	'160': ['000'],
	'140': ['000'],
	'120': ['000'],
        '110': ['000'],
	'100': ['000'],
        '950': ['000'],
	'900': ['000'],
	'850': ['000'],
	'800': ['000'],
	'750': ['000', '0005', '00025', '0001'],
        '725': ['0005', '00025', '0001'],
        '700': ['000', '0005', '00025', '0001'],
        '675': ['000'],
        '638': ['000', '0005', '00025', '0001']
    },
    '64.64.64.128': {
        '160': ['000']
    }
}
#"""

LOCAL_DATA = {}
HCGKEY = 'average CG iterations (hasenbusch)'
FCGKEY = 'average CG iterations (fermion)'

SKI: float
SKF: float

SGI: float
SGF: float

SFI: float
SFF: float

def vol(volume: str) -> str: return ''.join(['l', volume.split('.')[0], 't', volume.split('.')[-1]])

def configuration(f: str) -> int: return int(f.split('_')[-1].replace('.log', ''))

def files(path: str) -> (int, list[str]):
    # get thermalization cut (if it exists)
    cut = 0
    cf = '/'.join([path, 'cut'])
    if os.path.isfile(cf):
        with open(cf, 'r') as in_file: cut = int(in_file.readlines()[0])

    # add file to list of files (if log file and above cut)
    result = []
    for	f in os.listdir(path):
        if not f.endswith('.log'): continue
        try:
            configuration(f)
            result.append(f)
        except ValueError: continue
    result.sort(key = configuration)
    return (cut, result)

def finished(content: list[str]) -> bool:
    return ' s] Total time (Init - Finalize): ' in ''.join(content)
    
def simple_measurements(data: dict[str, any], line: list[str]) -> bool:
    global MEASPLAQ
    if not line: return False
    tag = line[0].replace(':', '')
    match tag:
        case 'MEASplaq':
            if not MEASPLAQ:
                MEASPLAQ = not MEASPLAQ
                return True
            data['spatial plaquette'].append(float(line[2]))
            data['temporal plaquette'].append(float(line[4]))
            return True
        case 'MEASploop':
            if not MEASPLAQ: return True
            data['Re[spatial Polyakov loop]'].append(float(line[2]))
            data['Im[spatial Polyakov loop]'].append(float(line[3]))
            data['Re[temporal Polyakov loop]'].append(float(line[5]))
            data['Im[temporal Polyakov loop]'].append(float(line[6]))
            return True
        case 'ACC' | 'REJ':
            dH = float(line[1].replace(',', ''))
            data['dH'].append(dH)
            data['acceptance'].append(1 if tag == 'ACC' else 0)
            return True
        case _: pass
    return False

def complicated_measurements(data: dict[str, any], line: list[str]) -> None:
    global HCGKEY, FCGKEY, LOCAL_DATA
    global READCG, READPBP, HASENBUSCH
    global SKI, SGI, SFI
    global SKF, SGF, SFF
    
    if not line: return None
    tag = line[0].replace(':', '')
    match tag:
        case 'kinetic':
            READCG = not READCG
            if READCG and (HCGKEY in LOCAL_DATA):
                # conjugate gradient information
                data[HCGKEY].append(statistics.mean(LOCAL_DATA[HCGKEY]))
                data[FCGKEY].append(statistics.mean(LOCAL_DATA[FCGKEY]))

                # get action information
                SKF = float(line[1])
                SGF = float(line[3])
                SFF = float(line[5])

                # save initial action information
                data['kinetic-action-initial'].append(SKI)
                data['gauge-action-initial'].append(SGI)
                data['fermion-action-initial'].append(SFI)
                
                # save final action information
                data['kinetic-action'].append(SKF)
                data['gauge-action'].append(SGF)
                data['fermion-action'].append(SFF)

                # save change in action information
                data['kinetic-action-change'].append(SKF - SKI)
                data['gauge-action-change'].append(SGF - SGI)
                data['fermion-action-change'].append(SFF - SFI)
            if READCG:
                # conjugate gradient information
                (LOCAL_DATA[HCGKEY], LOCAL_DATA[FCGKEY]) = ([], [])

                # get initial action information
                SKI = float(line[1])
                SGI = float(line[3])
                SFI = float(line[5])
        case 'stagSolve':
            if not READCG: return None
            if HASENBUSCH:
                LOCAL_DATA[HCGKEY].append(float(line[1]))
                HASENBUSCH = False
            else:
                LOCAL_DATA[FCGKEY].append(float(line[1]))
                HASENBUSCH = True
        case 'MEASpbp': pass
    

def catalogue(volume: str, coupling: str, mass: str, smear_bc: str) -> None:
    global MEASPLAQ, START_NEW_TRAJECTORY
    
    # path information
    ensemble = ''.join([FLAVOR, vol(volume), 'b', coupling, 'm', mass, '_', smear_bc])
    path = '/'.join([CPATH, volume, ensemble, ''])

    # data to be collected
    data = {
        'dH':                         [],
        'kinetic-action':             [],
        'gauge-action':               [],
        'fermion-action':             [],
        'kinetic-action-initial':     [],
        'gauge-action-initial':       [],
        'fermion-action-initial':     [],
        'kinetic-action-change':      [],
        'gauge-action-change':        [],
        'fermion-action-change':      [],
        'spatial plaquette':          [],
        'temporal plaquette':         [],
        'Re[spatial Polyakov loop]':  [],
        'Im[spatial Polyakov loop]':  [],
        'Re[temporal Polyakov loop]': [],
        'Im[temporal Polyakov loop]': [],
        'chiral condensate':          [],
        FCGKEY:                       [],
        HCGKEY:                       [],
        'acceptance':                 [],
        'cut':                        [],
        'per-file-cut':               [],
        'trajectories':               [],
        'hosts':                      [],
        'nodes':                      [],
        'tasks-per-node':             [],
        'threads-per-task':           [],
        'total-time':                 []
    }

    # data collection
    if not os.path.isdir(path): return
    (cut, logs) = files(path)
    for log in logs:
        snapshot = data.copy()
        try:
            # first check for null bytes: this seems to be a problem w/ JLab
            with open(path + log, 'rb') as in_file:
                contents = in_file.read()
                if b'\x00' in contents: raise ValueError("found null bytes")
 
            # if no null bytes, continue
            with open(path + log, 'r') as in_file:
                lines = in_file.readlines()
                if not finished(lines): continue
                for line in lines:
                    spln = line.split()

                    # HMC information
                    simple = simple_measurements(data, spln)
                    if not simple: complicated_measurements(data, spln)
                    if 'kinetic:' in line:
                        if not START_NEW_TRAJECTORY:
                            data['cut'].append(0 if configuration(log) > cut else 1)
                        START_NEW_TRAJECTORY = not START_NEW_TRAJECTORY

                    # generic logging information
                    if 'starting configuration: ' in line:
                        starting_config = int(spln[-1])
                        data['per-file-cut'].append(0 if starting_config > cut else 1)
                    if 'number of trajectories: ' in line:
                        data['trajectories'].append(int(spln[-1]))
                    if 'host: ' in line: data['hosts'].append(spln[-1])
                    if 'nodes: ' in line:
                        try: data['nodes'].append(int(spln[-1]))
                        except ValueError: data['nodes'].append(1)
                    if 'tasks-per-node: ' in line:
                        try: data['tasks-per-node'].append(int(spln[-1]))
                        except ValueError: data['tasks-per-node'].append(1)
                    if 'threads-per-task: ' in line:
                        try: data['threads-per-task'].append(int(spln[-1]))
                        except ValueError: data['threads-per-task'].append(1)
                    if ' Total time (Init - Finalize): ' in line:
                        data['total-time'].append(float(spln[-2]))
                MEASPLAQ = False
        except (IndexError, ValueError):
            # report problem
            print('Warning: Problem reading ' + log)

            # restore defaults
            READCG = False
            READPBP = False
            HASENBUSCH = True
            START_NEW_TRAJECTORY = True
            MEASPLAQ = False
            MEASPLAQ = False

            # restore from snapshot
            data = snapshot.copy()

            # move onto next file
            continue
    data['running'] = ''.join([FLAVOR, vol(volume), 'b', coupling]) in subprocess.run(
        ['squeue', '--format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"'],
        capture_output = True,
        text = True,
        check = True
    ).stdout

    # read in information about ensemble
    with open(path + ensemble + '.json', 'r') as in_file:
        info = json.loads(re.sub(r'(?<=:)\s*0(\d+)', r'"\g<0>"', in_file.read()))
        for key, value in info.items(): data[key] = value
        
    # save data in json format to disk
    dpath = '/'.join([CPATH, 'hmc', ''])
    out_fn = dpath + ensemble + '-info.json'
    if not os.path.isdir(dpath): os.mkdir(CPATH + 'hmc')
    with open(out_fn, 'w+') as out_file: json.dump(data, out_file, indent = 4)

    # Permissions correction
    subprocess.run(['chmod', 'g+rwx', out_fn])
        
    # quick fix to occasional mismatches in size
    if len(data['spatial plaquette']) != len(data[FCGKEY]):
        if len(data['spatial plaquette']) > len(data[FCGKEY]):
            while len(data['spatial plaquette']) != len(data[FCGKEY]):
                del data['spatial plaquette'][-1]
                del data['temporal plaquette'][-1]
                del data['Re[spatial Polyakov loop]'][-1]
                del data['Re[temporal Polyakov loop]'][-1]
                del data['Im[spatial Polyakov loop]'][-1]
                del data['Im[temporal Polyakov loop]'][-1]
                
    # tell user that you've done your job
    print('saved hmc info:', dpath + ensemble + '-info.json')
 
if __name__ == '__main__':
    for volume, vDATA in DATA.items():
        for coupling, cvDATA in vDATA.items():
            for smear_bc in SMEAR_BCS:
                [*map(lambda mass: catalogue(volume, coupling, mass, smear_bc), cvDATA)]
