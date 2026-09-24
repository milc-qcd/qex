import os
import pathlib
import argparse

STK_FILE = pathlib.Path('transfers') / 'stock.txt'
LOG_FILE = pathlib.Path('transfers') / 'log.txt'
LOG_DIR = pathlib.Path('')

VOLS = [16, 20, 24, 32, 40, 48, 64]

CONFIGS = {}

def remove(lat, srng, prng):
    for f in [lat, srng, prng]:
        if dry_run: out_file.write('would remove: ' + str(f) + '\n')
        else:
            os.remove(f)
            out_file.write('removed: ' + str(f) + '\n')
    print('removed: ', str(lat).replace('.lat', ''))
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "configuration cleanup")
    parser.add_argument(
        '-d',
        '--dir',
        type = str,
        help = 'directory where data is kept'
    )
    parser.add_argument('--dry-run', action = 'store_true')
    args = parser.parse_args()

    data_dir = pathlib.Path(args.dir)
    if not data_dir.is_dir(): exit(0)

    dry_run = args.dry_run
    
    # look through dirs and get ConfigNo
    for vol in VOLS:
        vol_dir = LOG_DIR / '.'.join(str(vol if i < 3 else 2*vol) for i in range(4))
        if vol_dir.is_dir():
            for ens_dir in vol_dir.iterdir():
                if ens_dir.is_dir():
                    config_file = ens_dir / 'ConfigNo'
                    if config_file.is_file():
                        with open(config_file, 'r') as config_file:
                            config = config_file.readlines()[0].replace('\n', '')
                        CONFIGS[str(ens_dir).replace(str(data_dir), '')] = config
                        
    # look through stock file and remove anything in it that's not the last config
    with open(STK_FILE, 'r') as in_file, open(LOG_FILE, 'a') as out_file:    
        for line in in_file.readlines():
            line = line.strip('\n')
            entry = '/'.join(line.split('/')[i] for i in range(2))
            config = line.split('_')[-1]
            if entry in CONFIGS.keys():
                lat_file = data_dir / (line + '.lat')
                srng_file = data_dir / (line + '.parallelRNG')
                prng_file = data_dir / (line + '.serialRNG')
                if all([
                    int(CONFIGS[entry]) != int(config),
                    lat_file.is_file(),
                    srng_file.is_file(),
                    prng_file.is_file()
                ]): remove(lat_file, srng_file, prng_file)
                if int(CONFIGS[entry]) == int(config):
                    for f in [lat_file, srng_file, prng_file]:
                        if dry_run: out_file.write('will keep: ' + str(f) + '\n')
                        else: out_file.write('kept: ' + str(f) + '\n')
