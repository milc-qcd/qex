import pathlib as path

OUTPUT = path.Path('transfers') / 'stock.txt'
VOLUMES = [16, 20, 24, 32, 36, 40, 48, 64]

def log_file(lat_file):
    ens = str(lat_file).replace('.lat', '')
    srng_file = path.Path(ens + '.parallelRNG')
    prng_file = path.Path(ens + '.serialRNG')
    if srng_file.is_file() and prng_file.is_file():
        out_file.write(ens + '\n')
    
def log_files(ens_dir):
    for f in ens_dir.iterdir():
        if str(f).endswith('.lat'): log_file(f)
            

if __name__ == '__main__':
    with open(OUTPUT, 'w+') as out_file:
        for vol in VOLUMES:
            vol_dir = path.Path('.'.join(str(vol if i != 3 else 2*vol) for i in range(4)))
            for entity in vol_dir.iterdir():
                if entity.is_dir(): log_files(entity)
    
