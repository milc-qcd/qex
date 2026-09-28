import os
import pathlib as path

OUTPUT = path.Path('logs') / 'sanitize.log'
VOLUMES = [16, 20, 24, 32, 40, 48, 64]

def remove(f):
    # remove ordinary file
    f.unlink()
    out_file.write('Removed: ' + str(f) + '\n')

    # remove Emacs backup
    emacs_f = path.Path(str(f) + '~')
    if emacs_f.is_file():
        emacs_f.unlink()
        out_file.write('Removed: ' + str(emacs_f) + '\n')
        
def sanitize(ens_dir):
    print('working on: ', ens_dir)
    for f in ens_dir.iterdir():
        if str(f).endswith('~'): remove(f)
        if str(f).endswith('.out'): remove(f)

if __name__ == '__main__':
    with open(OUTPUT, 'a') as out_file:
        for vol in VOLUMES:
            vol_dir = path.Path('.'.join(str(vol if i != 3 else 2*vol) for i in range(4)))
            for entity in vol_dir.iterdir():
                if entity.is_dir(): sanitize(entity)
