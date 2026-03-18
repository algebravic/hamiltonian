import re
from pathlib import Path
from subprocess import Popen, PIPE
from tempfile import TemporaryDirectory
from typing import Dict, Any
from pysat.formula import CNF, IDPool

GANAKDIR = Path('~/Programming/Satisfiability/ganak').expanduser()

def extract_ganak(val: str) -> int:
    status = None
    result = None
    for line in val.split('\n'):
        # if mymatch := re.match(r's\s+mc\s+([0-9]+)', line):
        #     result = int(mymatch[1])
        if mymatch := re.match(r'c\s+s exact arb int\s+([0-9]+)', line):
            result = int(mymatch[1])
        elif mymatch := re.match(r's\s+SATISFIABLE', line):
            status = True
        elif mymatch := re.match(r's\s+UNSAT', line):
            status = False
        elif mymatch := re.match(r'c\s+o\s+Total time \[Arjun\+GANAK\]:\s*([0-9.]+)', line):
            print(f"Elapsed time = {mymatch[1]} seconds.")
    if status is None:
        print("Incomplete solving status!")
    elif status:
        if result is None:
            print("No count available")
            return -1
        else:
            return result

def ganak_count_models(cnf: CNF, pool: IDPool,
                       projected_prefix: str = '',
                       options: Dict[str, Any] = {}, debug: bool = False) -> int:
    with TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / 'temp.cnf'
        with open(fpath, 'w') as fil:
            if projected_prefix != '':
                pvars = [val for key, val in pool.obj2id.items()
                         if isinstance(key, tuple) and key[0] == projected_prefix]
                fil.write('c t pmc\n')
            else:
                pvars = []
            fil.write(cnf.to_dimacs())
            if pvars:
                fil.write('\n')
                fil.write(f"c p show {' '.join(map(str,pvars))} 0\n")
                          
        # Feed it to ganak
        options = [f'--{key}={val}' for key, val in options.items()]
        process = Popen([f"{GANAKDIR}/ganak"] + options + [f"{fpath}"], stdout = PIPE)
        (output, err) = process.communicate()
        _ = process.wait()
        if debug:
            print(f"Output = {output}")
            print(f"Error = {err}")
        return extract_ganak(output.decode())
