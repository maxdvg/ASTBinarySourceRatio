# Max Van Gelder
# 25/4/23
# Automated Software Testing ETH Spring '23

# Automatically generates C/C++ source code, compiles
# said code, and computes the ratio between the source code
# size and the compiled binary size

#TODO: add pip/python requirements file 
#(currently tested on Python 3.10.6 and no additional packages required)

import subprocess
import pickle
import pathlib
import typing
import tempfile
from config import CREDUCE_PATH, CSMITH_PATH
from dataclasses import dataclass

@dataclass
class BinarySourcePair:
    binary_size: int
    source_size: int
    source_file: pathlib.Path = None
    compiled_file: pathlib.Path = None


def gen_code_and_binary(num_code_statements: float = 10,
                        src_tmp_file: pathlib.Path = pathlib.Path('stmp.C'),
                        cmp_tmp_file: pathlib.Path = pathlib.Path('ctmp')) -> BinarySourcePair:
    if src_tmp_file.exists():
        raise ValueError(f"File {src_tmp_file} already exists. Aborting to avoid overwriting potentially valuable info")
    if cmp_tmp_file.exists():
        raise ValueError(f"File {cmp_tmp_file} already exists. Aborting to avoid overwriting potentially valuable info")
    with open(src_tmp_file, 'w') as random_src_file:
        # Randomly generate a C/C++ source code file using CSmith
        csmith_rand_prog = subprocess.call(f"{CSMITH_PATH} > {src_tmp_file}",
                                          shell=True)
        src_size = src_tmp_file.stat().st_size

        # Compile the randomly generated program
        
        
if __name__ == "__main__":
    gen_code_and_binary()
