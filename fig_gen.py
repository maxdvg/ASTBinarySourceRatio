# 26.4.23
# Automated Software Testing ETH Zurich

# Computes [binary file] / [source file] ratios for random C++/C programs
# and produces a simple bar chart showing the distribution of these ratios 

# Assumes that 'simple_driver.sh' has already been run in the directory
# in which this file is called. 'simple_driver.sh' makes CSmith produce
# a bunch of random C/C++ programs. Then these are compiled to binaries.
# The programs produced by CSmith are named simply 'random{i}.c' and their
# binaries which they compiled to are named 'random{i}' where for example 'random3.c'
# is compiled to 'random3'.

import matplotlib.pyplot as plt
import subprocess
import os
from dataclasses import dataclass
import typing
import pathlib
from numpy import histogram

@dataclass
class BinarySrcSizes:
    """
    For a given source code, stores the source code size and the binary size
    * Currently using Bytes as the metric for size but will probably switch to SLOC
    * for src_size
    """
    binary_size: int = None
    src_size: int = None

    @property
    def ratio(self):
        """
        Binary to Source Code ratio
        """
        return self.binary_size / self.src_size

def extract_prgrm_sizes(experiments_dir: pathlib.Path()) -> typing.Dict[int, BinarySrcSizes]:
    """
    Compute the Binary/Source ratio for C++/C files and their binaries within 'experiments_dir'

    :requires: 'simple_driver.sh' already run in 'experiments_dir'. i.e.
    there must be files of the structure:
        + random1.c
        + random1
        + random2.c
        + random2
        ...
        + randomN.c
        + randomN
    in 'experiments_dir', where 'randomN' is the binary which 'randomN.c' was compiled to.
    We refer to each of these pairs of files 'randomN.c' and 'randomN' as an "experiment"

    :param experiments_dir: The directory in which 'simple_driver.sh' was called and therefore
    contains the experimental results which should be analyzed for their binary/source ratios

    :returns: A dictionary mapping from experiment number to the Binary/Source ratio for said
    experiment, e.g. 3 -> 2.1233 inside of the returned dictionary tells us that the Binary/Source
    ratio for 'random3.c' and 'random3' is 2.12333
    """
    files = os.listdir(experiments_dir)
    files = [f for f in files if "random" in f]
    num_to_ratio_map: typing.Dict[int, BinarySrcSizes] = {}
    for file in files:
        src = False
        # trim the random off the front
        num: str = file[len("random"):]
        if num.endswith(".c"):
            # trim the ".c" off the end if it is there [means it is a src file]
            num = num[:-2]
            src = True
        num = int(num)
        # Populate the dictionary with the experiment and the size of the file being analyzed
        if num not in num_to_ratio_map:
            num_to_ratio_map[num] = BinarySrcSizes()
        if src:
            num_to_ratio_map[num].src_size = pathlib.Path(file).stat().st_size
        else:
            num_to_ratio_map[num].binary_size = pathlib.Path(file).stat().st_size
    return num_to_ratio_map

if __name__ == "__main__":
    program_size_map = extract_prgrm_sizes(pathlib.Path('.'))
    # Create a histogram of the ratios for the experiments in the directory
    # from which this program was called
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.hist([v.ratio for v in list(program_size_map.values())], bins='auto')
    fig.savefig('test.png')   # save the figure to file
