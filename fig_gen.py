import matplotlib.pyplot as plt
import subprocess
import os
from dataclasses import dataclass
import typing
import pathlib
from numpy import histogram

@dataclass
class BinarySrcSizes:
    binary_size: int = None
    src_size: int = None

    @property
    def ratio(self):
        return self.binary_size / self.src_size

def extract_prgrm_sizes() -> typing.Dict[int, BinarySrcSizes]:
    files = os.listdir()
    files = [f for f in files if "random" in f]
    num_to_ratio_map: typing.Dict[int, BinarySrcSizes] = {}
    for file in files:
        src = False
        num = file[len("random"):]
        if num.endswith(".c"):
            num = num[:-2]
            src = True
        num = int(num)
        if num not in num_to_ratio_map:
            num_to_ratio_map[num] = BinarySrcSizes()
        if src:
            num_to_ratio_map[num].src_size = pathlib.Path(file).stat().st_size
        else:
            num_to_ratio_map[num].binary_size = pathlib.Path(file).stat().st_size
    return num_to_ratio_map

if __name__ == "__main__":
    program_size_map = extract_prgrm_sizes()
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.hist([v.ratio for v in list(program_size_map.values())], bins='auto')
    fig.savefig('test.png')   # save the figure to file
