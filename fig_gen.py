import os
import subprocess
import pickle
import pandas as pd
from typing import List, Optional
from pathlib import Path


def run_experiments_on_flags(num_files: int,
                             flags: List[str],
                             results_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:

    # Add csmith to the PATH
    csmith_path = os.path.join(os.environ['HOME'], 'csmith', 'bin')
    os.environ['PATH'] += os.pathsep + csmith_path

    # Create an empty DataFrame if results_df is not provided
    if results_df is None:
        columns = ['Index', 'Original Size', 'Binary Size'] + flags
        results_df = pd.DataFrame(columns=columns)

    # Create random files using csmith and compile them with gcc
    for i in range(1, num_files + 1):
        random_file = f'random{i}.c'
        binary_file = f'random{i}'

        # Find a unique filename for the random C file
        while Path(random_file).exists():
            i += 1
            random_file = f'random{i}.c'

        # Find a unique filename for the binary file
        while Path(binary_file).exists():
            i += 1
            binary_file = f'random{i}'

        # Generate random C code using csmith with the specified flags and values
        csmith_command = ['csmith']
        for flag in flags:
            if '=' in flag:
                flag, value = flag.split('=')
                csmith_command.extend([flag, value])
            else:
                csmith_command.append(flag)
        csmith_command.extend(['>', random_file])
        subprocess.call(' '.join(csmith_command), shell=True)

        # Compile the generated C code using gcc
        gcc_command = f'gcc {random_file} -I{os.environ["HOME"]}/csmith/include -o {binary_file}'
        subprocess.call(gcc_command, shell=True)

        # Get the size of the original C code
        original_size = os.path.getsize(random_file)

        # Get the size of the compiled binary
        binary_size = os.path.getsize(binary_file)

        # Create a dictionary to store the result
        result = {'Index': i, 'Original Size': original_size, 'Binary Size': binary_size}
        result.update({flag: value for flag, value in [f.split('=') for f in flags if '=' in f]})

        # Append the result to the DataFrame
        results_df = results_df.append(result, ignore_index=True)

    return results_df


if __name__ == "__main__":
    num_files: int = 100
    csmith_flags: List[str] = [
        '--stop-by-stmt=35',
        '--max-struct-nested-level',
        '--no-signed-char-index',
        '--max-nested-struct-level=3',
        '--fixed-struct-fields',
        '--no-ptrs=1'
    ]

    # Check if the experiment_results.pkl file exists
    if Path('experiment_results.pkl').exists():
        # Load the DataFrame from the pickle file
        with open('experiment_results.pkl', 'rb') as file:
            results_df: pd.DataFrame = pickle.load(file)
    else:
        # Create a new DataFrame if the file doesn't exist
        results_df: pd.DataFrame = pd.DataFrame()

    # Run experiments on flags and update the results DataFrame
    results_df = run_experiments_on_flags(num_files, csmith_flags, results_df)

    # Save the updated DataFrame to experiment_results.pkl
    with open('experiment_results.pkl', 'wb') as file:
        pickle.dump(results_df, file)