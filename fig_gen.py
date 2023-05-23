import os
import subprocess
import pickle
import pandas as pd
from typing import List, Optional
from pathlib import Path
from itertools import combinations
from random import sample
import sys
import re

def expand_numeric_flag_values(flag: str) -> List[str]:
    """Expands flags with multiple numerical values into individual flags.
    Args:
        flag (str): The flag to expand.
    Returns:
        List[str]: A list of expanded flags with each numerical value.
    Examples:
        >>> expand_numeric_flag_values('--stop-by-stmt=[5, 15, 25, 35, 45, 55]')
        ['--stop-by-stmt=5', '--stop-by-stmt=15', '--stop-by-stmt=25', '--stop-by-stmt=35', '--stop-by-stmt=45', '--stop-by-stmt=55']
    """
    matches = re.findall(r"\[(.*?)\]", flag)
    if matches:
        numeric_values = [int(x) for x in matches[0].split(',')]
        expanded_flags = [flag.replace(matches[0], str(val)) for val in numeric_values]
        return expanded_flags
    return [flag]

def run_experiments_on_flags(num_files: int,
                             flags: List[str],
                             results_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Runs experiments on a given set of flags and updates the results DataFrame.

    Args:
        num_files (int): The number of files to generate and analyze.
        flags (List[str]): A list of flags to use for generating the files.
        results_df (Optional[pd.DataFrame], optional): An optional DataFrame to append the results to.
            If not provided, a new DataFrame will be created. Defaults to None.

    Returns:
        results_df (pd.DataFrame): The updated results DataFrame.
    """
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
        '--stop-by-stmt=[5, 15, 25, 35, 45, 55]',
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

    # Generate flag combinations
    all_flag_combinations = []
    for flag in csmith_flags:
        expanded_flags = expand_numeric_flag_values(flag)
        all_flag_combinations.append(expanded_flags)

    all_flag_combinations = list(combinations(*all_flag_combinations))

    # Specify the number of flag combinations to sample
    if len(sys.argv) > 1:
        num_flag_combinations_to_sample = int(sys.argv[1])
    else:
        num_flag_combinations_to_sample = 5  # Default value

    # Randomly sample flag combinations
    sampled_flag_combinations = sample(all_flag_combinations, num_flag_combinations_to_sample)

    # Run experiments on each sampled flag combination and update the results DataFrame
    for flag_combination in sampled_flag_combinations:
        flag_combination = list(flag_combination)
        results_df = run_experiments_on_flags(num_files, flag_combination, results_df)

    # Save the updated DataFrame to experiment_results.pkl
    with open('experiment_results.pkl', 'wb') as file:
        pickle.dump(results_df, file)