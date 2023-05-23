"""
Random Code Generation and Analysis

Let me tell you folks, this incredible Python script right here generates the most amazing, tremendous, and absolutely random C code
using the csmith tool. It's fantastic, believe me! We compile this code with gcc, and let me tell you, it's a perfect compilation, just perfect.
Then, we perform some really smart analysis on the generated binary files. It's unbelievable what we can do!

Now, this script allows us to run experiments with different combinations of csmith flags. And let me tell you, these flags are tremendous,
the best flags you'll ever see. We store the results of these experiments in a powerful Pandas DataFrame. It's a beautiful DataFrame, folks!

But that's not all. We can also load existing results from a pickle file, run new experiments with different flag combinations,
and save the updated results back to the pickle file. It's incredible, really incredible!

And here's the best part. We have functions to generate a histogram of the ratio of binary size to source code size. We fit a Gaussian
distribution to this histogram. It's a distribution like no other, folks, the greatest distribution you'll ever see. And we calculate
the mean and standard deviation for each combination of flags. It's very, very smart, I can assure you!

So, this script is divided into several functions, really amazing functions. It's gonna make your head spin,
believe me. And in the "main" section, we showcase the usage of these functions. It's tremendous, folks!

This incredible script was written by the amazing Max Van Gelder. A very smart person, let me tell you.
And the date, folks, is 23.5.23. That's right, we're looking into the future!

Thank you, thank you very much!

This Python script generates random C code using the csmith tool, compiles the generated code using gcc,
and performs analysis on the generated binary files. It allows for running experiments with different
combinations of csmith flags and storing the results in a Pandas DataFrame. The script also provides
functionality to load the existing results DataFrame from a pickle file, run experiments with new flag
combinations, and save the updated results back to the pickle file. Additionally, it includes functions
to generate a histogram of the ratio of binary size to source code size, fit a Gaussian distribution
to the histogram, and calculate the mean and standard deviation for each combination of flags.

The script is divided into several functions:

- expand_numeric_flag_values(flag): Expands flags with multiple numerical values into individual flags.
- run_experiments_on_flags(num_files, flags, results_df): Runs experiments on a given set of flags
  and updates the results DataFrame.
- generate_histogram(results_df, flags): Generates a histogram of the ratio of binary size to source
  code size, fits a Gaussian distribution, and prints statistics.
- calculate_mean_std(results_df): Calculates the mean and standard deviation for each combination of flags.

The "__main__" section of the script demonstrates the usage of these functions. It allows for command-line
arguments to specify the number of flag combinations to sample for experiments. The results DataFrame is
loaded from an existing pickle file if available, and new experiments are run on sampled flag combinations.
The updated results are then saved back to the pickle file.

Author: Max Van Gelder
Date: 23.5.23
"""


import os
import subprocess
import pickle
import pandas as pd
from typing import List, Optional
from pathlib import Path
from itertools import combinations
from random import sample
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
import re
from typing import Dict


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

def calculate_mean_std(results_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculates the mean and standard deviation for each combination of flags.

    Args:
        results_df (pd.DataFrame): The DataFrame containing the experiment results.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary mapping each combination of flags to
            the mean and standard deviation of the experiments with that combination.

    Raises:
        ValueError: If the DataFrame does not contain any flag columns.
    """
    # Get the flag columns from the DataFrame
    flag_columns = [col for col in results_df.columns if col != 'Binary Size' and col != 'Original Size']

    # Check if any flag columns exist
    if not flag_columns:
        raise ValueError("The DataFrame does not contain any flag columns.")

    # Calculate the mean and standard deviation for each combination of flags
    result_dict = {}
    for _, row in results_df.iterrows():
        combination = tuple(row[flag_columns].values)
        size_ratio = row['Binary Size'] / row['Original Size']
        if combination in result_dict:
            result_dict[combination]['mean'].append(size_ratio)
        else:
            result_dict[combination] = {'mean': [size_ratio], 'std_dev': []}

    # Calculate mean and standard deviation for each combination
    for combination, values in result_dict.items():
        values['mean'] = np.mean(values['mean'])
        values['std_dev'] = np.std(values['mean'])

    return result_dict

def generate_histogram(results_df: pd.DataFrame, flags: List[str]) -> None:
    """Generates a histogram of the ratio of binary size to source code size,
    fits a Gaussian distribution, and prints statistics.

    Args:
        results_df (pd.DataFrame): The DataFrame containing the experiment results.
        flags (List[str]): The list of flags used for the experiments.

    Returns:
        None

    Raises:
        ValueError: If no entries matching the given flags are found in the dataframe.
    """
    # Filter the dataframe to get entries matching the given flags
    filtered_df = results_df.copy()
    for flag in flags:
        if flag not in filtered_df.columns:
            raise ValueError(f"No entries matching the flag '{flag}' found in the dataframe.")
        filtered_df = filtered_df[filtered_df[flag].notnull()]

    # Check if any entries matching the given flags are found
    if filtered_df.empty:
        raise ValueError("No entries matching the given flags found in the dataframe.")

    # Calculate the ratio of binary size to source code size
    filtered_df['Size Ratio'] = filtered_df['Binary Size'] / filtered_df['Original Size']

    # Plot the histogram
    plt.hist(filtered_df['Size Ratio'], bins=10, density=True, alpha=0.5, label='Histogram')

    # Fit a Gaussian distribution to the histogram
    mu, sigma = norm.fit(filtered_df['Size Ratio'])
    x = np.linspace(filtered_df['Size Ratio'].min(), filtered_df['Size Ratio'].max(), 100)
    y = norm.pdf(x, mu, sigma)
    plt.plot(x, y, 'r-', label='Gaussian Fit')

    # Set plot labels and title
    plt.xlabel('Binary Size / Source Code Size')
    plt.ylabel('Density')
    plt.title('Histogram of Binary Size to Source Code Size Ratio')
    plt.legend()

    # Calculate statistics
    mean = np.mean(filtered_df['Size Ratio'])
    std_dev = np.std(filtered_df['Size Ratio'])

    # Identify outliers using a threshold of 3 standard deviations
    threshold = 3 * std_dev
    outliers = filtered_df[filtered_df['Size Ratio'] > mean + threshold]

    # Add a dashed vertical line at the cutoff threshold for outliers
    plt.axvline(x=mean + threshold, color='k', linestyle='--', label='Outlier Threshold')

    # Print statistics and outliers
    print(f"Mean: {mean:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    print("Outliers:")
    print(outliers)

    # Set plot legend
    plt.legend()

    # Save the histogram plot
    filename = '_'.join(flags) + '_histogram.png'
    plt.savefig(filename)

    # Show the plot
    plt.show()

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
