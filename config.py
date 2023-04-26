# Max Van Gelder
# 26/4/23

# Primarily just a way to track dependencies for programs like
# CSmith and CReduce which don't necessarily have a standard path

import pathlib

# Path of CReduce Executable (i.e. result of 'which creduce' in Unix)
CREDUCE_PATH = pathlib.Path('/usr/bin/creduce') # Tested only with creduce 2.11.0 (unknown)
# Path of CSmith Executable (i.e. result of 'which csmith' in Unix)
CSMITH_PATH = pathlib.Path('/home/max/csmith/bin/csmith') # Tested only with csmith 2.4.0 [Git version: 92069e4]
# Path of compiler
COMPILER_PATH = pathlib.Path('/usr/bin/g++') # Tested only with g++ (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0


