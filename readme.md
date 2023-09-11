# Assignment-1

## Pre Requisites
- python 3 or above
- juptyer notebook 5.7.4 or above
- linux environment

## How to run
- The shell script can be run by using the command `./eval.sh <filename>` where filename is the name of the file to be evaluated.

## Assumptions:
- The input file is a `.npy` file for K nearest neighbours.
- The input file is a `.csv` file for Decision trees.
- There is a file named `eval.py` in the same directory as the shell script.
- The input file is given as argument as mentioned above as the shell script.

## Description
- The shell script utilizes the input file as testing data and subsequently evaluates it.
- Within 1.ipynb, there resides both the code and essential graphs.
- KNN classifier's data is stored in data.npy.
- Data for Decision trees is sourced from advertising.csv.
- The evaluation code is housed in eval.py, a Python script. This specific file is invoked by the shell script.