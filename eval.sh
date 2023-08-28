#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_file.npy>"
    exit 1
fi

# input file is given as first argument
input_file=$1

if [[ $input_file == *.npy ]]; then
    echo "Given a valid file"
else
    echo "Input file is not an .npy file, please give a valid .npy file"
    exit 1
fi

echo "Evaluating $input_file"

#check for input file
if [ ! -f $input_file ]; then
    echo "Input file not found!"
    exit 1
fi

python3 eval.py $input_file