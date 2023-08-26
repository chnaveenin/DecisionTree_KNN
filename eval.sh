#!/bin/bash

# Check if the required command-line arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 test_embeddings.npy test_labels.npy"
    exit 1
fi

# Assign the command-line arguments to variables
test_embeddings_file="$1"
test_labels_file="$2"

# Check if the provided files exist
if [ ! -f "$test_embeddings_file" ] || [ ! -f "$test_labels_file" ]; then
    echo "One or both input files not found."
    exit 1
fi

# Run Python script to calculate metrics
metrics=$(python3 calculate_metrics.py "$test_embeddings_file" "$test_labels_file")

# Print the metrics
echo "Metrics:"
echo "$metrics"
