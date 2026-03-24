#!/bin/bash
set -e  # exit on any error

echo "Lets go..."
echo "Running on $(hostname) with PID $$"
echo "Task $1"
echo "Path to train/val data $3"

if [ $# -eq 3 ]; then
    if [ "$1" = "train" ]; then
        echo "Training model on $2"
        python -m src.train --model-to-run $2 --data-path $3

    elif [ "$1" = "eval" ]; then
        echo "Evaluating model on $2"
        python -m src.eval --model-to-run $2 --data-path $3

    else
        echo "Error :: 1st argument should strictly be train/eval"
        exit 1
    fi
else
    echo "Wrong Format :: Please run bash run_model.sh train/eval model-to-run path-to-train-data"
    exit 1
fi