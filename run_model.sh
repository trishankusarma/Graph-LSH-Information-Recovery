#!/bin/bash
set -e

echo "Lets go..."
echo "Running on $(hostname) with PID $$"

if [ $# -lt 3 ]; then
    echo "Wrong Format :: bash run_model.sh train/eval dataset1 [dataset2 ...] path"
    exit 1
fi

task=$1
path=${@: -1}                      # last argument = path
datasets=("${@:2:$#-2}")          # everything between = datasets

echo "Task: $task"
echo "Datasets: ${datasets[@]}"
echo "Path: $path"

valid_datasets=("Cora" "CiteSeer" "PubMed" "ogbn-arxiv")

# validate all datasets
for ds in "${datasets[@]}"; do
    if [[ ! " ${valid_datasets[@]} " =~ " $ds " ]]; then
        echo "$ds :: Choose from: Cora, CiteSeer, PubMed, ogbn-arxiv"
        exit 1
    fi
done

# run for each dataset
for ds in "${datasets[@]}"; do
    if [ "$task" = "train" ]; then
        echo "Training model on $ds"
        python -m src.train --model-to-run "$ds" --data-path "$path"

    elif [ "$task" = "eval" ]; then
        echo "Evaluating model on $ds"
        python -m src.eval --model-to-run "$ds" --data-path "$path"

    else
        echo "Error :: 1st argument should strictly be train/eval"
        exit 1
    fi
done