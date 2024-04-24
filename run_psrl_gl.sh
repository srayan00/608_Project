#!/bin/sh

script_name=$1
job_name=$2
seed=$3
sampler=$4
states=$5
actions=$6
components=$7
episodes=$8
horizon=$9

sbatch --account=eecs602w24_class --cpus-per-task=8 --nodes=1 --mem-per-cpu=11500m --time=00-24:00:00 --job-name=$job_name --output="gllogs/$job_name.out" --error="gllogs/$job_name.err" $script_name $seed