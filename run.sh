#!/bin/bash

experiment_numbers=(0 1 2 3)
dataset=TerraIncognita

# Model selection through the proposed method
for number in "${experiment_numbers[@]}"
do
  output_dir="experiments_terra$number"
  exp_dir="exp_results/$dataset/$number"

  CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.diwa\
       --data_dir=DomainBed/\
       --output_dir=$output_dir/\
       --exp_dir=$exp_dir/\
       --dataset $dataset\
       --test_env $number\
       --weight_selection TEP\
       --trial_seed 0
done
