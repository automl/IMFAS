#!/bin/bash

cd ..

for MODEL in 'lcnet' # 'parametric_best_lc ' # 'successivehalving' # 'random' 'successivehalving'
#for MODEL in 'imfas_iclr'  'imfas_sh_scheduler'
do
    for DATASET in   'lcbench' # 'synthetic_func' 'task_set' # 'openml'
    do
        for FOLDIDX in {0..5}
        do
            for SEED in {0..5}
            do
              python main.py +experiment=$MODEL dataset=$DATASET  wandb.mode=online train_test_split.fold_idx=$FOLDIDX seed=$SEED wandb.group='lcnet_baseline'
            done
        done
    done
done
