#!/bin/bash

datasets='rbv2_aknn rbv2_ranger rbv2_rpart rbv2_super rbv2_svm rbv2_xgboost lcbench'

for dataset in $datasets
do
        for lim in 2
        do
                for s in 0 1 2 3 4
                do
                  python main.py +experiment=$dataset seed=$s training.test_lim=$lim
                done
        done
done
