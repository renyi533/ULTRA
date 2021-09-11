#!/bin/bash
#
# Copyright (c) 2017 Tencent Inc. All Rights Reserved
#
# File: run_mtl_pairwiseem_nodecay_nopointcorr_corrlr0.1.sh
# Author: root@tencent.com
# Ddate: 2021/06/26 19:06:07
# Brief: 

task=$1
n=1
while [ $n -le 10 ]
do
    echo $n
    #sh offline_exp_pipeline_mtl_pairwiseem_nodecay_nopointcorr_corrlr0.01.sh mtl_pairwiseem_nodecay_nopointcorr_corrlr0.01 > logs/mtl_pairwiseem_nodecay_nopointcorr_corrlr0.01_clicksimu${n}.log 2>&1
    sh offline_exp_pipeline_${task}.sh ${task}_${n} > logs/${task}_clicksimu${n}.log 2>&1 &
    if [ $(($n%2)) == 0  ]; then
	wait
    fi
    let n++
done
wait
python calc_avg_metric.py ${task} 10


