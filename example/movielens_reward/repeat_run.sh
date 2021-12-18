#!/bin/bash
#
# Copyright (c) 2017 Tencent Inc. All Rights Reserved
#
# File: run_mtl_pairwiseem_nodecay_nopointcorr_corrlr0.1.sh
# Author: root@tencent.com
# Ddate: 2021/06/26 19:06:07
# Brief: 

task=$1
if [ $# -gt 1 ]; then
    log_dir=$2
else
    log_dir="logs"
fi
n=1
while [ $n -le 10 ]
do
    echo $n
    sh offline_exp_pipeline_${task}.sh ${task}_${n} > ${log_dir}/${task}_clicksimu${n}.log 2>&1 &
    if [ $n == 5 ]; then
	wait
    fi
    let n++
done
wait
python calc_avg_metric.py ${task} 10 ${log_dir}


