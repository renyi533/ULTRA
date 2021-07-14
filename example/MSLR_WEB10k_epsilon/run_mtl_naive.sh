#!/bin/bash
#
# Copyright (c) 2017 Tencent Inc. All Rights Reserved
#
# File: run_mtl_naive.sh
# Author: root@tencent.com
# Ddate: 2021/06/26 19:06:07
# Brief: 
set -x

sh offline_exp_pipeline_mtl_naive.sh mtl_naive > logs/mtl_naive_clicksimu.log.2000 2>&1
sh offline_exp_pipeline_mtl_naive.sh mtl_naive > logs/mtl_naive_clicksimu.log.4000 2>&1
sh offline_exp_pipeline_mtl_naive.sh mtl_naive > logs/mtl_naive_clicksimu.log.6000 2>&1
sh offline_exp_pipeline_mtl_naive.sh mtl_naive > logs/mtl_naive_clicksimu.log.8000 2>&1
sh offline_exp_pipeline_mtl_naive.sh mtl_naive > logs/mtl_naive_clicksimu.log.10000 2>&1
