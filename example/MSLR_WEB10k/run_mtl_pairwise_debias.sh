#!/bin/bash
#
# Copyright (c) 2017 Tencent Inc. All Rights Reserved
#
# File: run_mtl_pairwiseem.sh
# Author: root@tencent.com
# Ddate: 2021/06/26 19:06:07
# Brief: 

sh offline_exp_pipeline_mtl_pairwise_debias.sh mtl_pairwise_debias > logs/mtl_pairwise_debias_clicksimu.log.2000 2>&1
sh offline_exp_pipeline_mtl_pairwise_debias.sh mtl_pairwise_debias > logs/mtl_pairwise_debias_clicksimu.log.4000 2>&1
sh offline_exp_pipeline_mtl_pairwise_debias.sh mtl_pairwise_debias > logs/mtl_pairwise_debias_clicksimu.log.6000 2>&1
sh offline_exp_pipeline_mtl_pairwise_debias.sh mtl_pairwise_debias > logs/mtl_pairwise_debias_clicksimu.log.8000 2>&1
sh offline_exp_pipeline_mtl_pairwise_debias.sh mtl_pairwise_debias > logs/mtl_pairwise_debias_clicksimu.log.10000 2>&1
