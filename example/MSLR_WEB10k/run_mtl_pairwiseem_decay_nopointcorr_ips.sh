#!/bin/bash
#
# Copyright (c) 2017 Tencent Inc. All Rights Reserved
#
# File: run_mtl_pairwiseem_decay_nopointcorr_ips.sh
# Author: root@tencent.com
# Ddate: 2021/06/26 19:06:07
# Brief: 
set -x

sh offline_exp_pipeline_mtl_pairwiseem_decay_nopointcorr_ips.sh mtl_pairwiseem_decay_nopointcorr_ips > logs/mtl_pairwiseem_decay_nopointcorr_ips_clicksimu.log.2000 2>&1
sh offline_exp_pipeline_mtl_pairwiseem_decay_nopointcorr_ips.sh mtl_pairwiseem_decay_nopointcorr_ips > logs/mtl_pairwiseem_decay_nopointcorr_ips_clicksimu.log.4000 2>&1
sh offline_exp_pipeline_mtl_pairwiseem_decay_nopointcorr_ips.sh mtl_pairwiseem_decay_nopointcorr_ips > logs/mtl_pairwiseem_decay_nopointcorr_ips_clicksimu.log.6000 2>&1
sh offline_exp_pipeline_mtl_pairwiseem_decay_nopointcorr_ips.sh mtl_pairwiseem_decay_nopointcorr_ips > logs/mtl_pairwiseem_decay_nopointcorr_ips_clicksimu.log.8000 2>&1
sh offline_exp_pipeline_mtl_pairwiseem_decay_nopointcorr_ips.sh mtl_pairwiseem_decay_nopointcorr_ips > logs/mtl_pairwiseem_decay_nopointcorr_ips_clicksimu.log.10000 2>&1
