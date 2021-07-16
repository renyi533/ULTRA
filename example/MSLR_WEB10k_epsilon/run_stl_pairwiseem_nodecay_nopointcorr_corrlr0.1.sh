#!/bin/bash
#
# Copyright (c) 2017 Tencent Inc. All Rights Reserved
#
# File: run_stl_pairwiseem_nodecay_nopointcorr_corrlr0.1.sh
# Author: root@tencent.com
# Ddate: 2021/06/26 19:06:07
# Brief: 
set -x

sh offline_exp_pipeline_stl_pairwiseem_nodecay_nopointcorr_corrlr0.1.sh stl_pairwiseem_nodecay_nopointcorr_corrlr0.1 > logs/stl_pairwiseem_nodecay_nopointcorr_corrlr0.1_clicksimu.log.2000 2>&1
exit 0
sh offline_exp_pipeline_stl_pairwiseem_nodecay_nopointcorr_corrlr0.1.sh stl_pairwiseem_nodecay_nopointcorr_corrlr0.1 > logs/stl_pairwiseem_nodecay_nopointcorr_corrlr0.1_clicksimu.log.4000 2>&1
sh offline_exp_pipeline_stl_pairwiseem_nodecay_nopointcorr_corrlr0.1.sh stl_pairwiseem_nodecay_nopointcorr_corrlr0.1 > logs/stl_pairwiseem_nodecay_nopointcorr_corrlr0.1_clicksimu.log.6000 2>&1
sh offline_exp_pipeline_stl_pairwiseem_nodecay_nopointcorr_corrlr0.1.sh stl_pairwiseem_nodecay_nopointcorr_corrlr0.1 > logs/stl_pairwiseem_nodecay_nopointcorr_corrlr0.1_clicksimu.log.8000 2>&1
sh offline_exp_pipeline_stl_pairwiseem_nodecay_nopointcorr_corrlr0.1.sh stl_pairwiseem_nodecay_nopointcorr_corrlr0.1 > logs/stl_pairwiseem_nodecay_nopointcorr_corrlr0.1_clicksimu.log.10000 2>&1
