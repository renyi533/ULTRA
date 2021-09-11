models="mtl_naive mtl_pairwiseem_nodecay_nopointcorr_extractips_evallinear mtl_pairwiseem_nodecay_nopointcorr_extractips_linearndcg_evallinear mtl_naive_relevance mtl_biastower_sum_evallinear mtl_biastower_sum_sigmoid_evallinear"
#models="mtl_pairwiseem_nodecay_nopointcorr_extractips_evallinear mtl_pairwiseem_nodecay_nopointcorr_extractips_linearndcg_evallinear mtl_naive_relevance mtl_biastower_sum_evallinear mtl_biastower_sum_sigmoid_evallinear mtl_biastower_sep_clickdot_evallinear mtl_biastower_sep_clicksum_evallinear mtl_pairwisedebias_evallinear mtl_dla_evallinear"
for model in ${models}
do
    sh repeat_run.sh ${model} > logs/repeat_${model}.log 2>&1
done
