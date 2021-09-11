models="mtl_naive_pairceloss mtl_naive_relevance_pairloss"
#models="mtl_biastower_sep_clickdot_evallinear mtl_biastower_sep_clicksum_evallinear mtl_pairwisedebias_evallinear mtl_dla_evallinear"
#models="mtl_biastower_sep_clickdot_evallinear"
#models="mtl_pairwiseem_nodecay_nopointcorr_extractips_evallinear mtl_pairwiseem_nodecay_nopointcorr_extractips_linearndcg_evallinear mtl_naive_relevance mtl_biastower_sum_evallinear mtl_biastower_sum_sigmoid_evallinear mtl_biastower_sep_clickdot_evallinear mtl_biastower_sep_clicksum_evallinear mtl_pairwisedebias_evallinear mtl_dla_evallinear"
set -x
for model in ${models}
do
    echo $model
    sh repeat_run.sh ${model} > logs/repeat_${model}.log 2>&1
done
