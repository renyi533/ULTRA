FAIL=0

echo "starting"

i="0"

wait_function () {
    sleep 10
    for job in `jobs -p`
    do
        echo $job
        wait $job || let "FAIL+=1"
    done
}

while [ $i -lt 22 ]
do

echo "starting round $i"

bash ./offline_exp_pipeline_naive_directlabel_pairwise.sh > $log_dir/directlabel_pairwise_$i.log 2>&1 &
bash ./offline_exp_pipeline_naive_directlabel_lambdarank.sh > $log_dir/directlabel_lambdarank_$i.log 2>&1 &
bash ./offline_exp_pipeline_naive.sh > $log_dir/naive_$i.log 2>&1 &
bash ./offline_exp_pipeline_naive_directlabel.sh > $log_dir/directlabel_$i.log 2>&1 &
wait_function

bash ./offline_exp_pipeline_regression_em_pairwise.sh > $log_dir/regression_em_pairwise_$i.log 2>&1 &
bash ./offline_exp_pipeline_naive_pairwise.sh > $log_dir/naive_pairwise_$i.log 2>&1 &
bash ./offline_exp_pipeline_pairwise_debias.sh > $log_dir/pairwise_debias_$i.log 2>&1 &
bash ./offline_exp_pipeline_dla.sh > $log_dir/dla_$i.log 2>&1 &
wait_function

bash ./offline_exp_pipeline_pairwise_reg_em.sh > $log_dir/pair_reg_em_$i.log 2>&1 &
bash ./offline_exp_pipeline_pairwise_reg_em_trustcorr_1.0_ndcg.sh > $log_dir/pair_reg_em_trustcorr1.0_ndcg_$i.log 2>&1 &
bash ./offline_exp_pipeline_pairwise_reg_em_trustcorr_1.0.sh > $log_dir/pair_reg_em_trustcorr1.0_$i.log 2>&1 &
wait_function

bash ./offline_exp_pipeline_pairwise_reg_em_pointips.sh > $log_dir/pair_reg_em_pointips_$i.log 2>&1 &
bash ./offline_exp_pipeline_pairwise_reg_em_trustcorr_1.0_pointips.sh > $log_dir/pair_reg_em_trustcorr1.0_pointips_$i.log 2>&1 &
bash ./offline_exp_pipeline_pairwise_reg_em_trustcorr_1.0_simips_pointips.sh > $log_dir/pair_reg_em_trustcorr1.0_simips_pointips_$i.log 2>&1 &
wait_function

comment='bash ./offline_exp_pipeline_pairwise_reg_em_trustcorr_0.1_noips.sh > $log_dir/pair_reg_em_trustcorr0.1_noips_$i.log 2>&1 &
bash ./offline_exp_pipeline_pairwise_reg_em_trustcorr_0.1_ndcg.sh > $log_dir/pair_reg_em_trustcorr0.1_ndcg_$i.log 2>&1 &
wait_function'

echo "total failure: $FAIL"

i=$[$i+1]
done

if [ "$FAIL" == "0" ];
then
echo "YAY!"
else
echo "FAIL! ($FAIL)"
fi

echo "results:"
find $log_dir -type f -name "*log" -exec awk '{s=$0};END{print FILENAME,s}' {} \; | sort
