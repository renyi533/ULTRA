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

bash offline_exp_pipeline_dla.sh> $log_dir/dla_theta1.0_$i.log 2>&1 &
bash offline_exp_pipeline_regression_em.sh > $log_dir/reg_em_theta1.0_$i.log 2>&1 &
bash offline_exp_pipeline_naive_pairwise.sh> $log_dir/naive_pairwise_theta1.0_$i.log 2>&1 &
wait_function
bash offline_exp_pipeline_pairwise_reg_em_trustcorr_1.0_ndcg.sh> $log_dir/pairwise_reg_em_trustcorr_1.0_ndcg_theta1.0_$i.log 2>&1 &
bash offline_exp_pipeline_pairwise_reg_em_trustcorr_1.0.sh> $log_dir/pairwise_reg_em_trustcorr_1.0_theta1.0_$i.log 2>&1 &
wait_function

bash ./theta_test/offline_exp_pipeline_dla_theta0.5.sh > $log_dir/dla_theta0.5_$i.log 2>&1 &
bash ./theta_test/offline_exp_pipeline_reg_em_theta0.5.sh > $log_dir/reg_em_theta0.5_$i.log 2>&1 &
bash ./theta_test/offline_exp_pipeline_naive_pairwise_theta0.5.sh > $log_dir/naive_pairwise_theta0.5_$i.log 2>&1 &
wait_function
bash ./theta_test/offline_exp_pipeline_pairwise_reg_em_trustcorr_1.0_ndcg_theta0.5.sh > $log_dir/pairwise_reg_em_trustcorr_1.0_ndcg_theta0.5_$i.log 2>&1 &
bash ./theta_test/offline_exp_pipeline_pairwise_reg_em_trustcorr_1.0_theta0.5.sh > $log_dir/pairwise_reg_em_trustcorr_1.0_theta0.5_$i.log 2>&1 &
wait_function

bash ./theta_test/offline_exp_pipeline_dla_theta1.5.sh > $log_dir/dla_theta1.5_$i.log 2>&1 &
bash ./theta_test/offline_exp_pipeline_reg_em_theta1.5.sh > $log_dir/reg_em_theta1.5_$i.log 2>&1 &
bash ./theta_test/offline_exp_pipeline_naive_pairwise_theta1.5.sh > $log_dir/naive_pairwise_theta1.5_$i.log 2>&1 &
wait_function
bash ./theta_test/offline_exp_pipeline_pairwise_reg_em_trustcorr_1.0_ndcg_theta1.5.sh > $log_dir/pairwise_reg_em_trustcorr_1.0_ndcg_theta1.5_$i.log 2>&1 &
bash ./theta_test/offline_exp_pipeline_pairwise_reg_em_trustcorr_1.0_theta1.5.sh > $log_dir/pairwise_reg_em_trustcorr_1.0_theta1.5_$i.log 2>&1 &
wait_function

bash ./theta_test/offline_exp_pipeline_dla_theta0.3.sh > $log_dir/dla_theta0.3_$i.log 2>&1 &
bash ./theta_test/offline_exp_pipeline_reg_em_theta0.3.sh > $log_dir/reg_em_theta0.3_$i.log 2>&1 &
bash ./theta_test/offline_exp_pipeline_naive_pairwise_theta0.3.sh > $log_dir/naive_pairwise_theta0.3_$i.log 2>&1 &
wait_function
bash ./theta_test/offline_exp_pipeline_pairwise_reg_em_trustcorr_1.0_ndcg_theta0.3.sh > $log_dir/pairwise_reg_em_trustcorr_1.0_ndcg_theta0.3_$i.log 2>&1 &
bash ./theta_test/offline_exp_pipeline_pairwise_reg_em_trustcorr_1.0_theta0.3.sh > $log_dir/pairwise_reg_em_trustcorr_1.0_theta0.3_$i.log 2>&1 &
wait_function

bash ./theta_test/offline_exp_pipeline_dla_theta2.0.sh > $log_dir/dla_theta2.0_$i.log 2>&1 &
bash ./theta_test/offline_exp_pipeline_reg_em_theta2.0.sh > $log_dir/reg_em_theta2.0_$i.log 2>&1 &
bash ./theta_test/offline_exp_pipeline_naive_pairwise_theta2.0.sh > $log_dir/naive_pairwise_theta2.0_$i.log 2>&1 &
wait_function
bash ./theta_test/offline_exp_pipeline_pairwise_reg_em_trustcorr_1.0_ndcg_theta2.0.sh > $log_dir/pairwise_reg_em_trustcorr_1.0_ndcg_theta2.0_$i.log 2>&1 &
bash ./theta_test/offline_exp_pipeline_pairwise_reg_em_trustcorr_1.0_theta2.0.sh > $log_dir/pairwise_reg_em_trustcorr_1.0_theta2.0_$i.log 2>&1 &
wait_function

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
