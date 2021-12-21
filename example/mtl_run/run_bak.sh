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

while [ $i -lt 10 ]
do

echo "starting round $i"

bash ./offline_exp_pipeline_pairwise_reg_em_trustcorr_1.0_dcg.sh > $log_dir/pairwise_reg_em_dcg_$i.log 2>&1 &
bash ./offline_exp_pipeline_biastower_mtl.sh > $log_dir/biastower_mtl_$i.log 2>&1 &
bash ./offline_exp_pipeline_biastower_mse.sh > $log_dir/biastower_mse_$i.log 2>&1 &
bash ./offline_exp_pipeline_pairwise_reg_em_notrustcorr.sh > $log_dir/pairwise_reg_em_notrustcorr_$i.log 2>&1 &
wait_function
bash ./offline_exp_pipeline_pairwise_reg_em_trustcorr_1.0.sh > $log_dir/pairwise_reg_em_$i.log 2>&1 &
bash ./offline_exp_pipeline_pairwise_reg_em_trustcorr_1.0_dcg_propensity.sh > $log_dir/pairwise_reg_em_dcg_propensity_$i.log 2>&1 &
bash ./offline_exp_pipeline_biastower.sh > $log_dir/biastower_$i.log 2>&1 &
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
