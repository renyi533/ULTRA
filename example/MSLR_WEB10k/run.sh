mkdir logs

FAIL=0

echo "starting"

bash ./offline_exp_pipeline_pairwise_reg_em_noips.sh > logs/pair_reg_em_noips.log 2>&1 &
bash ./offline_exp_pipeline_pairwise_reg_em_trustcorr_1.0_noips.sh > logs/pair_reg_em_trustcorr1.0_noips.log 2>&1 &

sleep 10
for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

bash ./offline_exp_pipeline_pairwise_reg_em_trustcorr_0.1_noips.sh > logs/pair_reg_em_trustcorr0.1_noips.log 2>&1 &
bash ./offline_exp_pipeline_pairwise_reg_em_trustcorr_0.06_noips.sh > logs/pair_reg_em_trustcorr0.06_noips.log 2>&1 &

sleep 10
for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

bash ./offline_exp_pipeline_dla.sh > logs/dla.log 2>&1 &
bash ./offline_exp_pipeline_naive.sh > logs/naive.log 2>&1 &
sleep 10
for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

bash ./offline_exp_pipeline_pairwise_reg_em.sh > logs/pair_reg_em.log 2>&1 &
bash ./offline_exp_pipeline_pairwise_reg_em_trustcorr_1.0.sh > logs/pair_reg_em_trustcorr1.0.log 2>&1 &

sleep 10
for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

bash ./offline_exp_pipeline_pairwise_reg_em_trustcorr_0.1.sh > logs/pair_reg_em_trustcorr0.1.log 2>&1 &
bash ./offline_exp_pipeline_pairwise_reg_em_trustcorr_0.06.sh > logs/pair_reg_em_trustcorr0.06.log 2>&1 &

sleep 10
for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

bash ./offline_exp_pipeline_pairwise_reg_em_trustcorr_0.03.sh > logs/pair_reg_em_trustcorr0.03.log 2>&1 &

sleep 10
for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

echo $FAIL

if [ "$FAIL" == "0" ];
then
echo "YAY!"
else
echo "FAIL! ($FAIL)"
fi