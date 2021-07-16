Data_path="./MSLR_10k_letor"   ## Data path where to unzip the data
Data_folder="Fold1"            ## subfolder after unzip
Feature_number=136              ## how many features for LETOR data
Prepro_fun=""                ## additional function to do preprocessing, available, "log", "None", we default normalize data to -1 and 1. If choosing log, it will first using log function to the data and then normalize it to -1 and 1. 
prefix=""                       ## name before data, for example setl.train.txt, prefix=set1.
Data_zip_file=./MSLR-WEB10K.zip ## zipped data file path.
cd ../../
# Download MSLR-WEB10K dataset.
# view https://www.microsoft.com/en-us/research/project/mslr/ for the download link

algo_name=pairwise_reg_em_trustcorr_0.06
rm -rf $Data_path/tmp_model_$algo_name/*

export SETTING_ARGS="--data_dir=$Data_path/tmp_data/ --model_dir=$Data_path/tmp_model_$algo_name/ --output_dir=$Data_path/tmp_output_$algo_name/ --setting_file=./example/offline_setting/pairwise_regression_EM_noisy_trustcorr_0.06_exp_settings_noips.json"
echo $SETTING_ARGS
# Run model
python main.py --max_train_iteration=5000 $SETTING_ARGS

# Test model
python main.py --test_only=True $SETTING_ARGS