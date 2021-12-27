if [ -z "$Data_path" ]
then
      echo "\$Data_path is empty. set to ./MSLR_10k_letor"
      Data_path="./MSLR_10k_letor"   ## Data path where to unzip the data
else
      echo "\$Data_path already set to $Data_path"
fi

if [ -z "$Data_folder" ]
then
      echo "\$Data_folder is empty. set to Fold1"
      Data_folder="Fold1"            ## subfolder after unzip
else
      echo "\$Data_folder already set to $Data_folder"
fi

if [ -z "$Data_zip_file" ]
then
      echo "\$Data_zip_file is empty. set to ./MSLR-WEB10K.zip"
      Data_zip_file=./MSLR-WEB10K.zip ## zipped data file path.
else
      echo "\$Data_zip_file already set to $Data_zip_file"
fi

if [ -z "$Feature_number" ]
then
      echo "\$Feature_number is empty. set to 136"
      Feature_number=136              ## how many features for LETOR data
else
      echo "\$Feature_number already set to $Feature_number"
fi

Prepro_fun=""                ## additional function to do preprocessing, available, "log", "None", we default normalize data to -1 and 1. If choosing log, it will first using log function to the data and then normalize it to -1 and 1. 
prefix=""                    ## name before data, for example setl.train.txt, prefix=set1.
cd ../../
# Download MSLR-WEB10K dataset.
# view https://www.microsoft.com/en-us/research/project/mslr/ for the download link

algo_name=naive_pairwise_1.5
rm -rf $Data_path/tmp_model_$algo_name/*

export SETTING_ARGS="--data_dir=$Data_path/tmp_data/ --model_dir=$Data_path/tmp_model_$algo_name/ --output_dir=$Data_path/tmp_output_$algo_name/ --setting_file=./example/offline_setting_theta/naive_algorithm_pairwise_noisy_exp_settings_theta1.5.json"
echo $SETTING_ARGS
# Run model
python main.py --max_train_iteration=5000 $SETTING_ARGS

# Test model
python main.py --test_only=True $SETTING_ARGS