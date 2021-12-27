export Data_path="./MSLR_30k_letor"   ## Data path where to unzip the data
export Data_folder="Fold1"            ## subfolder after unzip
export Feature_number=136              ## how many features for LETOR data
export Data_zip_file=./MSLR-WEB30K.zip ## zipped data file path.
export log_dir="./30k_theta_logs"
[ -d "$log_dir" ] || mkdir $log_dir

bash ./theta_test/run_theta.sh