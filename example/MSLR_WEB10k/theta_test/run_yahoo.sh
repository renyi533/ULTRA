export Data_path="./Yahoo_letor"   ## Data path where to unzip the data
export Data_folder=""            ## subfolder after unzip
export Feature_number=700 ## how many features for LETOR data
export Data_zip_file=./Yahoodata/ltrc_yahoo.tgz ## zipped data file path.
export log_dir="./yahoo_theta_logs"
[ -d "$log_dir" ] || mkdir $log_dir

bash ./theta_test/run_theta.sh
