export Data_path="./MSLR_10k_letor"   ## Data path where to unzip the data
export Data_folder="Fold1"            ## subfolder after unzip
export Feature_number=136              ## how many features for LETOR data
export Data_zip_file=./MSLR-WEB10K.zip ## zipped data file path.
export log_dir="./10klogs"
[ -d "$log_dir" ] || mkdir $log_dir

bash ./run.sh