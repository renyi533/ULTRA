export Data_path="./MovieLens"   ## Data path where to unzip the data
export Data_folder=""            ## subfolder after unzip
export Feature_number=180 ## how many features for LETOR data
export Data_zip_file=./Yahoodata/ltrc_yahoo.tgz ## zipped data file path.
export log_dir="./movie_lens_logs"
[ -d "$log_dir" ] || mkdir $log_dir

bash ./run.sh
