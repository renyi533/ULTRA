task_name=$1
Data_path="/workspace/user_code/violatang/l2-paper/movielens_binary"
Model_path="/ceph/szgpu/10929/violatang/l2r/models/movielens_binary"
Data_folder=""            ## subfolder after unzip
Feature_number=180              ## how many features for LETOR data
Prepro_fun=""                ## additional function to do preprocessing, available, "log", "None", we default normalize data to -1 and 1. If choosing log, it will first using log function to the data and then normalize it to -1 and 1. 
prefix=""                       ## name before data, for example setl.train.txt, prefix=set1.
#Data_zip_file=./Yahoodata/ltrc_yahoo.tar.bz2 ## zipped data file path.
cd ../../
# Download Yahoo! Letor dataset.
# wget https://webscope.sandbox.yahoo.com/download.php?r=39201&d=
# The link might be expired. If so, please go to https://webscope.sandbox.yahoo.com/catalog.php?datatype=c&did=64 for more information.

<<'COMMENT'
#tar -zxvf dataset.tgz
#mkdir Yahoodata/
#mv Learning\ to\ Rank\ Challenge/* Yahoodata/
#mkdir $Data_path
mkdir $Data_path/cleaned_data  # path to store data after cleaning
mkdir $Data_path/normalized    # path to store data after nomalization
mkdir $Data_path/tmp_toy       # path to store toy version of training data which is 1% of total dataset
mkdir $Data_path/tmp_toy/data 
mkdir $Data_path/tmp_toy/tmp
mkdir $Data_path/tmp_toy/tmp_data_toy
#tar  -xjf   $Data_zip_file --directory $Data_path
# Prepare the dataset.
# Sort features, sort query id, remove duplicates, and remove queries without relevant documents in validation and test set.

valid_name=$Data_path/$Data_folder/${prefix}vali.txt
if [ ! -f "$valid_name" ]
then
    echo "no vali, try to find valid"
    valid_name=$Data_path/$Data_folder/${prefix}valid.txt
    if [ ! -f "$valid_name" ]
    then
        echo "no valid, we will split trian with default rate"
        mv $Data_path/$Data_folder/${prefix}train.txt $Data_path/$Data_folder/${prefix}train_orig.txt
        python ./libsvm_tools/split_libsvm_data.py $Data_path/$Data_folder/${prefix}train_orig.txt ${valid_name} $Data_path/$Data_folder/${prefix}train.txt 0.1
    fi
fi
echo "begin cleaning"
python ./libsvm_tools/clean_libsvm_file.py $Data_path/$Data_folder/${prefix}train.txt $Data_path/cleaned_data/train.txt 0
python ./libsvm_tools/clean_libsvm_file.py ${valid_name}  $Data_path/cleaned_data/valid.txt 1
python ./libsvm_tools/clean_libsvm_file.py $Data_path/$Data_folder/${prefix}test.txt $Data_path/cleaned_data/test.txt 1
# Normalize the data
# Extract the feature statistics for later normalization.
echo "extract statistics for normalization"
python ./libsvm_tools/extrac_feature_statistics.py $Data_path/cleaned_data/
# Normalize the data.
echo "begin normalization"
python ./libsvm_tools/normalize_feature.py $Data_path/cleaned_data/feature_scale.json  $Data_path/cleaned_data/test.txt $Data_path/normalized/test.txt $Prepro_fun
python ./libsvm_tools/normalize_feature.py $Data_path/cleaned_data/feature_scale.json  $Data_path/cleaned_data/train.txt $Data_path/normalized/train.txt $Prepro_fun
python ./libsvm_tools/normalize_feature.py $Data_path/cleaned_data/feature_scale.json  $Data_path/cleaned_data/valid.txt $Data_path/normalized/valid.txt $Prepro_fun
# Sample 1% training data to build the initial ranker.
echo "sample 0.01 for intiial ranker"
python ./libsvm_tools/sample_libsvm_data.py $Data_path/normalized/train.txt $Data_path/normalized/sampled_train.txt 0.01

# Download SVMrank.
wget http://download.joachims.org/svm_rank/current/svm_rank_linux64.tar.gz
tar xvzf svm_rank_linux64.tar.gz

# Conduct initial ranking with SVMrank.
python ./libsvm_tools/initial_ranking_with_svm_rank.py \
    ./ \
    $Data_path/normalized/sampled_train.txt \
    $Data_path/normalized/valid.txt \
    $Data_path/normalized/test.txt \
    $Data_path/tmp/
./svm_rank_classify $Data_path/normalized/train.txt $Data_path/tmp/model.dat $Data_path/tmp/train.predict


# Prepare model input.
python ./libsvm_tools/prepare_exp_data_with_svmrank.py $Data_path/normalized/ $Data_path/tmp/ $Data_path/tmp_data/ $Feature_number


cp $Data_path/normalized/sampled_train.txt $Data_path/tmp_toy/data/train.txt
cp $Data_path/normalized/sampled_train.txt $Data_path/tmp_toy/data/valid.txt
cp $Data_path/normalized/sampled_train.txt $Data_path/tmp_toy/data/test.txt
./svm_rank_classify $Data_path/tmp_toy/data/train.txt $Data_path/tmp/model.dat $Data_path/tmp_toy/tmp/train.predict
./svm_rank_classify $Data_path/tmp_toy/data/valid.txt $Data_path/tmp/model.dat $Data_path/tmp_toy/tmp/valid.predict
./svm_rank_classify $Data_path/tmp_toy/data/test.txt $Data_path/tmp/model.dat $Data_path/tmp_toy/tmp/test.predict
python ./libsvm_tools/prepare_exp_data_with_svmrank.py $Data_path/tmp_toy/data/ $Data_path/tmp_toy/tmp/ $Data_path/tmp_toy/tmp_data_toy/ $Feature_number

COMMENT


export SETTING_ARGS="--data_dir=$Data_path/tmp_data/ --model_dir=$Model_path/tmp_model_${task_name}/ --output_dir=$Data_path/tmp_output/ --setting_file=./example/offline_setting_reward/biastower_evallinear_exp_settings.json"
echo $SETTING_ARGS

#rm -rf $Data_path/tmp_model_${task_name}/

# Run model
#python3 main.py --max_train_iteration=8000 $SETTING_ARGS

# Test model
python3 main.py --test_only=True $SETTING_ARGS

