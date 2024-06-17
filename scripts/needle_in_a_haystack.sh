config=$1 # set your config
result_dir=result/needle/$config
mkdir result/needle/
mkdir $result_dir
(
python -u benchmark/needle_in_haystack.py --s_len 0 --e_len 128000\
    --config_path config/$config.yaml \
    --output_dir $result_dir
) 2>&1  | tee $result_dir/log.log