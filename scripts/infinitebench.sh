datasets='longbook_choice_eng,passkey,number_string,code_debug,math_find,kv_retrieval' # infinite bench
world_size=8

config=$1 # set your config
config_path=config/$config.yaml
output_dir_path=result/infinitebench/$config

# make prediction
bash scripts/multiprocessing-benchmark.sh \
    --world_size $world_size \
    --config_path $config_path \
    --output_dir_path ${output_dir_path} \
    --datasets $datasets

# merge multi-process results
python benchmark/merge.py \
    --output_dir_path ${output_dir_path} \
    --datasets ${datasets} \
    --world_size ${world_size}

# evaluation
python benchmark/eval.py \
    --dir_path ${output_dir_path}