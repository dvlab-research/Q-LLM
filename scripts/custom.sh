datasets='custom_book_harrypotter,custom_paper_compare'  # feel free to add your custom datasets

config=$1 # set your config
config_path=config/$config.yaml
output_dir_path=result/custom/$config

# make prediction
CUDA_VISIBLE_DEVICES=0 python benchmark/pred.py \
    --config_path ${config_path} \
    --output_dir_path ${output_dir_path} \
    --datasets ${datasets} 
