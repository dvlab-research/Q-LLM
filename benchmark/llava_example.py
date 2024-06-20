from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from qllm.utils import patch_hf, GreedySearch, patch_model_center
from omegaconf import OmegaConf
import time 

from PIL import Image
import requests
import copy
import torch
from omegaconf import OmegaConf
import argparse
import json 
import os 

parser = argparse.ArgumentParser(description="Q-LLM for LLaVA-Next-LLaMA3 Infer Example")
parser.add_argument('--question', type=str, 
                    default='Which model performs the best in the image?')
parser.add_argument('--image_path', type=str, 
                    default="https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true")
parser.add_argument('--output_path', type=str, 
                    default="results/demo/llava_v1_5_radar/")
args = parser.parse_args()
    
pretrained = "../models/llama3-llava-next-8b" # your llava path
model_name = "llava_llama3"
config_path = 'config/llama3-qllm-repr4-l1k-bs128-topk8-w4.yaml'

image = Image.open(
    requests.get(args.image_path, stream=True).raw if args.image_path.startswith('http') else args.image_path)

os.makedirs(args.output_path, exist_ok=True)
output_path = os.path.join(args.output_path, args.question.replace(' ', '_')+'.jsonl') 

tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map="cuda") # Add any other thing you want to pass in llava_model_args
model.eval()
model.tie_weights()

image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device="cuda") for _image in image_tensor]

conv_template = "llava_llama_3" # Make sure you use correct chat template for different models
question = args.question + "\n" + DEFAULT_IMAGE_TOKEN
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
image_sizes = [image.size]

# add patch_hf
config = OmegaConf.load(config_path) # yamls in config
model = patch_hf(model, config.model.type, **config.model)
model = GreedySearch(model, tokenizer)

start_time = time.time()

cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_length=512,
    max_new_tokens=256,
)

text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
t = time.time() - start_time
if hasattr(model, 'clear'):
    model.clear()
    
data_to_save = {
    "config_path": config_path,
    "question": question,
    "image_path": args.image_path,
    "text_outputs": text_outputs,
    "t": t
}
print(data_to_save)

with open(output_path, 'a') as jsonl_file:
    jsonl_file.write(json.dumps(data_to_save) + '\n')