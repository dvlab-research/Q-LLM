import torch
from qllm.models import LlamaForCausalLM
from transformers import AutoTokenizer
import transformers

from omegaconf import OmegaConf
from qllm.utils import patch_hf, GreedySearch, patch_model_center

conf = OmegaConf.load("config/llama3-qllm-repr4-l1k-bs128-topk8-w4.yaml")
model_path = "models/Meta-Llama-3-8B-Instruct"

model = LlamaForCausalLM.from_pretrained(
	model_path,
	torch_dtype=torch.bfloat16,
	trust_remote_code=True
	).to("cuda:0")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, add_bos_token=True, add_eos_token=False)

model = patch_hf(model, "qllm", conf.model)
model = GreedySearch(model, tokenizer)

text = "xxx"

encoded_text = tokenizer.encode(text)
input_ids = torch.tensor(encoded_text).unsqueeze(0).to("cuda:0")

output = model.generate(input_ids, max_length=200)
print(output)
model.clear()