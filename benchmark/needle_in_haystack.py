"""
This script is adapted from 
https://github.com/gkamradt/LLMTest_NeedleInAHaystack

# require you to download the model first
# Mistral Q-LLM 512
config=mistral-qllm-repr4-l256-bs64-topk4-w1
mkdir result/$config
(
python -u benchmark/needle_in_haystack.py --s_len 0 --e_len 128000\
    --config_path config/$config.yaml \
    --output_dir result/$config
) 2>&1  | tee result/$config/log.log

# Mistral InfLLM 512
config=mistral-inf-llm-repr4-l256-bs64-topk4
mkdir result/$config
(
python -u benchmark/needle_in_haystack.py --s_len 0 --e_len 128000\
    --config_path config/$config.yaml \
    --output_dir result/$config
) 2>&1  | tee result/$config/log.log
"""
from tqdm import tqdm 
import tiktoken
import os 
import glob
import json
import tensor_parallel as tp
from transformers import AutoModelForCausalLM, AutoTokenizer
from anthropic import Anthropic
#from dotenv import load_dotenv
import numpy as np
import argparse
from rouge_score import rouge_scorer
import tensor_parallel as tp
import random
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

from openai import OpenAI
from datetime import datetime, timezone
import time
import torch

from omegaconf import OmegaConf
from qllm.utils import patch_hf, GreedySearch, patch_model_center
from qllm.utils.extract_question import extract_question_id


def reset_rope(model, model_max_train_len, scaling_factor):
    for l in model.model.layers:
        l.self_attn.rotary_emb.scaling_factor = scaling_factor
        l.self_attn.rotary_emb._set_cos_sin_cache(seq_len=model_max_train_len, device="cpu", dtype=torch.float32)
    return

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
                 haystack_dir="benchmark/data/PaulGrahamEssays",
                 retrieval_question="What is the best thing to do in San Francisco?",
                 results_version = 1,
                 context_lengths_min = 1000,
                 context_lengths_max = 128000,
                 context_lengths_num_intervals = 10,
                 context_lengths = None,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 10,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 openai_api_key=None,
                 anthropic_api_key = None,
                 model_name='',
                 model_name_suffix=None,
                 num_concurrent_requests = 1,
                 save_results = True,
                 save_contexts = True,
                 final_context_length_buffer = 200,
                 seconds_to_sleep_between_completions = None,
                 print_ongoing_status = True,
                 config=None,
                 output_dir=None,
                 reverse_order=False):
        """        
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_provider: The provider of the model. Must be either 'OpenAI' or 'Anthropic'. Default is 'OpenAI'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param anthropic_api_key: The API key for Anthropic. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = config.conv_type
        self.testing_results = []
        self.config = config.model
        self.output_dir = output_dir
        self.chunk_size = config.chunk_size
        self.conv_type = config.conv_type
        model_name = config.model.path

        if("/" in model_name):
            self.model_version = model_name.split("/")[-1]
        else: self.model_version = model_name
        if(model_name_suffix is not None): self.model_version += "_" + model_name_suffix

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
                if reverse_order:
                    self.context_lengths = self.context_lengths[::-1]
        else:
            self.context_lengths = context_lengths

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        
        self.model_name = model_name

        if(self.model_provider not in ["OpenAI", "Anthropic"]):
            self.enc = AutoTokenizer.from_pretrained(model_name)
            print("loading from %s" % model_name)
            if self.conv_type == 'mistral-inst':
                from qllm.models.modeling_mistral import MistralForCausalLM
                self.model_to_test = MistralForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda").eval()
            elif self.conv_type == 'llama-3-inst':
                from qllm.models.modeling_llama import LlamaForCausalLM
                self.model_to_test = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda").eval()
            else:
                self.model_to_test = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda").eval()


            if self.config.type in ['qllm', 'inf-llm', 'stream-llm', 'infinite-lm']:
                # self.model_to_test = tp.tensor_parallel(self.model_to_test, sharded=True)
                self.model_to_test = patch_hf(self.model_to_test, self.config.type, **self.config).cuda()
                self.model_to_test = GreedySearch(self.model_to_test, self.enc)
                self.model_to_test.device = 'cuda'
            else:
                scaling_factor = 10 # hardcode
                reset_rope(self.model_to_test, model_max_train_len=81920, scaling_factor=scaling_factor)
                self.model_to_test = tp.tensor_parallel(self.model_to_test, sharded=True)
        else: 
            self.model_to_test = OpenAI(api_key=openai_api_key)
            if(self.model_provider == "OpenAI"):
                self.enc = tiktoken.encoding_for_model(self.model_name)
            elif(self.model_provider == "Anthropic"):
                self.enc = Anthropic().get_tokenizer()

        self.model_to_test_description = model_name
        
        self.evaluation_model = None
        self.debug='debug'
        model_name = model_name.split('/')[-1]

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)
    
    def bound_evaluate_and_log(self, *args):
        self.evaluate_and_log(*args)

    def run_test(self, args):
        # Run through each iteration of context_lengths and depths
        tasks = []
        total_iters = len(self.context_lengths) * len(self.document_depth_percents)
        progress_bar = tqdm(total=total_iters, desc="Processing tasks")
        
        for context_length in self.context_lengths:
            if context_length < args.s_len or context_length > args.e_len: continue
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(context_length, depth_percent)
                progress_bar.update()  # Manually update the progress bar after each task
        
        progress_bar.close()  # Close the progress bar at the end

    def generate_prompt(self, context):
        # Generate the prompt for the Anthropic model
        # Replace the following line with the appropriate prompt structure
        if(self.model_provider not in ["OpenAI", "Anthropic"]):
            # if self.model_provider in ['llama-3-inst', 'mistral-inst']:
            #     messages = [
            #         {
            #             'role': 'user', 
            #             'content': f"Based on the content of the book, Question: {self.retrieval_question} The book begins. {context} The book ends. Based on the content of the book, Question: {self.retrieval_question} \n "
            #         }
            #     ]
            #     test_format = self.enc.apply_chat_template(
            #         messages, tokenize=False, add_generation_prompt=True)
            # else:
            #     test_format=f"<|im_start|> This is a very long story book: <book> {context} </book>.\n Based on the content of the book, Question: {self.retrieval_question}\nAnswer:"
            test_format=f"<|im_start|> Based on the content of the book, Question: {self.retrieval_question}. This is a very long story book: <book> {context} </book>.\n Based on the content of the book, Question: {self.retrieval_question}\nAnswer:"
            return test_format
        else: 
            return [
                {
                    "role": "system",
                    "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
                },
                {
                    "role": "user",
                    "content": context
                    },
                {
                    "role": "user",
                    "content": f"{self.retrieval_question} Don't give information outside the document or repeat your findings. The document definitely contains the answer, and I'm 100% sure. So try your best to find it."
                },
                {
                    "role": "assistant",
                    "content":"",
                },
                
            ]

    def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                print("result exists, skipping")
                return
            else:
                print("result does not exist, testing")

        # Go generate the required length context and place your needle statement in
        context = self.generate_context(context_length, depth_percent)

        # Prepare your message to send to the model you're going to evaluate
        prompt = self.generate_prompt(context)
        test_start_time = time.time()
        if(self.model_provider in ["OpenAI", "Anthropic"]):
            # import ipdb; ipdb.set_trace()
            response = self.model_to_test.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                max_tokens=300,
                temperature=0
            )
            response = response.choices[0].message.content
        else:
            prompt = self.enc(prompt, return_tensors="pt")
            input_ids = prompt['input_ids'].to(self.model_to_test.device)
            kwargs = dict(
                input_ids=input_ids, 
                max_new_tokens=50
            )
            if self.config.type in ['qllm']:
                kwargs['question_ids'] = extract_question_id(dataset=None, tokenizer=self.enc, tokenized_prompt=input_ids, data=None, question=self.needle)
            if self.config.type in ['qllm', 'inf-llm', 'stream-llm', 'infinite-lm']:
                kwargs['chunk_size'] = self.chunk_size

            with torch.no_grad():
                output_ids = self.model_to_test.generate(**kwargs)
                if self.config.type not in ['qllm', 'inf-llm', 'stream-llm', 'infinite-lm']:
                    response = self.enc.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
                else:
                    response = output_ids[0]

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        score = scorer.score(self.needle, response)['rouge1'].fmeasure*10

        results = {
            # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
            'model' : self.model_to_test_description,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : self.needle,
            'model_response' : response,
            'score' : score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"Score: {score}")
            print (f"Response: {response}\n")

        context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'

        if self.save_contexts:
            results['file_name'] = context_file_location

            # Save the context to file for retesting
            if not os.path.exists('contexts'):
                os.makedirs('contexts')

            if not os.path.exists(f'contexts/{self.model_version}'):
                os.makedirs(f'contexts/{self.model_version}')

            with open(f'contexts/{self.model_version}/{context_file_location}_context.txt', 'w') as f:
                f.write(context)
            
        if self.save_results:            
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            # Save the result to file for retesting
            p = f'{self.output_dir}/{context_file_location}_results.json'
            print("Writing at %s" % p)
            with open(p, 'w') as f:
                json.dump(results, f)

    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = self.output_dir
        print("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    # import ipdb; ipdb.set_trace()
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context
    
    def encode_text_to_tokens(self, text):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM", 'raise', 'llama-3-inst', 'mistral-inst']:
            return self.enc.encode(text)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
    
    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle) # [1:]
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = max(int(len(tokens_context) * (depth_percent / 100)), 1)
            # import ipdb; ipdb.set_trace()

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            if(self.model_provider in ["LLaMA", "LongLLaMA", 'llama-3-inst']): period_tokens = [29889, 869]
            elif(self.model_provider in ["Mistral", 'mistral-inst']): period_tokens = [842, 28723]
            elif(self.model_provider == "GLM"): period_tokens = [918, 30930]
            else: period_tokens = self.encode_text_to_tokens('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            print("insertion at %d" % insertion_point)
            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM", 'llama-3-inst', 'mistral-inst']:
            return len(self.enc.encode(context))
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            encoded = self.enc.encode(context)
            return len(self.enc.encode(context).ids)
        else:
            
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)
        files = glob.glob(f"{self.haystack_dir}/*.txt")
        assert len(files) > 0, f'not find txt in {self.haystack_dir}'
        current_l = 0
        while current_l < max_context_length:
            print(f'{current_l}/{max_context_length}', end='\r')
            for file in files:
                with open(file, 'r') as f:
                    context += f.read()
            current_l = self.get_context_length_in_tokens(context)
        print()
        return context

    def get_tokens_from_context(self, context):
        if self.model_provider in ['llama-3-inst', 'mistral-inst', "OpenAI", "LLaMA", "Mistral", "GLM"]:
            return self.enc.encode(context)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(context).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        
    def decode_tokens(self, tokens, context_length=None):
        if self.model_provider in ['llama-3-inst', 'mistral-inst', "OpenAI", "LLaMA", "Mistral", "GLM"]:
            return self.enc.decode(tokens[:context_length])
        elif self.model_provider == "Anthropic":
            # Assuming you have a different decoder for Anthropic
            return self.enc.decode(tokens[:context_length])
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self, args):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        #asyncio.run(self.run_test())
        self.run_test(args)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    seed_everything(42)
    
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--s_len', metavar='N', type=int, help='a number')
    parser.add_argument('-e', '--e_len', metavar='N', type=int, help='a number')
    parser.add_argument('--api_key', type=str, default="", help='OpenAI API Key')
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--reverse_order", action='store_true')
    # parser = add_args(parser)
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config_path)

    ht = LLMNeedleHaystackTester(save_contexts=True,
                                 save_results=True,
                                 openai_api_key=args.api_key,
                                 config=config,
                                 output_dir=args.output_dir,
                                 reverse_order=args.reverse_order,
                                 )

    ht.start_test(args)