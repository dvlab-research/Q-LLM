# <img src="img/quickllama.png" width="50"> QuickLLaMA: Query-aware Inference Acceleration for Large Language Models

## Overview
We introduce Query-aware Inference for LLMs (Q-LLM), a system designed to process extensive sequences akin to human cognition. By focusing on memory data relevant to a given query, Q-LLM can accurately capture pertinent information within a fixed window size and provide precise answers to queries. It doesn't require extra training and can be seamlessly integrated with any LLMs. Q-LLM using LLaMA3 (QuickLLaMA) can read Harry Potter within 30s and accurately answer the questions. Q-LLM improved by 7.17% compared to the current state-of-the-art on LLaMA3, and by 3.26% on Mistral on the $\infty$-bench. In the Needle-in-a-Haystack task, On widely recognized benchmarks, Q-LLM improved upon the current SOTA by 7.0% on Mistral and achieves 100% on LLaMA3. 

![alt text](img/framework.png)

## Examples
![alt text](img/exp_harrypotter_details.png) 
![alt text](img/exp_unpretraied.png)
![alt text](img/exp_mood_summarize.png) 
![alt text](img/exp_mood_connection.png) 
![alt text](img/exp_mood_improvement.png) 
![alt text](img/exp_paper_review.png) 
![alt text](img/exp_paper_summarize.png) 
![alt text](img/exp_sum_papers.png) 
![alt text](img/exp_needle.png) 
![alt text](img/exp_kv_retrieval.png) 
![alt text](img/exp_jouney_to_west.png) 


## Usage
The code will be released soon.