import torch

def find_subtensor(main_tensor, sub_tensor):
    main_length = main_tensor.size(0)
    sub_length = sub_tensor.size(0)
    
    # 如果子序列比主序列长，直接返回未找到
    if sub_length > main_length:
        return (-1, -1)
    
    if sub_length > 2:
        sub_tensor = sub_tensor[1:-1]
        sub_length = sub_tensor.size(0)

        # 窗口滑动比较
        for start_index in range(main_length - sub_length + 1):
            end_index = start_index + sub_length
            # 检查从 start_index 到 end_index 的区域是否与子 tensor 相匹配
            if torch.equal(main_tensor[start_index:end_index], sub_tensor):
                return (start_index-1, end_index+1)  # 返回起点和终点位置

        # print(main_tensor[:128], '\n', sub_tensor)
        return (-1, -1)
    else:
        for start_index in range(main_length - sub_length + 1):
            end_index = start_index + sub_length
            # 检查从 start_index 到 end_index 的区域是否与子 tensor 相匹配
            if torch.equal(main_tensor[start_index:end_index], sub_tensor):
                return (start_index, end_index)  # 返回起点和终点位置

        # print(main_tensor[:128], '\n', sub_tensor)
        return (-1, -1)

def extract_question_id(dataset=None, tokenizer=None, tokenized_prompt=None, data=None, question=None):
    if question is None:
        assert dataset is not None
        if dataset.startswith('custom'):
            question = '{input}'.format(**data)
        elif dataset in ['passkey', 'number_string', 
            'narrativeqa', 'qasper', "multifieldqa_en",
            "hotpotqa", "2wikimqa", "musique", "qmsum",
            "passage_retrieval_en",]:
            question = '{input}'.format(**data)
        # infinite bench
        elif dataset.startswith('kv_retrieval'):
            question = '{key}'.format(**data)
        elif dataset == 'code_debug':
            question = 'Which funtion has deliberate error?'.format(**data)
        elif dataset in ['math_find']:
            question = '{prefix}'.format(**data)
        elif dataset in ['longbook_choice_eng', 'longbook_qa_eng']:
            question = '{question}'.format(**data)
        elif dataset == 'longbook_sum_eng':
            question = 'Summarize the following book'
        elif dataset == 'longdialogue_qa_eng':
            question = 'Below is a dialogue script where one random occurrence of a character name is replaced with "$$MASK$$", and you should try to guess who that character is'
        elif dataset == 'math_calc':
            question = 'Compute the intermediate values'
        elif dataset == 'code_run':
            question = 'Please give me the exact number of the return value of {func_call}'.format(**data)
        # long bench
        elif dataset == 'gov_report':
            question = 'Write a one-page summary of the report'
        elif dataset == "multi_news":
            question = 'Write a one-page summary of all news'
        elif dataset == "trec":
            question = '{input}'.format(**data).split('\n')[0]
        elif dataset == "triviaqa":
            question = '{input}'.format(**data).split('Answer:')[0]
        elif dataset == "samsum":
            question = '{input}'.format(**data).split('Summary:')[0]
        elif dataset == 'passage_count':
            question = 'how many non-repeating paragraphs are there in total'
        elif dataset in ["lcc", 'repobench-p']:
            question = 'Please complete the code given below'
        else:
            import pdb;pdb.set_trace()
            return [(-1, -1)]
        
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenized_question = tokenizer(
        question, truncation=False, return_tensors="pt", add_special_tokens=False, padding=True,
        # FIXME: whether to use padding or truncation or other techniques
    ).input_ids

    st, ed = find_subtensor(tokenized_prompt, tokenized_question[tokenized_question != tokenizer.pad_token_id])
    return st, ed
