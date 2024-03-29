import os
import re
import json
import yaml
import random
import argparse
import asyncio
from typing import List
from tqdm import tqdm
import pandas as pd
from utils import OpenAIChat
import google.generativeai as genai

import time
# import sys
# sys.path.insert(0, "/home/dky/khfeng/easy-arena")


def get_raw_prompts(dataset_name):
    if dataset_name in ['Writing', 'Coding', 'Reasoning', 'Understanding']:
        f = open(f'./data/seeds/{dataset_name}.jsonl', 'r')
        data = [json.loads(line) for line in f.readlines()]
    else:
        raise NotImplementedError
    
    prompt_list = []
    for d in data:
        # you should ensure the <question> or <instruction> is the first element. 
        if dataset_name == 'Understanding':
            prompt_list.append((d['question'].strip(), d['topic'].strip(), d['answer'].strip()))
        elif dataset_name == 'Coding':
            prompt_list.append((d['instruction'].strip(), ))
        elif dataset_name == 'Writing':
            prompt_list.append((d['question'].strip(), ))
        elif dataset_name == 'Reasoning':
            prompt_list.append((d['question'].strip(), d['answer'].strip()))
    return prompt_list


if __name__ == "__main__":
    # set args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--output_path', type=str, default="./data/instruction")
    parser.add_argument('--model', type=str, default="gpt-4-1106-preview", help="gpt-3.5, gpt-4. The default prompt is not suitable for gemini-pro.")
    parser.add_argument('--max_tokens', type=int, default=2048, help="max tokens")
    parser.add_argument('--temperature', type=float, default=0.7, help="temperature")
    parser.add_argument('--top_p', type=float, default=0.95, help="top_p")
    parser.add_argument('--api_batch', type=int, default=100)
    parser.add_argument('--iter', type=int, default=2,
                        help="sample num, -1 means all, if you want to sample, please set it greater than 0")
    args = parser.parse_args()

    # collect the prompt about instruction generation and self-curation.
    with open("./utils/prompts/instruction.yaml", "r") as f:
        prompt_gens = yaml.load(f, Loader=yaml.FullLoader)
    prompt_gen = prompt_gens[args.dataset_name]

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
    # load chatbot
    print(f"using model: {args.model}")
    chat = OpenAIChat(model_name=args.model, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, json_mode=True)

    prompt_list = get_raw_prompts(args.dataset_name)
    # print(f">>>>> iter 0, total prompt num: {len(prompt_list)}")
    
    save_dict_list, data_t = [], []         # all data, data in iter t
    for index in tqdm(range(len(prompt_list)), total=len(prompt_list), desc='sample data...'):
        if args.dataset_name == 'Understanding':
            data_t.append({'id': index, 'instruction': prompt_list[index][0], 'domain': prompt_list[index][1],})
        elif args.dataset_name == 'Reasoning':
            data_t.append({'id': index, 'instruction': prompt_list[index][0], 'output': prompt_list[index][1]})
        else:
            data_t.append({'id': index, 'instruction': prompt_list[index][0]})
    # TODO: sample data
    data_t = data_t[:5]
    print(f">>>>> iter 0, total prompt num: {len(data_t)}")
    
    for t in range(args.iter):
        message_list = [
            [
                {'role': "system", 'content': prompt_gen['system']},
                {'role': "user", 'content': prompt_gen['user'].format(**sample)}
            ] for sample in data_t
        ]
        print(message_list[0][1]['content'])
        
        print(">>>>>>> message len: ", len(message_list))
        
        # run chat
        responses = []
        print('>>>>>> Start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('='*30, 'OpenAI Chat API OUTPUT: ', '='*30)
        batch_size = args.api_batch
        total_len = range(0, len(message_list), batch_size)
        for index in tqdm(total_len, total=len(total_len), desc=f'>>>>>> Progress: '):
            responses_index = asyncio.run(chat.async_run(messages_list=message_list[index:index+batch_size], expected_type=List))
            for i in range(len(responses_index)):
                # json model response reformat
                response = json.loads(responses_index[i])
                data_t[index+i]['response'] = response['new_prompt'] if args.dataset_name != 'Reasoning' else response['question']
                if args.dataset_name in ['Understanding', 'Reasoning', 'Coding']:
                    data_t[index+i]['answer'] = response['answer']

            if len(total_len) > 1:
                time.sleep(60)
            if args.dataset_name == 'Coding' or args.dataset_name == 'Understanding' or args.dataset_name == 'Writing':
                responses += [json.loads(d)['new_prompt'] for d in responses_index]
            elif args.dataset_name == 'Reasoning':
                responses += [json.loads(d) for d in responses_index]
            else:
                responses += responses_index
        print('='*65)
        
        print('>>>>>> Finish time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        assert len(responses) == len(message_list)
        
        save_dict_list += data_t
        if args.dataset_name == 'Reasoning':
            data_t = [{'id': j, 'instruction': response['question'], 'output': response['answer']} for j, response in enumerate(responses)]
        else:
            data_t = [{'id': j, 'instruction': response} for j, response in enumerate(responses)]
    
    f = open(os.path.join(args.output_path), "w")
    for d in save_dict_list:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
        
