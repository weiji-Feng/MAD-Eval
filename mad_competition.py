import os
import json
import yaml
import random
import argparse
import asyncio
from typing import List
from tqdm import tqdm

from utils import OpenAIChat
import numpy as np
import time
from datetime import datetime


def get_instruction_embedding(dev_set, embedding_model='text-embedding-ada-002'):
    # this function is used to get the instruction embedding
    f = open(f'./data/instruction/{dev_set}.jsonl', 'r')
    instruction_seed = [json.loads(line) for line in f.readlines()]
    f.close()
    
    if 'response_embedding' not in instruction_seed[0].keys():
        # to generate the instruction embedding if there is no embedding
        chat = OpenAIChat(model_name=embedding_model)
        batch_size = 100
        total_len = range(0, len(instruction_seed), batch_size)
        for index in tqdm(total_len, total=len(total_len), desc=f'>>>>>> Generate embeddings...'):
            responses_index = asyncio.run(chat.async_run(messages_list=[m['response'] for m in instruction_seed[index:index+batch_size]], expected_type=List))
            for i in range(len(responses_index)):
                instruction_seed[index+i]['response_embedding'] = responses_index[i]
            time.sleep(5)
        assert 'response_embedding' in instruction_seed[-1].keys(), "generate instruction embedding failed"
        
        # save the instruction embedding
        f = open(f'./data/instruction/{dev_set}.jsonl', 'w')
        for line in instruction_seed:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    
    # get the instruction embedding
    if dev_set == 'Reasoning' or dev_set == 'Understanding' or dev_set == 'Code':
        instruction_dict = {x['response'].strip(): [x['response_embedding'], x['answer']] for x in instruction_seed}
    else:
        instruction_dict = {x['response'].strip(): x['response_embedding'] for x in instruction_seed}
    return instruction_dict

def cosine_similarity(v1, v2):
    # calculate the cosine similarity of the two embeddings
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def instruction_similarity(item, selected_list):
    # Param selected_list: the corrently selected data items
    # Param item: the data item
    assert len(selected_list) > 0, 'the number of the corrently selected data items must be more than 0'
    instructions = [x['instruction_embedding'] for x in selected_list]
    similarities = [cosine_similarity(item['instruction_embedding'], x) for x in instructions]
    return max(similarities)

def top_K_items(data_items, dev_set, K=10, lambda_1=0.0):
    # the data_items is filtered.
    def similarity_sort(item, s, lambda_1):
        return float(item['score']) + lambda_1 * instruction_similarity(item, s)
    
    # step 1: get the instruction embedding of each data item
    instruction_dict = get_instruction_embedding(dev_set)
    for x, y in instruction_dict.items():
        assert y[0] is not None and y[1] is not None, f"get instruction embedding failed..."
    
    for i in tqdm(range(len(data_items)), total=len(data_items), desc='get instruction embedding'):
        if dev_set == 'Reasoning' or dev_set == 'Understanding' or dev_set == 'Code':
            data_items[i]['instruction_embedding'] = instruction_dict.get(data_items[i]['instruction'].strip(), None)[0]
            data_items[i]['answer'] = instruction_dict.get(data_items[i]['instruction'].strip(), None)[1]
        else:
            data_items[i]['instruction_embedding'] = instruction_dict.get(data_items[i]['instruction'].strip(), None)
        assert data_items[i]['instruction_embedding'] is not None, f"data {i+1}: {data_items[i]['instruction']} get instruction embedding failed..."
    
    # step 2: select the top K unsimilar data items
    select_data = []
    while len(select_data) < K:
        print(">>>>>> Select data item: ", len(select_data) + 1)
        # sort the data items
        if len(select_data) == 0:
            data_items.sort(key=lambda x: float(x['score']), reverse=False)
        else:
            data_items.sort(key=lambda x: similarity_sort(x, select_data, lambda_1), reverse=False)
        select_data.append(data_items.pop(0))

    return select_data


if __name__ == "__main__":
    # set args
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='Writing')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--eval_model', type=str, default="text-embedding-ada-002", help='text-embedding-ada-002, bert-score, gpt-4-1106-preview')
    args = parser.parse_args()
    # domain: the domain of evaluation
    # eval_model: the metric to evaluate similarity

    # the directory saving llm inference responses of a specific scenario
    eval_dir = f'./outputs/eval/{args.eval_model}/{args.domain}' 

    save_dict_list = []     # The selected instruction and responses using MAD competition.
    if not os.path.exists(f'./outputs/MAD/{args.eval_model}/{args.domain}.jsonl'):
        # the path to save select data
        os.makedirs(f'./outputs/MAD/{args.eval_model}', exist_ok=True)
        save_f = open(f'./outputs/MAD/{args.eval_model}/{args.domain}.jsonl', 'w')

    for file in os.listdir(eval_dir):
        now_path = os.path.join(eval_dir, file)
        print(">>>>>> current data length: ", len(save_dict_list))
        f = open(now_path, 'r')     # get each evaluation file
        model_a, model_b = file.replace(f'.jsonl', '').split('-vs-')
        
        data = [json.loads(line) for line in f.readlines()] # load the data
        for i, d in enumerate(data):
            try:
                data[i]['score'] = float(d['score'])
            except:
                data[i]['score'] = 2.0
        # delete the useless data
        data = [d for d in data if d['response_1'] != None and d['response_2'] != None and len(d['response_1'].strip()) > 0 and len(d['response_2'].strip()) > 0]
        data = [d for d in data if max(len(d['response_1']), len(d['response_2'])) / min(len(d['response_1'])+1, len(d['response_2'])+1) < 4]
        data = [d for d in data if len(d['response_1'].strip().split()) > 20 and len(d['response_2'].strip().split()) > 20]
        
        if args.domain == 'Chatbot_Arena':
            data.sort(key=lambda x: float(x['score']), reverse=False)
            top_k_data = data[:args.top_k]      # don't need to calculate instruction similarity
        else:
            top_k_data = top_K_items(data, args.domain, K=args.top_k)
        
        for d in top_k_data:
            top_k_data[i]['source'] = file
            save_f.write(json.dumps(d, ensure_ascii=False) + "\n")
    save_f.close()