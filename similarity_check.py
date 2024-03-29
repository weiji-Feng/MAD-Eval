import os
import json
import yaml
import random
import argparse
import asyncio
from typing import List
from tqdm import tqdm

from utils import OpenAIChat
from transformers import AutoTokenizer
import google.generativeai as genai
from bert_score import score
import openai
# from openai import OpenAI
import pandas as pd
import numpy as np

import time
import itertools
# import sys
# sys.path.insert(0, "/home/dky/khfeng/easy-arena")
            

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


if __name__ == "__main__":
    # set args
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_prompt_type', type=str, required=True)
    parser.add_argument('--dev_set', type=str, required=True)
    parser.add_argument('--eval_models', type=str, required=True, help="the models mentioned in your --instruction_path")
    parser.add_argument('--output_path', type=str, default="./output/eval")
    parser.add_argument('--model', type=str, default="gpt3", help="gpt3 or gpt4")
    parser.add_argument('--max_tokens', type=int, default=2048, help="max tokens")
    parser.add_argument('--temperature', type=float, default=0.7, help="temperature")
    parser.add_argument('--top_p', type=float, default=0.95, help="top_p")
    parser.add_argument('--eval_only', action="store_true")
    parser.add_argument('--api_batch', type=int, default=5)
    parser.add_argument('--sample_num', type=int, default=-1, 
                        help="sample num, -1 means all, if you want to sample, please set it greater than 0")
    args = parser.parse_args()
    
    # collect the prompt about instruction generation and self-curation.
    with open("./utils/prompts/similarity.yaml", "r") as f:
        prompt_gens = yaml.load(f, Loader=yaml.FullLoader)
    prompt_gen = prompt_gens[args.gen_prompt_type]
    
    if args.dev_set != 'chatbot_arena':
        # instruction path include all the inference results of all the models
        inference_path = os.path.join('./outputs/inference', args.dev_set)
        instruction_path = os.path.join('./data/instruction', f'{args.dev_set}.jsonl')
        models_for_eval = args.eval_models.split(",")   # the models that need to be evaluated     
        model_pairs = list(itertools.combinations(models_for_eval, 2)) 
        print(f'>>>>>> total models: {len(models_for_eval)}, we will evaluate {len(model_pairs)} pairs of models')
    else:
        arena_path = './outputs/eval/chatbot_arena'
        model_pairs = [file.replace('.jsonl', '').split('-vs-') for file in os.listdir(arena_path)]
        print(f'>>>>>> we will evaluate {len(model_pairs)} pairs of models')
    
    # load chatbot
    print(f"using model: {args.model}")
    chat = OpenAIChat(model_name=args.model, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, json_mode=True)
    
    for model_pair in model_pairs:
        model_1, model_2 = model_pair
        # set output path
        output_path = os.path.join(args.output_path, args.dev_set, f'{model_1}-vs-{model_2}.jsonl')
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        
        if args.dev_set != 'chatbot_arena':
            # load the responses
            with open(os.path.join(inference_path, f'{model_1}.jsonl'), "r") as f:
                responses_1 = [json.loads(line) for line in f.readlines()]
                responses_1 = responses_1[:args.sample_num] if args.sample_num > 0 else responses_1
            f.close()
            with open(os.path.join(inference_path, f'{model_2}.jsonl'), "r") as f:
                responses_2 = [json.loads(line) for line in f.readlines()]
                responses_2 = responses_2[:args.sample_num] if args.sample_num > 0 else responses_2
            f.close()   
            assert len(responses_1) == len(responses_2), "responses_1 and responses_2 must have the same length"
            with open(instruction_path, "r") as f:
                instructions = [json.loads(line) for line in f.readlines()]
                # instructions = json.load(f)
                if args.dev_set == 'mbppplus':
                    for i, d in enumerate(instructions[:len(responses_1)]):
                        if responses_1[i]['correct'] > responses_2[i]['correct']:
                            winner = model_1
                        elif responses_1[i]['correct'] < responses_2[i]['correct']:
                            winner = model_2
                        else:
                            winner = 'tie'
                        instructions[i] = {'instruction': d['instruction'].replace('    ', '\t'), 'input': "", 'output': winner}
                else:
                    instructions = [{'instruction': d['response'].strip(), 'input': "", 'output': ""} for d in instructions[:len(responses_1)]]
            f.close()      
            for r1, r2, ins in zip(responses_1, responses_2, instructions):
                assert ins['instruction'] in r1['instruction'].strip() and ins['instruction'] in r2['instruction'].strip(), f"instructions differ"
            
            save_dict_list = [
                {
                    'instruction': d['instruction'], 
                    'input': d['input'], 
                    'output': d['output'], 
                    'response_1': x['output'], 
                    'response_2': y['output']
                } for d, x, y in zip(instructions, responses_1, responses_2)
            ]
        
        else:
            with open(os.path.join(arena_path, f'{model_1}-vs-{model_2}.jsonl'), "r") as f:
                save_dict_list = [json.loads(line) for line in f.readlines()]
            f.close()
        
        # delete the useless data
        save_dict_list = [d for d in save_dict_list if d['response_1'] != None and d['response_2'] != None and len(d['response_1'].strip()) > 0 and len(d['response_2'].strip()) > 0]
        save_dict_list = [d for d in save_dict_list if max(len(d['response_1']), len(d['response_2'])) / min(len(d['response_1']), len(d['response_2'])) < 4]
        save_dict_list = [d for d in save_dict_list if len(d['response_1'].strip().split()) > 20 and len(d['response_2'].strip().split()) > 20]
        print(f">>>>>> Filtered data has {len(save_dict_list)} items.")
        
        if 'text-embedding-ada' in args.model or args.model == 'bert-score':
            message_list = [[d['response_1'].replace('\n', ' '), d['response_2'].replace('\n', ' ')] for d in save_dict_list]
        else:
            message_list = [
                [
                    {'role': "system", 'content': prompt_gen['system']},
                    {'role': "user", 'content': prompt_gen['user'].format(response_1=d['response_1'], response_2=d['response_2'])}
                ] for d in save_dict_list
            ]
        message_list = message_list[:args.sample_num] if args.sample_num > 0 else message_list
        print('>>>>>> the number of messages: ', len(message_list))
        
        # load the saved responses
        if os.path.exists(output_path):
            # continue
            f = open(output_path, "r")
            lines = f.readlines()
            begin_index = len([json.loads(line) for line in lines]) if len(lines) > 0 else 0
            responses = [json.loads(line) for line in lines] if len(lines) > 0 else []
            f.close()
        else:
            begin_index = 0
            responses = []
        # new chat
        f = open(os.path.join(output_path), "w")
        # step 1: save the original response
        for i in range(begin_index):
            assert save_dict_list[i]['instruction'] == responses[i]['instruction'], "save_dict_list and responses must have the same instruction"
            f.write(json.dumps(responses[i]) + "\n")
        # step 2: evaluate the similarity for the new responses
        print('>>>>>> Start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('='*30, f'OpenAI Chat API - {model_1} vs {model_2} OUTPUT: ', '='*30)
        batch_size = args.api_batch
        total_len = range(begin_index, len(message_list), batch_size)
        for index in tqdm(total_len, total=len(total_len), desc=f'>>>>>> Progress: {model_1} vs {model_2}'):
            if args.model == "gemini-pro":
                responses_index = [chat.generate_content(m).text for m in message_list[index:index+batch_size]]
            
            elif "text-embedding-ada" in args.model:
                responses_1 = asyncio.run(chat.async_run(messages_list=[m[0] for m in message_list[index:index+batch_size]], expected_type=List))
                responses_2 = asyncio.run(chat.async_run(messages_list=[m[1] for m in message_list[index:index+batch_size]], expected_type=List))
                responses_index = [(r1, r2) for r1, r2 in zip(responses_1, responses_2)]
                
            elif args.model == 'bert-score':
                response_1 = [m[0] for m in message_list[index:index+batch_size]]
                response_2 = [m[1] for m in message_list[index:index+batch_size]]
                _, _, responses_index = score(response_1, response_2, model_type="microsoft/deberta-xlarge-mnli", verbose=True)
                responses_index = responses_index.tolist()
            else:
                responses_index = asyncio.run(chat.async_run(messages_list=message_list[index:index+batch_size], expected_type=List))
            for i in range(len(responses_index)):
                if args.model == 'text-embedding-ada-002':
                    embedding_1 = responses_index[i][0]
                    embedding_2 = responses_index[i][1]
                    similarity_score = cosine_similarity(embedding_1, embedding_2)
                    save_dict_list[index+i]['score'] = similarity_score
                    save_dict_list[index+i]['explanation'] = ""
                    save_dict_list[index+i]['embedding_1'] = embedding_1
                    save_dict_list[index+i]['embedding_2'] = embedding_2
                elif args.model == 'bert-score':
                    similarity_score = responses_index[i]
                    save_dict_list[index+i]['score'] = similarity_score
                    save_dict_list[index+i]['explanation'] = ""
                else:
                    try:
                        response = json.loads(responses_index[i])
                    except:
                        save_dict_list[index+i]['explanation'] = responses_index[i]
                        f.write(json.dumps(save_dict_list[index+i]) + "\n")
                        continue
                    try:
                        save_dict_list[index+i]['score'] = response['score']
                    except:
                        save_dict_list[index+i]['score'] = 2.0
                    try:
                        save_dict_list[index+i]['explanation'] = response['explanation']
                    except:
                        save_dict_list[index+i]['explanation'] = ""
                f.write(json.dumps(save_dict_list[index+i]) + "\n")
            if len(total_len) > 1 and args.model != 'bert-score':
                time.sleep(10)
            responses += responses_index
        print('='*65)
        print('>>>>>> Finish time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        f.close()
        assert len(responses) == len(message_list)