from utils import OpenAIChat
import google.generativeai as genai
import os
import pdb
import re
import json
import pdb
import argparse
import time
import asyncio
from typing import List
from tqdm import tqdm
# import sys
# sys.path.insert(0, '/home/dky/khfeng/easy-arena')


system = {}
system['openchat-3.5'] = ''
system['vicuna-13b'] = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
system['chatglm3-6b'] = "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown."
system['WizardLM-13b'] = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
system['wizardlm-13b'] = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
system['gpt-3.5-turbo-1106'] = 'You are a helpful assistant.'
system['gpt-4-1106-preview'] = 'You are a helpful assistant.'
system['qwen-14b'] = 'You are a helpful assistant.'

def get_raw_inputs(dev_set):
    # in this function, we will get the raw queries for a target dev set
    # TODO: need improvement to avoid hard-coded paths
    try:
        with open(f'./data/instruction/{dev_set}.jsonl', "r") as f:
            data = f.readlines()
    except:
        raise ValueError('no such dev set available')

    prompt_list = []
    for line in data:
        xline = line.strip()
        if xline == '':
            continue
        d = json.loads(xline)
        if dev_set == 'Reasoning':
            instruction = "Please solve the following question step by step and provide the final answer:\n"
            prompt_list.append({'instruction': instruction + d['response'].strip()})
        elif dev_set == 'Science_Knowledge' or dev_set == 'Writing' or dev_set == 'Code':
            prompt_list.append({'instruction': d['response'].strip()})
        else:
            raise ValueError('no such dev set available')
    return prompt_list


if __name__ == '__main__':
    # set args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--output_file_name', type=str, default='./output/infernece')
    parser.add_argument('--dev_set', type=str, default='Writing', help='Writing, Science_Knowledge, Reasoning, Code or Chatbot_Arena')
    parser.add_argument('--sample_num', type=int, default=-1, )
    parser.add_argument('--api_batch', type=int, default=100, )
    args = parser.parse_args()
    
    print(f">>>>>using model: {args.model_name}")
    # This code can inference multiple dev set.
    for dev_set in args.dev_set.split(','):
        output_file_name = os.path.join(args.output_file_name, dev_set, f'{args.model_name}.jsonl')
        if not os.path.exists(os.path.dirname(output_file_name)):
            os.makedirs(os.path.dirname(output_file_name))
        # load prompt data
        prompt_list = get_raw_inputs(dev_set)
        prompt_list = prompt_list[:args.sample_num] if args.sample_num > 0 else prompt_list
        print(f">>>>> load {dev_set}, total prompt num: {len(prompt_list)}")
        
        # load chatbot, using openai API or google genai
        if args.model_name != 'gemini-pro':
            chat = OpenAIChat(model_name=args.model_name, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)
            if args.model_name == 'openchat-3.5' or args.model_name == 'chatglm3-6b':
                message_list = [[{'role': "user", 'content': d['instruction']}] for d in prompt_list]
            else:
                message_list = [[{'role': 'system', 'content': system[args.model_name]}, {'role': "user", 'content': d['instruction']}] for d in prompt_list]
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'], transport='rest')
            generate_config = dict(max_output_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)
            chat = ChatGoogleGenerativeAI(model='gemini-pro', generation_config=generate_config, safety_settings=safety_settings)
            message_list = [d['instruction'] for d in prompt_list]

        save_dict_list = [{'instruction': d['instruction']} for d in prompt_list]
        # run chat
        responses = []
        print('>>>>>> Start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        if args.model_name != 'gemini-pro':
            # Start openai api
            print('='*30, 'OpenAI Chat API OUTPUT: ', '='*30)
            batch_size = args.api_batch
            total_len = range(0, len(message_list), batch_size)
            for index in tqdm(total_len, total=len(total_len), desc=f'>>>>>> Progress {dev_set}: '):
                responses_index = asyncio.run(chat.async_run(messages_list=message_list[index:index+batch_size], expected_type=List))
                for i in range(len(responses_index)):
                    response = responses_index[i]
                    save_dict_list[index+i]['output'] = response
                if len(total_len) > 1 and ('gpt-3.5' in args.model_name or 'gpt-4' in args.model_name):
                    # if use gpt-3.5 or gpt-4, sleep 60s
                    time.sleep(20)
                responses += responses_index
            print('='*65)
        else:
            print('='*30, 'Gemini Chat API OUTPUT: ', '='*30)
            batch_size = 10
            total_len = range(0, len(message_list), batch_size)
            for index in tqdm(total_len, total=len(total_len), desc=f'>>>>>> Progress {dev_set}: '):
                if len(total_len) > 1:
                    time.sleep(10)
                messages = message_list[index:index+batch_size]
                responses_index = []
                for _ in range(10):
                    # test 3 times
                    try:
                        responses_index += chat.batch(messages)
                        break
                    except Exception as e:
                        try:
                            error_i = int(re.findall("index: (.*)", str(e))[0])  # get the error index
                            if error_i > 0:
                                responses_index += chat.batch(messages[:error_i])              # first: collect the responses before the error
                            # second try to generate the response of the error message
                            try:
                                model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)
                                responses_index += [model.generate_content(messages[error_i]).text]
                            except Exception as e:
                                print(f"data {index + error_i} can't be generated")
                                responses_index += [None]
                            messages = messages[error_i+1:]
                            assert len(messages) + len(responses_index) == batch_size, 'Length error'
                            if len(messages) == 0:
                                break
                        except:
                            model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)
                            if len(messages) > 0:
                                for m in messages:
                                    try:
                                        responses_index += [model.generate_content(m).text]
                                    except:
                                        responses_index += [None]
                            break

                for i in range(len(responses_index)):
                    try:
                        response = responses_index[i].content
                    except:
                        response = responses_index[i]
                    save_dict_list[index+i]['output'] = response
                responses += responses_index
            print('='*65)
        print('>>>>>> Finish time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        assert len(responses) == len(message_list)
        
        # save
        f = open(output_file_name, 'w')
        for d in save_dict_list:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    