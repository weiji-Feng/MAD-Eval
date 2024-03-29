from vllm import LLM, SamplingParams
# from deepspeed.ops.adam import FusedAdam
import os
import pdb
import re
import json
import pdb
import argparse
import torch
from tqdm import tqdm
# import sys
# sys.path.insert(0, '/home/dky/khfeng/easy-arena')

model_prompt = {
    'openchat-3.5': 'GPT4 Correct User: {input}<|end_of_turn|>GPT4 Correct Assistant:',
    'null': '{input}\n',
}
model_prompt['vicuna-13b'] = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {input} ASSISTANT:"""
model_prompt['chatglm3-6b'] = "<|system|>\nYou are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.<|user|>\n{input}<|assistant|>"
model_prompt['WizardLM-13b'] = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {input} ASSISTANT:"""
model_prompt['Qwen-14b-chat'] = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"


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
            instruction = "Answer the following question step by step:\n"
            prompt_list.append((instruction + d['response'].strip(), ))
        elif dev_set == 'Understanding' or dev_set == 'Writing' or dev_set == 'Coding':
            prompt_list.append((d['response'].strip(), ))
        else:
            raise ValueError('no such dev set available')
    return prompt_list


if __name__ == '__main__':
    # set args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--tokenizer_dir', type=str, default=None)
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.05)
    parser.add_argument('--frequency_penalty', type=float, default=0.05)
    parser.add_argument('--output_file_name', type=str, default='./output/infernece')
    parser.add_argument('--stop', type=str, nargs='+', default=[],
                        help="you can pass one or multiple stop strings to halt the generation process.")
    parser.add_argument('--dev_set', type=str, default='Writing')
    parser.add_argument('--prompt_type', type=str, default='null')
    parser.add_argument('--sample_num', type=int, default=-1, )
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--max_num_batched_tokens', type=int, default=4096)
    args = parser.parse_args()
    
    if args.eval_only == False:
        # part 1 we set the model
        num_gpus = torch.cuda.device_count()
        another_args = {'max_num_batched_tokens': args.max_num_batched_tokens} 
        tokenizer_dir = args.tokenizer_dir if args.tokenizer_dir else args.model_dir
        llm = LLM(model=args.model_dir, tokenizer=tokenizer_dir, tensor_parallel_size=num_gpus, trust_remote_code=True, **another_args)
        print('>>>>>> model loaded')
        # part 2 we set the sampling params
        sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                                            stop=args.stop, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty)
        
        dev_sets = args.dev_set.split(',')
        for dev_set in dev_sets:
            output_file_name = os.path.join(args.output_file_name, dev_set, f'{args.prompt_type}.jsonl')
            if not os.path.exists(os.path.dirname(output_file_name)):
                os.makedirs(os.path.dirname(output_file_name))
            # part 3 we prepare raw queries and wrap them with target prompt
            raw_queries = get_raw_inputs(dev_set)
            save_dict_list = [query[0] for query in raw_queries] if args.dev_set != 'alpaca-eval' else [query for query in raw_queries]
            # prompt = prompt_mapping[args.prompt_type]
            prompt = model_prompt[args.prompt_type]
            if args.dev_set == 'Reasoning':
                processed_prompts = [query[0] for query in raw_queries]
            else:
                processed_prompts = [prompt.format(input=query[0]) for query in raw_queries]
            processed_prompts = processed_prompts[:args.sample_num] if args.sample_num > 0 else processed_prompts
            print(processed_prompts[0])
            
            # part 4 we generate, note that vllm is async so extra sorting is needed
            outputs = llm.generate(processed_prompts, sampling_params)
            sorted_outputs = sorted(outputs, key=lambda output: int(output.request_id))
            for i, output in enumerate(sorted_outputs):
                counter = 5
                if len(output.outputs[0].text.strip()) < 10:
                    while len(output.outputs[0].text.strip()) < 10 and counter > 0:
                        output = llm.generate(processed_prompts[i], sampling_params)[0]
                        counter -= 1
                    sorted_outputs[i] = output
            print('>>>>>> generation done')

            # part 5 we save the results
            with open(output_file_name, "w") as f:
                for id, (instruction, output) in enumerate(zip(save_dict_list, sorted_outputs)):
                    # note that `prompt`s are the wrapped ones
                    f.write(json.dumps({'id': id, 'instruction': instruction, 'output': output.outputs[0].text}, ensure_ascii=False) + '\n')
            print('>>>>>> saving done')
