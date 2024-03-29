DEV_SET=Writing
EVAL_MODELS=chatglm3-6b,gpt-3.5-turbo-1106         # qwen-14b,vicuna-13b,wizardlm-13b,chatglm3-6b,gpt-4-1106-preview,gpt-3.5-turbo-1106,openchat-3.5,gemini-pro
model=gpt-4-1106-preview       # gpt-4-1106-preview, gpt-3.5-turbo-1106, text-embedding-ada-002
gen_prompt_type=gpt-4-eval
max_tokens=1024
temperature=0.0
sample_num=-1
output_path=./outputs/eval/${model}

export OPENAI_API_KEY='sk-qY9kEIKe3BL3ZI4BA2Fd3141E3Bc41DaBf1f8a67D9E36a3b'
CUDA_VISIBLE_DEVICES=0,1 python similarity_check.py \
    --gen_prompt_type ${gen_prompt_type} \
    --dev_set ${DEV_SET} \
    --eval_models ${EVAL_MODELS} \
    --model ${model} \
    --max_tokens ${max_tokens} \
    --temperature ${temperature} \
    --sample_num ${sample_num} \
    --output_path ${output_path} \
    --api_batch 200 \
