MODEL_NAME=chatglm3-6b
BASE_MODEL=/media/weight/${MODEL_NAME}
echo ${BASE_MODEL}
TOKENIZER_PATH=${BASE_MODEL}
MAX_TOKENS=2048
MAX_BATCH_TOKENS=8192
DEV_SET=Writing    # Reasoning,Writing,Summarization,Science_Knowledge
PROMPT_TYPE=${MODEL_NAME}
EVAL_ONLY=false
GEN_OUTPUT_PATH=./outputs/inference/

export RAY_memory_monitor_refresh_ms=0
CUDA_VISIBLE_DEVICES=1 python vllm_inference.py \
    --model_dir ${BASE_MODEL} \
    --tokenizer_dir ${TOKENIZER_PATH} \
    --max_tokens ${MAX_TOKENS} \
    --max_num_batched_tokens ${MAX_BATCH_TOKENS} \
    --temperature 0.7 \
    --output_file_name ${GEN_OUTPUT_PATH} \
    --stop '</s>' '<|user|>' '<|observation|>' '<|assistant|>' \
    --dev_set ${DEV_SET} \
    --prompt_type ${PROMPT_TYPE} \
    --sample_num -1 \

# chatglm3-6b stop '<|user|>' '<|observation|>' '<|assistant|>'