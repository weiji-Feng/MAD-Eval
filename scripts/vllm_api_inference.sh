MODEL_NAME=gpt-3.5-turbo-1106
MAX_TOKENS=1024 # qwen 1024, other 2048
DEV_SET=Writing    # Reasoning,Writing,Understanding,Coding,Chatbot_Arena
GEN_OUTPUT_PATH=./outputs/inference/

export OPENAI_API_KEY='sk-qY9kEIKe3BL3ZI4BA2Fd3141E3Bc41DaBf1f8a67D9E36a3b'
export GOOGLE_API_KEY=AIzaSyDFZSnFnticlXwZRO8ldnfX3xfCSCGN6cM
python vllm_api_inference.py \
    --model_name ${MODEL_NAME} \
    --max_tokens ${MAX_TOKENS} \
    --temperature 0.0 \
    --output_file_name ${GEN_OUTPUT_PATH} \
    --dev_set ${DEV_SET} \
    --sample_num -1 \
    --api_batch 100 \
