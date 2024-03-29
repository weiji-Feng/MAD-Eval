dataset_name=Writing
model=gpt-4-1106-preview
output_path=./data/instruction/${dataset_name}.jsonl
max_tokens=2048
temperature=0.7
top_p=0.9
iter=1

export OPENAI_API_KEY='sk-qY9kEIKe3BL3ZI4BA2Fd3141E3Bc41DaBf1f8a67D9E36a3b'
python instruction_evol.py \
  --dataset_name ${dataset_name} \
  --output_path ${output_path} \
  --model ${model} \
  --max_tokens ${max_tokens} \
  --temperature ${temperature} \
  --top_p ${top_p} \
  --iter ${iter} \
  --api_batch 200 \

# --instruction_gen use when generating instruction