domain=Writing
top_k=10
eval_model=text-embedding-ada-002  # text-embedding-ada-002, bert-score, gpt-4

python mad_competition.py \
    --domain ${domain} \
    --top_k ${top_k} \
    --eval_model ${eval_model}