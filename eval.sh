# model path /data/yangtao/cot/llama-main/model/llama-7b-hf
# device_map set to auto/balanced_low_0


CUDA_VISIBLE_DEVICES=4,5 python main.py mmlu --model_name llama --model_path /data/yangtao/cot/llama-main/model/llama-7b-hf --no_split "LlaMADecoderLayer"
CUDA_VISIBLE_DEVICES=4,5 python main.py mmlu --model_name llama --model_path /data/yangtao/cot/llama-main/model/llama-7b-hf --model_para
# CUDA_VISIBLE_DEVICES=4,5 python main.py mmlu --model_name causal --model_path /data/yangtao/gpt2
# python main.py mmlu --model_name llama --model_path /data/yangtao/cot/llama-main/model/llama-7b-hf 