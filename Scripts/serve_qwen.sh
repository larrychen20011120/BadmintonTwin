vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 16000 \
    --dtype bfloat16 \
    --max-model-len  16384 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92

# vllm serve Qwen/Qwen3.5-9B \ 
#     --port 16000 \ 
#     --tensor-parallel-size 1 \
#     --max-model-len 8192 \
#     --reasoning-parser qwen3 \
#     --gpu-memory-utilization 0.4 \
#     --language-model-only
