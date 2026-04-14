vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 16000 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9