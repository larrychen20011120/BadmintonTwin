export CUDA_VISIBLE_DEVICES=0

vllm serve Qwen/Qwen3.5-4B \
    --port 16001 \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92 \
    --generation-config vllm \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --reasoning-parser qwen3 \
    --structured-outputs-config.backend xgrammar \
    --enable-force-include-usage \
    --language-model-only \
    --enable-prefix-caching
