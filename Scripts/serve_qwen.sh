export CUDA_VISIBLE_DEVICES=0

vllm serve Qwen/Qwen3.5-4B \
    --port "${BADMINTON_LLM_PORT:-16000}" \
    --max-model-len 16384 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.7 \
    --generation-config vllm \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --reasoning-parser qwen3 \
    --enable-force-include-usage \
    --language-model-only \
    --enable-prefix-caching
