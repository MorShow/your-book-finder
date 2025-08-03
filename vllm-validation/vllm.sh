docker run --gpus device=0 -p 8000:8000 \
  -e VLLM_USE_V1=0 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.8.5 \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --max-model-len=1024 \
  --gpu_memory_utilization=0.96 \
  --trust-remote-code