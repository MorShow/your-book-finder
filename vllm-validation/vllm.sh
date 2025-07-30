docker run --gpus device=0 -p 8000:8000 \
  vllm/vllm-openai:v0.8.5 \
  --model meta-llama/Llama-3.1-70B \
  --trust-remote-code \
  --gpu_memory_utilization 0.8