MODEL_NAME="deepseek-ai/DeepSeek-R1"

# Set the tensor parallel size.
TENSOR_PARALLEL_SIZE=8

# Set the GPU memory utilization (e.g., 0.9 for 90%).
GPU_MEMORY_UTILIZATION=0.95

HF_HOME_DIR="/workspace/models"

ENABLE_PREFIX_CACHING="--enable_prefix_caching"

# Set the port number (default: 8000)
PORT=${PORT:-8001}

echo "Starting vLLM server for model ${MODEL_NAME}"
echo "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
echo "GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Download Directory: ${HF_HOME_DIR}"
echo "Prefix Caching Enabled"
echo "Port: ${PORT}"

# Serve the model with vLLM
vllm serve "${MODEL_NAME}" \
  --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
  --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
  --download_dir "${HF_HOME_DIR}" \
  --enable-reasoning \
  --reasoning-parser deepseek_r1 \
  --enable-prefix-caching \
  --port ${PORT} \
  --max-model-len 128000 \
  --trust-remote-code \
  --max-num-batched-tokens 8192 \
  --enforce-eager