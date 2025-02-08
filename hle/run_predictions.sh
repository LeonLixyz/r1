#!/bin/bash

# Set environment variables if needed
export TOGETHER_API_KEY="9cb3e0c48d9412890a81871625ce2d9e68b422932268935a3158a018dba8232e"

# Run the prediction script
python hle_eval/run_model_predictions.py \
    --dataset "cais/hle" \
    --model "deepseek-ai/DeepSeek-R1" \
    --max_tokens 32768 \
    --temperature 0.6 \
    --num_workers 16
