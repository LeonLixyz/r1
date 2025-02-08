#!/bin/bash
# launch.sh - A script to launch the vLLM prediction pipeline

# Default configuration values
MODEL_NAME="deepseek-ai/DeepSeek-R1"
DATASET="cais/hle"
TEMPERATURE=0.6
NUM_WORKERS=512
DEFAULT_HTTP_URL="http://localhost:8001/v1"
STREAM=false  # Default value


# Check if an HTTP URL is provided as the first argument; otherwise, use the default.
if [ $# -ge 1 ]; then
    HTTP_URL="$1"
else
    HTTP_URL="$DEFAULT_HTTP_URL"
fi

echo "Launching predictions with the following settings:"
echo "  Model:         $MODEL_NAME"
echo "  Dataset:       $DATASET"
echo "  Temperature:   $TEMPERATURE"
echo "  Num Workers:   $NUM_WORKERS"
echo "  HTTP URL:      $HTTP_URL"
echo "  Stream:        $STREAM"

# Run the prediction script with the specified parameters.
COMMAND="python run_model_predictions_multiprocessing.py \
    --dataset \"$DATASET\" \
    --model \"$MODEL_NAME\" \
    --temperature \"$TEMPERATURE\" \
    --http_url \"$HTTP_URL\""
    

if [ "$STREAM" = true ]; then
    COMMAND="$COMMAND --stream"
fi

eval "$COMMAND"
