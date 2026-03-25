#!/bin/bash
# Side-by-side quality comparison: baseline vs q4_0 vs tbq4_0 K-cache
# Uses file redirect + sleep to avoid interactive mode issues

MODEL="${1:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
CLI="./build/bin/llama-cli"
NGL=99
N_PREDICT=48
WAIT=15

PROMPTS=(
    "What is 2+2? Answer in one sentence."
    "Name the three primary colors."
    "Explain gravity in two sentences."
    "What is the capital of Japan? Answer briefly."
    "List the four seasons."
)

CONFIGS=("f16" "q4_0" "tbq4_0")

echo "=== Quality Comparison ==="
echo "Model: $MODEL"
echo "========================="

for i in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$i]}"
    echo ""
    echo "--- PROMPT $((i+1)): $prompt ---"

    for config in "${CONFIGS[@]}"; do
        tmpfile="/tmp/quality_${config}_${i}.txt"
        pkill -f "llama-cli" 2>/dev/null
        sleep 1

        $CLI -m "$MODEL" --cache-type-k "$config" --cache-type-v f16 -ngl $NGL \
            -p "$prompt" -n $N_PREDICT --no-warmup --seed 42 \
            > "$tmpfile" 2>&1 &

        sleep $WAIT

        # Extract the generated text (between prompt echo and timing line)
        output=$(grep -A5 "^> " "$tmpfile" | grep "^|" | sed 's/^|- //' | sed 's/^| //' | head -3)
        timing=$(grep "Prompt:" "$tmpfile" | head -1)

        echo "  [$config]: $output  $timing"
        pkill -f "llama-cli" 2>/dev/null
    done
done

echo ""
echo "=== Done ==="
