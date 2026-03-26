#!/bin/bash
# WHT Quality Test: 5-prompt comparison across KV cache configurations
# Tests: baseline (f16/f16), TBQ K-only (tbq/f16), TBQ K+V with flash attn (tbq/tbq -fa on)
# Model: qwen2.5-3b-instruct Q4_K_M (head_dim=128, matches QK_TBQ)

set -euo pipefail

MODEL="${1:-./models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf}"
CLI="./build/bin/llama-completion"
RESULTS_DIR="./tests/wht-quality-results"
SEED=42
N_PREDICT=128
TEMP=0

mkdir -p "$RESULTS_DIR"

# 5 test prompts
PROMPTS=(
    "Explain the concept of entropy in thermodynamics in simple terms."
    "Write a Python function that checks if a string is a palindrome."
    "What are the main differences between TCP and UDP protocols?"
    "Translate the following to French: The weather is beautiful today and I would like to go for a walk in the park."
    "Solve step by step: If a train travels at 60 mph for 2.5 hours, how far does it travel?"
)

PROMPT_NAMES=(
    "science_explanation"
    "code_generation"
    "technical_comparison"
    "translation"
    "math_reasoning"
)

# Config name, -ctk, -ctv, extra flags
CONFIG_NAMES=("f16_baseline" "tbq4_k_only" "tbq3_k_only" "tbq4_kv_fa" "tbq3_kv_fa")
CONFIG_CTK=("f16" "tbq4_0" "tbq3_0" "tbq4_0" "tbq3_0")
CONFIG_CTV=("f16" "f16" "f16" "tbq4_0" "tbq3_0")
CONFIG_EXTRA=("" "" "" "-fa on" "-fa on")

echo "============================================"
echo "WHT Quality Test — $(date)"
echo "Model: $MODEL"
echo "Seed: $SEED, N_Predict: $N_PREDICT, Temp: $TEMP"
echo "Configs: ${CONFIG_NAMES[*]}"
echo "============================================"
echo ""

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found at $MODEL"
    exit 1
fi

# Run all combinations
for ci in "${!CONFIG_NAMES[@]}"; do
    cname="${CONFIG_NAMES[$ci]}"
    ctk="${CONFIG_CTK[$ci]}"
    ctv="${CONFIG_CTV[$ci]}"
    extra="${CONFIG_EXTRA[$ci]}"
    echo "--- Config: $cname (K=$ctk, V=$ctv${extra:+ $extra}) ---"

    for pi in "${!PROMPTS[@]}"; do
        prompt="${PROMPTS[$pi]}"
        pname="${PROMPT_NAMES[$pi]}"
        outfile="$RESULTS_DIR/${pname}_${cname}.txt"
        timefile="$RESULTS_DIR/${pname}_${cname}.time"

        echo -n "  Prompt $((pi+1))/5: $pname ... "

        START=$(date +%s)
        # shellcheck disable=SC2086
        "$CLI" \
            -m "$MODEL" \
            -p "$prompt" \
            -n "$N_PREDICT" \
            -s "$SEED" \
            --temp "$TEMP" \
            -ctk "$ctk" \
            -ctv "$ctv" \
            $extra \
            --no-display-prompt \
            -ngl 99 \
            2>"$timefile" > "$outfile" || {
                echo "FAILED (exit code $?)"
                echo "FAILED" > "$outfile"
                continue
            }
        END=$(date +%s)
        ELAPSED=$((END - START))

        words=$(wc -w < "$outfile" | tr -d ' ')
        echo "done (${ELAPSED}s, ${words} words)"
    done
    echo ""
done

# Generate comparison report
REPORT="$RESULTS_DIR/comparison_report.md"
{
    echo "# WHT Quality Comparison Report"
    echo ""
    echo "**Date:** $(date)"
    echo "**Model:** $(basename "$MODEL")"
    echo "**Settings:** seed=$SEED, n_predict=$N_PREDICT, temp=$TEMP"
    echo ""
    echo "**Configurations tested:**"
    echo "| Config | K Cache | V Cache | Flash Attn |"
    echo "|--------|---------|---------|------------|"
    echo "| f16_baseline | f16 | f16 | auto |"
    echo "| tbq4_k_only | tbq4_0 | f16 | auto |"
    echo "| tbq3_k_only | tbq3_0 | f16 | auto |"
    echo "| tbq4_kv_fa | tbq4_0 | tbq4_0 | on |"
    echo "| tbq3_kv_fa | tbq3_0 | tbq3_0 | on |"
    echo ""
} > "$REPORT"

for pi in "${!PROMPT_NAMES[@]}"; do
    pname="${PROMPT_NAMES[$pi]}"
    prompt="${PROMPTS[$pi]}"

    echo "## Prompt $((pi+1)): $pname" >> "$REPORT"
    echo "" >> "$REPORT"
    echo "> $prompt" >> "$REPORT"
    echo "" >> "$REPORT"

    for cname in "${CONFIG_NAMES[@]}"; do
        outfile="$RESULTS_DIR/${pname}_${cname}.txt"
        echo "### $cname" >> "$REPORT"
        echo '```' >> "$REPORT"
        if [ -f "$outfile" ]; then
            cat "$outfile" >> "$REPORT"
        else
            echo "(no output)" >> "$REPORT"
        fi
        echo '```' >> "$REPORT"
        echo "" >> "$REPORT"
    done

    # Diff analysis against baseline
    baseline="$RESULTS_DIR/${pname}_f16_baseline.txt"
    for cname in "tbq4_k_only" "tbq3_k_only" "tbq4_kv_fa" "tbq3_kv_fa"; do
        outfile="$RESULTS_DIR/${pname}_${cname}.txt"
        if [ -f "$baseline" ] && [ -f "$outfile" ] && ! grep -q "FAILED" "$outfile" 2>/dev/null; then
            if diff -q "$baseline" "$outfile" > /dev/null 2>&1; then
                echo "**${cname} vs baseline:** IDENTICAL" >> "$REPORT"
            else
                wdiff=$(diff <(tr ' ' '\n' < "$baseline") <(tr ' ' '\n' < "$outfile") | grep -c '^[<>]' || true)
                total=$(wc -w < "$baseline" | tr -d ' ')
                echo "**${cname} vs baseline:** $wdiff word-level diffs out of ~$total words" >> "$REPORT"
            fi
        elif grep -q "FAILED" "$outfile" 2>/dev/null; then
            echo "**${cname} vs baseline:** FAILED" >> "$REPORT"
        fi
    done
    echo "" >> "$REPORT"
done

# Summary table
{
    echo "## Summary"
    echo ""
    echo "| Prompt | tbq4_k_only | tbq3_k_only | tbq4_kv_fa | tbq3_kv_fa |"
    echo "|--------|-------------|-------------|------------|------------|"
} >> "$REPORT"

for pi in "${!PROMPT_NAMES[@]}"; do
    pname="${PROMPT_NAMES[$pi]}"
    baseline="$RESULTS_DIR/${pname}_f16_baseline.txt"
    row="| ${pname} |"

    for cname in "tbq4_k_only" "tbq3_k_only" "tbq4_kv_fa" "tbq3_kv_fa"; do
        outfile="$RESULTS_DIR/${pname}_${cname}.txt"
        if grep -q "FAILED" "$outfile" 2>/dev/null; then
            row+=" FAIL |"
        elif diff -q "$baseline" "$outfile" > /dev/null 2>&1; then
            row+=" IDENTICAL |"
        else
            wdiff=$(diff <(tr ' ' '\n' < "$baseline") <(tr ' ' '\n' < "$outfile") | grep -c '^[<>]' || true)
            row+=" ${wdiff} diffs |"
        fi
    done
    echo "$row" >> "$REPORT"
done

echo "" >> "$REPORT"

# Overall pass/fail assessment
echo "### Quality Assessment" >> "$REPORT"
echo "" >> "$REPORT"

any_fail=false
for cname in "${CONFIG_NAMES[@]}"; do
    [ "$cname" = "f16_baseline" ] && continue
    for pname in "${PROMPT_NAMES[@]}"; do
        outfile="$RESULTS_DIR/${pname}_${cname}.txt"
        if grep -q "FAILED" "$outfile" 2>/dev/null; then
            any_fail=true
        fi
    done
done

if $any_fail; then
    echo "**RESULT: PARTIAL FAIL** — Some configurations failed to produce output." >> "$REPORT"
    echo "" >> "$REPORT"
    echo "Note: V-cache quantization without flash attention is expected to fail (see T05)." >> "$REPORT"
else
    echo "**RESULT: PASS** — All configurations produced output successfully." >> "$REPORT"
fi
echo "" >> "$REPORT"

echo "============================================"
echo "DONE — Report: $REPORT"
echo "============================================"
echo ""
cat "$REPORT"
