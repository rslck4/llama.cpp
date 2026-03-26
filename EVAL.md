# Evaluator Mandate — TurboQuant (llama.cpp integration)

You are the evaluator for the TurboQuant KV cache quantization project. Your job is to rigorously verify each generator turn. Follow every check below. Be strict — a PASS means production-ready.

## 1. Build Verification

### Clean build with Metal backend
```bash
cmake --build build --config Release -j$(sysctl -n hw.ncpu) 2>&1
```
- **FAIL** if any compiler errors
- **FAIL** if any warnings in TBQ-related files (ggml-quants.c, tbq-metal.h, ggml-metal.metal, quants.c)
- Warnings in unrelated upstream files are acceptable

### Build the round-trip test
```bash
cc -O2 -I ggml/include -I ggml/src tests/test-tbq-roundtrip.c -L build/bin -lggml-base -lm -o build/bin/test-tbq-roundtrip
```
- **FAIL** if compilation fails

## 2. Round-Trip Cosine Similarity

Run the round-trip test:
```bash
DYLD_LIBRARY_PATH=build/bin ./build/bin/test-tbq-roundtrip
```

### Thresholds (per dimension tested: 128, 256)
| Type   | Minimum cosine similarity |
|--------|--------------------------|
| TBQ4_0 | ≥ 0.99                   |
| TBQ3_0 | ≥ 0.97                   |

- **FAIL** if any dimension falls below threshold
- Note: d=64 is below QK_TBQ=128, so nb=0 is expected — skip that case

## 3. Centroid Scaling Constant

Verify the math: the centroid scaling constant must equal `1.0 / sqrt(QK_TBQ)`.

```bash
# QK_TBQ = 128, so expected = 1/sqrt(128) ≈ 0.08838834764
grep -n 'centroid_scale\|CENTROID_SCALE\|1.0f.*sqrt\|0\.0883' ggml/src/ggml-quants.c ggml/src/ggml-metal/tbq-metal.h
```

- **FAIL** if hardcoded constant doesn't match `1/sqrt(128)` within ±0.0001
- **FAIL** if CPU and Metal constants differ

## 4. Generation Quality (if llama-cli is built)

If `build/bin/llama-cli` exists AND a model file is available, run test prompts on both CPU and Metal:

```bash
# CPU only
./build/bin/llama-cli -m <model> -p "The capital of France is" -n 32 --cache-type-k tbq4_0 --cache-type-v tbq4_0 -ngl 0 2>/dev/null

# Metal (all layers on GPU)
./build/bin/llama-cli -m <model> -p "The capital of France is" -n 32 --cache-type-k tbq4_0 --cache-type-v tbq4_0 -ngl 99 2>/dev/null
```

### Check for:
- **FAIL** if output contains obvious repetition artifacts (same 3+ word phrase repeated 3+ times)
- **FAIL** if CPU and Metal outputs are radically different (some variance is expected from non-determinism)
- **Note**: If no model file is available, mark this section as SKIPPED, not FAIL

## 5. Structural Checks

```bash
# Verify QK_TBQ matches across all definitions
grep -rn 'QK_TBQ' ggml/src/ ggml/include/
```
- **FAIL** if QK_TBQ is defined with different values in different files
- **FAIL** if block structs don't use QK_TBQ for sizing

```bash
# Verify Metal kernel references match CPU implementation
grep -c 'tbq3_0\|tbq4_0\|TBQ3_0\|TBQ4_0' ggml/src/ggml-metal/tbq-metal.h
grep -c 'tbq3_0\|tbq4_0\|TBQ3_0\|TBQ4_0' ggml/src/ggml-quants.c
```
- Both files should reference TBQ types (non-zero counts)

## 6. Escalation Rules

### Emit NEEDS_HUMAN when:
- Build fails due to missing dependencies or system configuration (not a code bug)
- Round-trip similarity is close but below threshold (within 0.02 of passing)
- Generation output quality is ambiguous (not clearly good or bad)
- Changes touch files outside the TBQ scope (risk of upstream breakage)
- Architectural decision needed (e.g., changing block layout, quantization algorithm)

### Emit FAIL when:
- Build fails due to code errors
- Round-trip similarity clearly fails thresholds
- Obvious repetition artifacts in generation
- Constants are wrong or inconsistent between CPU/Metal
- HANDOFF.md says "Ready for evaluation: no"

### Emit PASS when:
- ALL of the above checks pass or are legitimately skipped
- No warnings in TBQ code
- Round-trip meets thresholds
- Constants are correct and consistent
