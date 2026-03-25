IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

# TurboQuant — llama.cpp KV Cache Quantization

## What This Is
Custom quantization types (TBQ3_0, TBQ4_0) for llama.cpp KV cache, using rotation-based quantization with centroid scaling. Targets lower memory usage during inference with minimal quality loss.

## Tech Stack
- C / Objective-C (ggml core)
- Metal shaders (GPU backend)
- CMake build system
- QK_TBQ=128 (block size matching full head_dim rotation)

## Build
```bash
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

## Key Files
- `ggml/src/ggml-common.h` — block structs, QK_TBQ constant
- `ggml/src/ggml-quants.c` — CPU quantize/dequantize
- `ggml/src/ggml-cpu/quants.c` — CPU vec_dot kernels
- `ggml/src/ggml-metal/tbq-metal.h` — Metal shader implementations
- `ggml/include/ggml.h` — type enum registration
- `tests/test-tbq-roundtrip.c` — round-trip quality test

## Quality Thresholds
- TBQ4_0 round-trip cosine similarity ≥ 0.99
- TBQ3_0 round-trip cosine similarity ≥ 0.97
- Centroid scale = 1/sqrt(QK_TBQ) ≈ 0.08839

---

## Agentic Harness (Generator/Evaluator Loop)

This repo includes a reusable agentic harness for automated work + verification loops.

### How It Works
1. **Generator** — A Claude Code session does the work, writes `HANDOFF.md` describing what was done and how to verify
2. **Evaluator** — A fresh Claude Code session runs verification per `EVAL.md`, writes results to `EVAL_RESULTS.md`
3. **Loop** — `harness.sh` parses STATUS from `EVAL_RESULTS.md` and loops until PASS or NEEDS_HUMAN
4. **Safety** — Stops after 3 consecutive identical failures to avoid infinite loops

### Files
| File | Purpose |
|------|---------|
| `harness.sh` | Orchestration script — run this |
| `EVAL.md` | Evaluator mandate (project-specific checks) |
| `HANDOFF.md` | Generator → Evaluator handoff (filled each turn) |
| `EVAL_RESULTS.md` | Evaluator results (STATUS on line 1) |
| `harness.log` | Timestamped log of all turns |

### Usage
```bash
./harness.sh "Fix the TBQ4_0 dequantization to handle edge case with zero blocks"
```

### Environment Variables
- `MAX_TURNS` — max generator/evaluator cycles (default: 10)
- `CLAUDE_MODEL` — model override (e.g., `sonnet`)

### Adapting to Other Projects
The harness is project-agnostic. To use in another repo:
1. Copy `harness.sh`, `HANDOFF.md`, `EVAL_RESULTS.md` templates
2. Write a new `EVAL.md` with project-specific verification checks
3. Run `./harness.sh "your task"`
