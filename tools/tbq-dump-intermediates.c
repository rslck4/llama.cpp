// Dump intermediate TBQ quantization state for cross-validation
// Build: cc -O2 -I../ggml/include -I../ggml/src tbq-dump-intermediates.c -L../build/bin -lggml-base -lm -o tbq-dump-intermediates
// Run: DYLD_LIBRARY_PATH=../build/bin ./tbq-dump-intermediates > intermediates.json

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

// Re-expose internal helpers we need to dump — these are static in ggml-quants.c,
// so we replicate the PRNG and sign generation here for dumping only.

static uint64_t dump_splitmix64(uint64_t * state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void dump_generate_signs(int d, uint64_t seed, int8_t * signs) {
    uint64_t state = seed;
    for (int i = 0; i < d; i += 64) {
        uint64_t bits = dump_splitmix64(&state);
        for (int j = 0; j < 64 && (i + j) < d; j++) {
            signs[i + j] = (bits & (1ULL << j)) ? 1 : -1;
        }
    }
}

static void dump_wht_inplace(float * x, int d) {
    for (int len = 1; len < d; len <<= 1) {
        for (int i = 0; i < d; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float u = x[i + j];
                float v = x[i + j + len];
                x[i + j]       = u + v;
                x[i + j + len] = u - v;
            }
        }
    }
}

static void print_float_array(const char * name, const float * arr, int n) {
    printf("  \"%s\": [", name);
    for (int i = 0; i < n; i++) {
        if (i > 0) printf(", ");
        printf("%.10g", arr[i]);
    }
    printf("]");
}

static void print_int_array(const char * name, const uint8_t * arr, int n) {
    printf("  \"%s\": [", name);
    for (int i = 0; i < n; i++) {
        if (i > 0) printf(", ");
        printf("%d", arr[i]);
    }
    printf("]");
}

static void print_sign_array(const char * name, const int8_t * arr, int n) {
    printf("  \"%s\": [", name);
    for (int i = 0; i < n; i++) {
        if (i > 0) printf(", ");
        printf("%d", arr[i]);
    }
    printf("]");
}

int main(void) {
    const int d = QK_TBQ;  // 128

    // Known deterministic input: same as roundtrip test
    float x[QK_TBQ];
    for (int i = 0; i < d; i++) {
        x[i] = sinf((float)i * 0.3f) * 1.5f + cosf((float)i * 0.7f);
    }

    // Compute sign vector (same as C++ internals)
    uint64_t seed = 0x54425131ULL ^ (uint64_t)d;
    int8_t signs[QK_TBQ];
    dump_generate_signs(d, seed, signs);

    // Compute norm
    float norm = 0.0f;
    for (int i = 0; i < d; i++) norm += x[i] * x[i];
    norm = sqrtf(norm);
    float inv_norm = 1.0f / norm;

    // Normalize
    float normalized[QK_TBQ];
    for (int i = 0; i < d; i++) normalized[i] = x[i] * inv_norm;

    // Forward transform (replicate exactly)
    float rotated[QK_TBQ];
    for (int i = 0; i < d; i++) rotated[i] = normalized[i] * signs[i];
    dump_wht_inplace(rotated, d);
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; i++) rotated[i] *= inv_sqrt_d;

    // --- TBQ4 quantization ---
    static const float tbq4_centroids_unit[16] = {
        -2.6778f, -2.0402f, -1.6007f, -1.2450f,
        -0.9354f, -0.6525f, -0.3859f, -0.1277f,
         0.1277f,  0.3859f,  0.6525f,  0.9354f,
         1.2450f,  1.6007f,  2.0402f,  2.6778f
    };
    static const float tbq4_boundaries_unit[15] = {
        -2.3590f, -1.8204f, -1.4229f, -1.0902f,
        -0.7940f, -0.5192f, -0.2568f, 0.0000f,
         0.2568f,  0.5192f,  0.7940f,  1.0902f,
         1.4229f,  1.8204f,  2.3590f
    };

    float boundaries4[15];
    for (int i = 0; i < 15; i++) boundaries4[i] = tbq4_boundaries_unit[i] * inv_sqrt_d;

    float centroids4[16];
    for (int i = 0; i < 16; i++) centroids4[i] = tbq4_centroids_unit[i] * inv_sqrt_d;

    uint8_t indices4[QK_TBQ];
    for (int i = 0; i < d; i++) {
        uint8_t idx = 0;
        for (int b = 0; b < 15; b++) {
            if (rotated[i] > boundaries4[b]) idx = b + 1;
        }
        indices4[i] = idx;
    }

    // Norm correction
    float dot_rc4 = 0.0f, dot_cc4 = 0.0f;
    for (int i = 0; i < d; i++) {
        float c = centroids4[indices4[i]];
        dot_rc4 += rotated[i] * c;
        dot_cc4 += c * c;
    }
    float corrected_norm4 = norm * (dot_rc4 / dot_cc4);

    // Also run via the actual library function for comparison
    block_tbq4_0 q4[1];
    quantize_row_tbq4_0_ref(x, q4, d);
    float lib_norm4 = GGML_FP16_TO_FP32(q4[0].d);

    // Unpack library indices for comparison
    uint8_t lib_indices4[QK_TBQ];
    for (int i = 0; i < d / 2; i++) {
        lib_indices4[2*i]     = q4[0].qs[i] & 0x0F;
        lib_indices4[2*i + 1] = q4[0].qs[i] >> 4;
    }

    // Dequantize via library
    float dequant4[QK_TBQ];
    dequantize_row_tbq4_0(q4, dequant4, d);

    // --- TBQ3 quantization ---
    static const float tbq3_centroids_unit[8] = {
        -2.1282f, -1.3357f, -0.7526f, -0.2443f,
         0.2443f,  0.7526f,  1.3357f,  2.1282f
    };
    static const float tbq3_boundaries_unit[7] = {
        -1.7320f, -1.0442f, -0.4984f, 0.0000f,
         0.4984f,  1.0442f,  1.7320f
    };

    float boundaries3[7];
    for (int i = 0; i < 7; i++) boundaries3[i] = tbq3_boundaries_unit[i] * inv_sqrt_d;

    float centroids3[8];
    for (int i = 0; i < 8; i++) centroids3[i] = tbq3_centroids_unit[i] * inv_sqrt_d;

    uint8_t indices3[QK_TBQ];
    for (int i = 0; i < d; i++) {
        uint8_t idx = 0;
        for (int b = 0; b < 7; b++) {
            if (rotated[i] > boundaries3[b]) idx = b + 1;
        }
        indices3[i] = idx;
    }

    float dot_rc3 = 0.0f, dot_cc3 = 0.0f;
    for (int i = 0; i < d; i++) {
        float c = centroids3[indices3[i]];
        dot_rc3 += rotated[i] * c;
        dot_cc3 += c * c;
    }
    float corrected_norm3 = norm * (dot_rc3 / dot_cc3);

    block_tbq3_0 q3[1];
    quantize_row_tbq3_0_ref(x, q3, d);
    float lib_norm3 = GGML_FP16_TO_FP32(q3[0].d);

    float dequant3[QK_TBQ];
    dequantize_row_tbq3_0(q3, dequant3, d);

    // Output as JSON
    printf("{\n");
    printf("  \"d\": %d,\n", d);
    printf("  \"seed\": %llu,\n", (unsigned long long)seed);
    printf("  \"input_norm\": %.10g,\n", norm);

    print_float_array("input", x, d); printf(",\n");
    print_sign_array("signs", signs, d); printf(",\n");
    print_float_array("normalized", normalized, d); printf(",\n");
    print_float_array("rotated", rotated, d); printf(",\n");

    // TBQ4
    printf("  \"tbq4\": {\n");
    printf("    \"corrected_norm\": %.10g,\n", corrected_norm4);
    printf("    \"lib_norm\": %.10g,\n", lib_norm4);
    printf("  ");
    print_int_array("indices", indices4, d); printf(",\n");
    printf("  ");
    print_int_array("lib_indices", lib_indices4, d); printf(",\n");
    printf("  ");
    print_float_array("dequantized", dequant4, d); printf("\n");
    printf("  },\n");

    // TBQ3
    printf("  \"tbq3\": {\n");
    printf("    \"corrected_norm\": %.10g,\n", corrected_norm3);
    printf("    \"lib_norm\": %.10g,\n", lib_norm3);
    printf("  ");
    print_int_array("indices", indices3, d); printf(",\n");
    printf("  ");
    print_float_array("dequantized", dequant3, d); printf("\n");
    printf("  }\n");

    printf("}\n");

    return 0;
}
