// Quick round-trip test for TurboQuant quantization
// Build: cc -O2 -I../ggml/include -I../ggml/src test-tbq-roundtrip.c -L../build/bin -lggml-base -lm -o test-tbq-roundtrip
// Run: DYLD_LIBRARY_PATH=../build/bin ./test-tbq-roundtrip

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

int main(void) {
    printf("QK_TBQ = %d\n", QK_TBQ);
    printf("sizeof(block_tbq3_0) = %zu\n", sizeof(block_tbq3_0));
    printf("sizeof(block_tbq4_0) = %zu\n", sizeof(block_tbq4_0));

    // Test dimensions matching Qwen 2.5 0.5B: n_embd_k_gqa = 128
    int dims[] = {64, 128, 256};
    int ndims = 3;

    for (int di = 0; di < ndims; di++) {
        int d = dims[di];
        printf("\n=== d=%d ===\n", d);
        if (d % QK_TBQ != 0) {
            printf("SKIP: d=%d not divisible by QK_TBQ=%d\n", d, QK_TBQ);
            continue;
        }
        int nb = d / QK_TBQ;
        printf("nb=%d blocks\n", nb);

        float * x = (float *)malloc(d * sizeof(float));
        float * y = (float *)calloc(d, sizeof(float));

        // Fill with a known pattern
        for (int i = 0; i < d; i++) {
            x[i] = sinf((float)i * 0.3f) * 1.5f + cosf((float)i * 0.7f);
        }

        // Compute input norm
        float input_norm = 0.0f;
        for (int i = 0; i < d; i++) input_norm += x[i] * x[i];
        input_norm = sqrtf(input_norm);
        printf("Input L2 norm: %.4f\n", input_norm);

        // --- TBQ4 ---
        block_tbq4_0 * q4 = (block_tbq4_0 *)malloc(nb * sizeof(block_tbq4_0));
        quantize_row_tbq4_0_ref(x, q4, d);

        float stored_norm = GGML_FP16_TO_FP32(q4[0].d);
        printf("TBQ4 stored norm (block 0): %.4f\n", stored_norm);
        for (int bi = 1; bi < nb; bi++) {
            float bn = GGML_FP16_TO_FP32(q4[bi].d);
            if (bn > 1e-6f) printf("  WARNING: block %d norm = %.4f (should be 0)\n", bi, bn);
        }

        dequantize_row_tbq4_0(q4, y, d);

        float mse4 = 0.0f, max_err4 = 0.0f;
        float output_norm = 0.0f;
        for (int i = 0; i < d; i++) {
            float diff = x[i] - y[i];
            mse4 += diff * diff;
            if (fabsf(diff) > max_err4) max_err4 = fabsf(diff);
            output_norm += y[i] * y[i];
        }
        mse4 /= d;
        output_norm = sqrtf(output_norm);
        printf("TBQ4: MSE=%.6f RMSE=%.6f MaxErr=%.6f OutputNorm=%.4f\n",
               mse4, sqrtf(mse4), max_err4, output_norm);

        // Cosine similarity
        float dot = 0.0f;
        for (int i = 0; i < d; i++) dot += x[i] * y[i];
        float cos_sim = dot / (input_norm * output_norm);
        printf("TBQ4: Cosine similarity: %.6f\n", cos_sim);

        printf("  First 5 original:  ");
        for (int i = 0; i < 5 && i < d; i++) printf("%.4f ", x[i]);
        printf("\n  First 5 decoded:   ");
        for (int i = 0; i < 5 && i < d; i++) printf("%.4f ", y[i]);
        printf("\n");

        // --- TBQ3 ---
        block_tbq3_0 * q3 = (block_tbq3_0 *)malloc(nb * sizeof(block_tbq3_0));
        quantize_row_tbq3_0_ref(x, q3, d);

        float * z = (float *)calloc(d, sizeof(float));
        dequantize_row_tbq3_0(q3, z, d);

        float mse3 = 0.0f, max_err3 = 0.0f;
        float output_norm3 = 0.0f;
        for (int i = 0; i < d; i++) {
            float diff = x[i] - z[i];
            mse3 += diff * diff;
            if (fabsf(diff) > max_err3) max_err3 = fabsf(diff);
            output_norm3 += z[i] * z[i];
        }
        mse3 /= d;
        output_norm3 = sqrtf(output_norm3);
        printf("TBQ3: MSE=%.6f RMSE=%.6f MaxErr=%.6f OutputNorm=%.4f\n",
               mse3, sqrtf(mse3), max_err3, output_norm3);

        dot = 0.0f;
        for (int i = 0; i < d; i++) dot += x[i] * z[i];
        cos_sim = dot / (input_norm * output_norm3);
        printf("TBQ3: Cosine similarity: %.6f\n", cos_sim);

        free(x); free(y); free(z); free(q4); free(q3);
    }

    printf("\nDone.\n");
    return 0;
}
