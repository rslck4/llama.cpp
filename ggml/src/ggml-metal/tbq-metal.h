// TurboQuant Metal shader functions
// Implements quantize/dequantize for TBQ3_0 and TBQ4_0 types
// Uses randomized Walsh-Hadamard transform per arxiv 2504.19874

#ifndef TBQ_METAL_H
#define TBQ_METAL_H

// Lloyd-Max centroids and boundaries for unit-variance Gaussian
// Pre-divided by sqrt(128) for QK_TBQ=128

constant float tbq4_centroids_scaled[16] = {
    -0.2366863173f, -0.1803299069f, -0.1414832281f, -0.1100434928f,
    -0.0826784604f, -0.0576733968f, -0.0341090634f, -0.0112871920f,
     0.0112871920f,  0.0341090634f,  0.0576733968f,  0.0826784604f,
     0.1100434928f,  0.1414832281f,  0.1803299069f,  0.2366863173f,
};

constant float tbq4_boundaries_scaled[15] = {
    -0.2085081121f, -0.1609021481f, -0.1257677799f, -0.0963609766f,
    -0.0701803480f, -0.0458912301f, -0.0226981277f,  0.0000000000f,
     0.0226981277f,  0.0458912301f,  0.0701803480f,  0.0963609766f,
     0.1257677799f,  0.1609021481f,  0.2085081121f,
};

constant float tbq3_centroids_scaled[8] = {
    -0.1881080815f, -0.1180603160f, -0.0665210704f, -0.0215932733f,
     0.0215932733f,  0.0665210704f,  0.1180603160f,  0.1881080815f,
};

constant float tbq3_boundaries_scaled[7] = {
    -0.1530886181f, -0.0922951126f, -0.0440527525f,  0.0000000000f,
     0.0440527525f,  0.0922951126f,  0.1530886181f,
};

// Random sign vector for d=128, seed = 0x54425131 ^ 128
// Generated deterministically from tbq_seed_for_dim(128)
constant int8_t tbq_signs_128[128] = {
    -1, 1, 1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1,-1, 1,-1,
    -1,-1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1,-1, 1, 1,
    -1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1, 1, 1, 1,-1,-1,
     1,-1,-1,-1,-1, 1,-1,-1, 1, 1,-1,-1,-1, 1, 1, 1,
     1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1, 1, 1, 1, 1,
     1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1,-1, 1,-1,
    -1,-1, 1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,
     1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
};

// --- Walsh-Hadamard Transform (in-place, unnormalized) ---
// d must be a power of 2
inline void tbq_wht_inplace(thread float * x, int d) {
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

// Forward transform: y = (1/sqrt(d)) * H * diag(signs) * x
inline void tbq_forward_transform(thread const float * x, thread float * y, int d) {
    float inv_sqrt_d = 1.0f / sqrt((float)d);
    for (int i = 0; i < d; i++) {
        y[i] = x[i] * tbq_signs_128[i];
    }
    tbq_wht_inplace(y, d);
    for (int i = 0; i < d; i++) {
        y[i] *= inv_sqrt_d;
    }
}

// Inverse transform: x = diag(signs) * H * (1/sqrt(d)) * y
inline void tbq_inverse_transform(thread const float * y, thread float * x, int d) {
    float inv_sqrt_d = 1.0f / sqrt((float)d);
    for (int i = 0; i < d; i++) {
        x[i] = y[i];
    }
    tbq_wht_inplace(x, d);
    for (int i = 0; i < d; i++) {
        x[i] *= inv_sqrt_d * tbq_signs_128[i];
    }
}

// Device memory variant for MUL_MV kernels (reads from device, writes to thread)
inline void tbq_forward_transform_device(device const float * x, thread float * y, int d) {
    float inv_sqrt_d = 1.0f / sqrt((float)d);
    for (int i = 0; i < d; i++) {
        y[i] = x[i] * tbq_signs_128[i];
    }
    tbq_wht_inplace(y, d);
    for (int i = 0; i < d; i++) {
        y[i] *= inv_sqrt_d;
    }
}

// --- TBQ4 quantize (for SET_ROWS kernel) ---
// Quantizes QK_TBQ=128 floats into one block_tbq4_0
void quantize_tbq4_0(device const float * src, device block_tbq4_0 & dst) {
#pragma METAL fp math_mode(safe)
    // 1. Compute L2 norm
    float norm = 0.0f;
    for (int i = 0; i < QK_TBQ; i++) {
        norm += src[i] * src[i];
    }
    norm = sqrt(norm);

    if (norm < 1e-10f) {
        dst.d = 0.0h;
        for (int i = 0; i < QK_TBQ/2; i++) dst.qs[i] = 0;
        return;
    }

    float inv_norm = 1.0f / norm;

    // 2. Normalize and apply randomized Hadamard transform
    float normalized[QK_TBQ];
    float rotated[QK_TBQ];
    for (int i = 0; i < QK_TBQ; i++) {
        normalized[i] = src[i] * inv_norm;
    }
    tbq_forward_transform(normalized, rotated, QK_TBQ);

    // 3. Quantize to 4-bit indices and pack, track for norm correction
    uint8_t idx_all[QK_TBQ];
    for (int i = 0; i < QK_TBQ/2; i++) {
        uint8_t idx0 = 0, idx1 = 0;
        for (int b = 0; b < 15; b++) {
            if (rotated[2*i]     > tbq4_boundaries_scaled[b]) idx0 = b + 1;
            if (rotated[2*i + 1] > tbq4_boundaries_scaled[b]) idx1 = b + 1;
        }
        idx_all[2*i]     = idx0;
        idx_all[2*i + 1] = idx1;
        dst.qs[i] = idx0 | (idx1 << 4);
    }

    // 4. Norm correction: minimize MSE
    float dot_rc = 0.0f, dot_cc = 0.0f;
    for (int i = 0; i < QK_TBQ; i++) {
        float c = tbq4_centroids_scaled[idx_all[i]];
        dot_rc += rotated[i] * c;
        dot_cc += c * c;
    }
    float corrected_norm = (dot_cc > 1e-20f) ? norm * (dot_rc / dot_cc) : norm;
    dst.d = half(corrected_norm);
}

// --- TBQ3 quantize (for SET_ROWS kernel) ---
// Quantizes QK_TBQ=128 floats into one block_tbq3_0
void quantize_tbq3_0(device const float * src, device block_tbq3_0 & dst) {
#pragma METAL fp math_mode(safe)
    float norm = 0.0f;
    for (int i = 0; i < QK_TBQ; i++) {
        norm += src[i] * src[i];
    }
    norm = sqrt(norm);

    if (norm < 1e-10f) {
        dst.d = 0.0h;
        for (int i = 0; i < QK_TBQ * 3 / 8; i++) dst.qs[i] = 0;
        return;
    }

    float inv_norm = 1.0f / norm;

    // Normalize and apply Hadamard transform
    float normalized[QK_TBQ];
    float rotated[QK_TBQ];
    for (int i = 0; i < QK_TBQ; i++) {
        normalized[i] = src[i] * inv_norm;
    }
    tbq_forward_transform(normalized, rotated, QK_TBQ);

    // Quantize to 3-bit indices
    uint8_t indices[QK_TBQ];
    for (int i = 0; i < QK_TBQ; i++) {
        uint8_t idx = 0;
        for (int b = 0; b < 7; b++) {
            if (rotated[i] > tbq3_boundaries_scaled[b]) idx = b + 1;
        }
        indices[i] = idx;
    }

    // Norm correction
    float dot_rc = 0.0f, dot_cc = 0.0f;
    for (int i = 0; i < QK_TBQ; i++) {
        float c = tbq3_centroids_scaled[indices[i]];
        dot_rc += rotated[i] * c;
        dot_cc += c * c;
    }
    float corrected_norm = (dot_cc > 1e-20f) ? norm * (dot_rc / dot_cc) : norm;
    dst.d = half(corrected_norm);

    // Pack 3-bit indices: 8 values -> 3 bytes (24 bits)
    for (int g = 0; g < QK_TBQ / 8; g++) {
        uint32_t packed = 0;
        for (int j = 0; j < 8; j++) {
            packed |= ((uint32_t)(indices[g*8 + j] & 7)) << (j * 3);
        }
        dst.qs[g*3 + 0] = (uint8_t)(packed & 0xFF);
        dst.qs[g*3 + 1] = (uint8_t)((packed >> 8) & 0xFF);
        dst.qs[g*3 + 2] = (uint8_t)((packed >> 16) & 0xFF);
    }
}

// --- TBQ4 dequantize (type4x4 interface for GET_ROWS) ---
// Produces 16 floats from a 128-element block. il selects which 16 (0..7).
// Must dequantize full block due to transform coupling.
template <typename type4x4>
void dequantize_tbq4_0(device const block_tbq4_0 * xb, short il, thread type4x4 & reg) {
    float norm = float(xb->d);

    float4x4 reg_f;

    if (abs(norm) < 1e-10f) {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                reg_f[i][j] = 0.0f;
        reg = (type4x4) reg_f;
        return;
    }

    // Unpack all 128 nibbles to centroids
    float rotated[QK_TBQ];
    for (int i = 0; i < QK_TBQ/2; i++) {
        rotated[2*i    ] = tbq4_centroids_scaled[xb->qs[i] & 0x0F];
        rotated[2*i + 1] = tbq4_centroids_scaled[xb->qs[i] >> 4];
    }

    // Inverse Hadamard transform for all 128 elements
    float unrotated[QK_TBQ];
    tbq_inverse_transform(rotated, unrotated, QK_TBQ);

    // Select the 16 elements for this il
    int base = il * 16;
    for (int k = 0; k < 16; k++) {
        reg_f[k/4][k%4] = unrotated[base + k] * norm;
    }

    reg = (type4x4) reg_f;
}

// --- TBQ3 dequantize (type4x4 interface for GET_ROWS) ---
template <typename type4x4>
void dequantize_tbq3_0(device const block_tbq3_0 * xb, short il, thread type4x4 & reg) {
    float norm = float(xb->d);

    float4x4 reg_f;

    if (abs(norm) < 1e-10f) {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                reg_f[i][j] = 0.0f;
        reg = (type4x4) reg_f;
        return;
    }

    // Unpack 3-bit indices and map to centroids
    float rotated[QK_TBQ];
    for (int g = 0; g < QK_TBQ / 8; g++) {
        uint32_t packed = (uint32_t)xb->qs[g*3 + 0]
                        | ((uint32_t)xb->qs[g*3 + 1] << 8)
                        | ((uint32_t)xb->qs[g*3 + 2] << 16);
        for (int j = 0; j < 8; j++) {
            uint8_t idx = (packed >> (j * 3)) & 7;
            rotated[g*8 + j] = tbq3_centroids_scaled[idx];
        }
    }

    // Inverse Hadamard transform
    float unrotated[QK_TBQ];
    tbq_inverse_transform(rotated, unrotated, QK_TBQ);

    // Select the 16 elements for this il
    int base = il * 16;
    for (int k = 0; k < 16; k++) {
        reg_f[k/4][k%4] = unrotated[base + k] * norm;
    }

    reg = (type4x4) reg_f;
}

// --- TBQ4 dequantize (type4 interface for flash_attn_ext_vec) ---
// Produces 4 floats from a 128-element block. il selects which 4 (0..31).
// Fallback: each thread does full 128-element inverse WHT independently.
// Used for NE>1 instantiations where il != tiisg.
template <typename type4>
void dequantize_tbq4_0_t4(device const block_tbq4_0 * xb, short il, thread type4 & reg) {
    float norm = float(xb->d);

    if (abs(norm) < 1e-10f) {
        for (int i = 0; i < 4; i++) reg[i] = 0.0f;
        return;
    }

    // Unpack all 128 nibbles to centroids
    float rotated[QK_TBQ];
    for (int i = 0; i < QK_TBQ/2; i++) {
        rotated[2*i    ] = tbq4_centroids_scaled[xb->qs[i] & 0x0F];
        rotated[2*i + 1] = tbq4_centroids_scaled[xb->qs[i] >> 4];
    }

    // Inverse Hadamard transform
    float unrotated[QK_TBQ];
    tbq_inverse_transform(rotated, unrotated, QK_TBQ);

    // Select the 4 elements for this il
    int base = il * 4;
    for (int i = 0; i < 4; i++) {
        reg[i] = unrotated[base + i] * norm;
    }
}

// --- TBQ3 dequantize (type4 interface for flash_attn_ext_vec) ---
// Fallback: each thread does full 128-element inverse WHT independently.
template <typename type4>
void dequantize_tbq3_0_t4(device const block_tbq3_0 * xb, short il, thread type4 & reg) {
    float norm = float(xb->d);

    if (abs(norm) < 1e-10f) {
        for (int i = 0; i < 4; i++) reg[i] = 0.0f;
        return;
    }

    // Unpack 3-bit indices and map to centroids
    float rotated[QK_TBQ];
    for (int g = 0; g < QK_TBQ / 8; g++) {
        uint32_t packed = (uint32_t)xb->qs[g*3 + 0]
                        | ((uint32_t)xb->qs[g*3 + 1] << 8)
                        | ((uint32_t)xb->qs[g*3 + 2] << 16);
        for (int j = 0; j < 8; j++) {
            uint8_t idx = (packed >> (j * 3)) & 7;
            rotated[g*8 + j] = tbq3_centroids_scaled[idx];
        }
    }

    // Inverse Hadamard transform
    float unrotated[QK_TBQ];
    tbq_inverse_transform(rotated, unrotated, QK_TBQ);

    // Select the 4 elements for this il
    int base = il * 4;
    for (int i = 0; i < 4; i++) {
        reg[i] = unrotated[base + i] * norm;
    }
}

// --- SIMD-cooperative inverse WHT ---
// 32 threads cooperatively compute a 128-element WHT using simd_shuffle.
// Each thread holds 4 contiguous elements: thread t holds [4t..4t+3].
// REQUIREMENT: il must equal tiisg (guaranteed for NE=1 flash_attn_ext_vec).
inline void tbq_inverse_wht_simd(thread float * x, short il) {
    // Stage len=1: butterfly between adjacent pairs (intra-thread)
    {
        float u0 = x[0], v0 = x[1];
        x[0] = u0 + v0; x[1] = u0 - v0;
        float u1 = x[2], v1 = x[3];
        x[2] = u1 + v1; x[3] = u1 - v1;
    }

    // Stage len=2: butterfly between elements 2 apart (intra-thread)
    {
        float u0 = x[0], v0 = x[2];
        x[0] = u0 + v0; x[2] = u0 - v0;
        float u1 = x[1], v1 = x[3];
        x[1] = u1 + v1; x[3] = u1 - v1;
    }

    // Stages len=4,8,16,32,64: cross-thread butterflies via simd_shuffle
    // At each stage, thread pairs at distance 'stride' exchange and combine.
    // stride = len/4 because each thread holds 4 elements.
    for (short stride = 1; stride <= 16; stride <<= 1) {
        ushort partner = il ^ stride;
        float p0 = simd_shuffle(x[0], partner);
        float p1 = simd_shuffle(x[1], partner);
        float p2 = simd_shuffle(x[2], partner);
        float p3 = simd_shuffle(x[3], partner);

        if ((il & stride) == 0) {
            // Lower half of butterfly: x[i+j] = u + v
            x[0] += p0; x[1] += p1; x[2] += p2; x[3] += p3;
        } else {
            // Upper half of butterfly: x[i+j+len] = u - v
            x[0] = p0 - x[0]; x[1] = p1 - x[1];
            x[2] = p2 - x[2]; x[3] = p3 - x[3];
        }
    }
}

// --- TBQ4 dequantize (SIMD-cooperative type4 for flash_attn_ext_vec NE=1) ---
// All 32 simdgroup threads must call this on the SAME block simultaneously
// with il == tiisg. Computes one cooperative WHT instead of 32 independent ones.
template <typename type4>
void dequantize_tbq4_0_t4_simd(device const block_tbq4_0 * xb, short il, thread type4 & reg) {
    float norm = float(xb->d);

    if (abs(norm) < 1e-10f) {
        for (int i = 0; i < 4; i++) reg[i] = 0.0f;
        return;
    }

    // Each thread unpacks only its own 4 centroids (not all 128)
    uint8_t qb0 = xb->qs[il * 2];
    uint8_t qb1 = xb->qs[il * 2 + 1];
    float x[4] = {
        tbq4_centroids_scaled[qb0 & 0x0F],
        tbq4_centroids_scaled[qb0 >> 4],
        tbq4_centroids_scaled[qb1 & 0x0F],
        tbq4_centroids_scaled[qb1 >> 4],
    };

    // Cooperative inverse WHT across all 32 simdgroup threads
    tbq_inverse_wht_simd(x, il);

    // Apply 1/sqrt(d) scaling, sign flip, and norm
    int base = il * 4;
    const float inv_sqrt_d = 1.0f / 11.313708498984761f; // 1/sqrt(128)
    for (int i = 0; i < 4; i++) {
        reg[i] = x[i] * inv_sqrt_d * tbq_signs_128[base + i] * norm;
    }
}

// --- TBQ3 dequantize (SIMD-cooperative type4 for flash_attn_ext_vec NE=1) ---
template <typename type4>
void dequantize_tbq3_0_t4_simd(device const block_tbq3_0 * xb, short il, thread type4 & reg) {
    float norm = float(xb->d);

    if (abs(norm) < 1e-10f) {
        for (int i = 0; i < 4; i++) reg[i] = 0.0f;
        return;
    }

    // Each thread unpacks only its own 4 centroids from 3-bit packing
    int g = il / 2;
    int pos_ofs = (il % 2) * 4;
    uint32_t packed = (uint32_t)xb->qs[g*3 + 0]
                    | ((uint32_t)xb->qs[g*3 + 1] << 8)
                    | ((uint32_t)xb->qs[g*3 + 2] << 16);
    float x[4];
    for (int j = 0; j < 4; j++) {
        uint8_t idx = (packed >> ((pos_ofs + j) * 3)) & 7;
        x[j] = tbq3_centroids_scaled[idx];
    }

    // Cooperative inverse WHT across all 32 simdgroup threads
    tbq_inverse_wht_simd(x, il);

    // Apply 1/sqrt(d) scaling, sign flip, and norm
    int base = il * 4;
    const float inv_sqrt_d = 1.0f / 11.313708498984761f; // 1/sqrt(128)
    for (int i = 0; i < 4; i++) {
        reg[i] = x[i] * inv_sqrt_d * tbq_signs_128[base + i] * norm;
    }
}

// --- TBQ4 MUL_MV kernel ---
// Computes dot product of quantized TBQ4 row with F32 vector.
// Optimization: transform the y vector, then dot in transform domain.
// dot = norm * sum_i centroid[i] * (H * diag(signs) * y / sqrt(d))[i]
template<short NR0, typename args_t>
void mul_mv_tbq4_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const int nb = args.ne00 / QK_TBQ;
    const int r0 = (tgpig.x * N_SG_TBQ4_0 + sgitg) * NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    const uint64_t offset1 = r1*args.nb11 + i12*args.nb12 + i13*args.nb13;
    device const float * y = (device const float *)(src1 + offset1);

    float sumf[NR0] = {0.0f};

    for (int row = 0; row < NR0; ++row) {
        if (r0 + row >= args.ne01) break;

        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
        device const block_tbq4_0 * x = (device const block_tbq4_0 *)(src0 + offset0);

        float sum = 0.0f;

        for (int bi = 0; bi < nb; bi++) {
            float norm_val = float(x[bi].d);
            if (abs(norm_val) < 1e-10f) continue;

            device const float * yb = y + bi * QK_TBQ;

            // Transform y block using Hadamard
            float qy[QK_TBQ];
            tbq_forward_transform_device(yb, qy, QK_TBQ);

            // Each thread handles 4 elements in transform domain
            int i0 = tiisg * 4;

            // Unpack centroids for these 4 elements
            uint8_t qb0 = x[bi].qs[tiisg * 2];
            uint8_t qb1 = x[bi].qs[tiisg * 2 + 1];
            float c0 = tbq4_centroids_scaled[qb0 & 0x0F];
            float c1 = tbq4_centroids_scaled[qb0 >> 4];
            float c2 = tbq4_centroids_scaled[qb1 & 0x0F];
            float c3 = tbq4_centroids_scaled[qb1 >> 4];

            sum += norm_val * (c0 * qy[i0] + c1 * qy[i0+1] + c2 * qy[i0+2] + c3 * qy[i0+3]);
        }

        sumf[row] = metal::simd_sum(sum);
    }

    device float * dst_f32 = (device float *) dst + im*args.ne0*args.ne1 + r1*args.ne0;

    for (int row = 0; row < NR0; ++row) {
        if (tiisg == 0 && r0 + row < args.ne01) {
            dst_f32[r0 + row] = sumf[row];
        }
    }
}

// --- TBQ3 MUL_MV kernel ---
template<short NR0, typename args_t>
void mul_mv_tbq3_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const int nb = args.ne00 / QK_TBQ;
    const int r0 = (tgpig.x * N_SG_TBQ3_0 + sgitg) * NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    const uint64_t offset1 = r1*args.nb11 + i12*args.nb12 + i13*args.nb13;
    device const float * y = (device const float *)(src1 + offset1);

    float sumf[NR0] = {0.0f};

    for (int row = 0; row < NR0; ++row) {
        if (r0 + row >= args.ne01) break;

        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
        device const block_tbq3_0 * x = (device const block_tbq3_0 *)(src0 + offset0);

        float sum = 0.0f;

        for (int bi = 0; bi < nb; bi++) {
            float norm_val = float(x[bi].d);
            if (abs(norm_val) < 1e-10f) continue;

            device const float * yb = y + bi * QK_TBQ;

            // Transform y block using Hadamard
            float qy[QK_TBQ];
            tbq_forward_transform_device(yb, qy, QK_TBQ);

            // Each thread handles 4 elements in transform domain
            int i0 = tiisg * 4;

            // Unpack 3-bit centroids for elements i0..i0+3
            float c[4];
            for (int e = 0; e < 4; e++) {
                int idx = i0 + e;
                int g = idx / 8;
                uint32_t packed = (uint32_t)x[bi].qs[g*3 + 0]
                                | ((uint32_t)x[bi].qs[g*3 + 1] << 8)
                                | ((uint32_t)x[bi].qs[g*3 + 2] << 16);
                c[e] = tbq3_centroids_scaled[(packed >> ((idx % 8) * 3)) & 7];
            }

            sum += norm_val * (c[0] * qy[i0] + c[1] * qy[i0+1] + c[2] * qy[i0+2] + c[3] * qy[i0+3]);
        }

        sumf[row] = metal::simd_sum(sum);
    }

    device float * dst_f32 = (device float *) dst + im*args.ne0*args.ne1 + r1*args.ne0;

    for (int row = 0; row < NR0; ++row) {
        if (tiisg == 0 && r0 + row < args.ne01) {
            dst_f32[r0 + row] = sumf[row];
        }
    }
}

#endif // TBQ_METAL_H
