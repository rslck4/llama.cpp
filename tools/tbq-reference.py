#!/usr/bin/env python3
"""
TBQ Reference Implementation in Python for cross-validation against C++.

This is the "MLX PoC equivalent" — a pure-Python/NumPy implementation of the
TBQ quantization pipeline. It replicates every step of the C++ code:
  1. splitmix64 PRNG → sign vector
  2. Normalize → randomized WHT → boundary quantize → norm correction
  3. Dequantize: centroids → inverse WHT → rescale

Usage:
  python3 tbq-reference.py                    # standalone self-test
  python3 tbq-reference.py --compare FILE     # compare against C++ dump JSON
"""

import numpy as np
import json
import sys
import argparse

# ── Lloyd-Max codebook (identical to C++ ggml-quants.c) ──

TBQ3_CENTROIDS_UNIT = np.array([
    -2.1282, -1.3357, -0.7526, -0.2443,
     0.2443,  0.7526,  1.3357,  2.1282
], dtype=np.float64)

TBQ3_BOUNDARIES_UNIT = np.array([
    -1.7320, -1.0442, -0.4984, 0.0000,
     0.4984,  1.0442,  1.7320
], dtype=np.float64)

TBQ4_CENTROIDS_UNIT = np.array([
    -2.6778, -2.0402, -1.6007, -1.2450,
    -0.9354, -0.6525, -0.3859, -0.1277,
     0.1277,  0.3859,  0.6525,  0.9354,
     1.2450,  1.6007,  2.0402,  2.6778
], dtype=np.float64)

TBQ4_BOUNDARIES_UNIT = np.array([
    -2.3590, -1.8204, -1.4229, -1.0902,
    -0.7940, -0.5192, -0.2568, 0.0000,
     0.2568,  0.5192,  0.7940,  1.0902,
     1.4229,  1.8204,  2.3590
], dtype=np.float64)

QK_TBQ = 128

# ── PRNG: splitmix64 (must match C++ exactly) ──

def splitmix64(state: int) -> tuple[int, int]:
    """Returns (new_state, output). All arithmetic mod 2^64."""
    m = (1 << 64) - 1
    state = (state + 0x9e3779b97f4a7c15) & m
    z = state
    z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & m
    z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & m
    z = z ^ (z >> 31)
    return state, z


def generate_signs(d: int, seed: int) -> np.ndarray:
    """Generate deterministic sign vector {-1, +1}^d from seed."""
    signs = np.zeros(d, dtype=np.int8)
    state = seed
    for i in range(0, d, 64):
        state, bits = splitmix64(state)
        for j in range(min(64, d - i)):
            signs[i + j] = 1 if (bits & (1 << j)) else -1
    return signs


def seed_for_dim(d: int) -> int:
    return 0x54425131 ^ d


# ── Walsh-Hadamard Transform ──

def wht_inplace(x: np.ndarray) -> np.ndarray:
    """In-place Walsh-Hadamard Transform (unnormalized). d must be power of 2."""
    d = len(x)
    length = 1
    while length < d:
        for i in range(0, d, length << 1):
            for j in range(length):
                u = x[i + j]
                v = x[i + j + length]
                x[i + j] = u + v
                x[i + j + length] = u - v
        length <<= 1
    return x


def forward_transform(x: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """y = (1/sqrt(d)) * H * diag(signs) * x"""
    d = len(x)
    y = x * signs.astype(np.float64)
    y = wht_inplace(y)
    y *= 1.0 / np.sqrt(d)
    return y


def inverse_transform(y: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """x = diag(signs) * H * (1/sqrt(d)) * y"""
    d = len(y)
    x = y.copy()
    x = wht_inplace(x)
    x *= (1.0 / np.sqrt(d)) * signs.astype(np.float64)
    return x


# ── Quantization ──

def boundary_quantize(values: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Quantize each value by scanning boundaries (matches C++ linear scan)."""
    indices = np.zeros(len(values), dtype=np.uint8)
    for i, v in enumerate(values):
        idx = 0
        for b_idx, b in enumerate(boundaries):
            if v > b:
                idx = b_idx + 1
        indices[i] = idx
    return indices


def quantize_tbq4(x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Quantize a single block (128 floats) using TBQ4_0.
    Returns: (indices, corrected_norm, rotated)
    """
    d = len(x)
    assert d == QK_TBQ

    signs = generate_signs(d, seed_for_dim(d))
    inv_sqrt_d = 1.0 / np.sqrt(d)

    # Norm
    norm = np.sqrt(np.sum(x ** 2))
    if norm < 1e-10:
        return np.zeros(d, dtype=np.uint8), 0.0, np.zeros(d)

    # Normalize + forward transform
    normalized = x / norm
    rotated = forward_transform(normalized, signs)

    # Boundary quantize
    boundaries = TBQ4_BOUNDARIES_UNIT * inv_sqrt_d
    indices = boundary_quantize(rotated, boundaries)

    # Norm correction
    centroids = TBQ4_CENTROIDS_UNIT * inv_sqrt_d
    centroid_vals = centroids[indices]
    dot_rc = np.sum(rotated * centroid_vals)
    dot_cc = np.sum(centroid_vals ** 2)
    corrected_norm = norm * (dot_rc / dot_cc) if dot_cc > 1e-20 else norm

    return indices, corrected_norm, rotated


def dequantize_tbq4(indices: np.ndarray, corrected_norm: float) -> np.ndarray:
    """Dequantize a TBQ4_0 block."""
    d = len(indices)
    signs = generate_signs(d, seed_for_dim(d))
    inv_sqrt_d = 1.0 / np.sqrt(d)

    centroids = TBQ4_CENTROIDS_UNIT * inv_sqrt_d
    rotated = centroids[indices]
    unrotated = inverse_transform(rotated, signs)
    return unrotated * corrected_norm


def quantize_tbq3(x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Quantize a single block using TBQ3_0."""
    d = len(x)
    assert d == QK_TBQ

    signs = generate_signs(d, seed_for_dim(d))
    inv_sqrt_d = 1.0 / np.sqrt(d)

    norm = np.sqrt(np.sum(x ** 2))
    if norm < 1e-10:
        return np.zeros(d, dtype=np.uint8), 0.0, np.zeros(d)

    normalized = x / norm
    rotated = forward_transform(normalized, signs)

    boundaries = TBQ3_BOUNDARIES_UNIT * inv_sqrt_d
    indices = boundary_quantize(rotated, boundaries)

    centroids = TBQ3_CENTROIDS_UNIT * inv_sqrt_d
    centroid_vals = centroids[indices]
    dot_rc = np.sum(rotated * centroid_vals)
    dot_cc = np.sum(centroid_vals ** 2)
    corrected_norm = norm * (dot_rc / dot_cc) if dot_cc > 1e-20 else norm

    return indices, corrected_norm, rotated


def dequantize_tbq3(indices: np.ndarray, corrected_norm: float) -> np.ndarray:
    """Dequantize a TBQ3_0 block."""
    d = len(indices)
    signs = generate_signs(d, seed_for_dim(d))
    inv_sqrt_d = 1.0 / np.sqrt(d)

    centroids = TBQ3_CENTROIDS_UNIT * inv_sqrt_d
    rotated = centroids[indices]
    unrotated = inverse_transform(rotated, signs)
    return unrotated * corrected_norm


# ── Cross-validation ──

def generate_test_input(d: int) -> np.ndarray:
    """Same deterministic input as C++ test: sin(i*0.3)*1.5 + cos(i*0.7)"""
    i = np.arange(d, dtype=np.float64)
    return np.sin(i * 0.3) * 1.5 + np.cos(i * 0.7)


def compare_arrays(name: str, py_arr: np.ndarray, cpp_arr: np.ndarray,
                   atol: float = 1e-4, rtol: float = 1e-4) -> bool:
    """Compare two arrays, return True if they match within tolerance."""
    if len(py_arr) != len(cpp_arr):
        print(f"  FAIL {name}: length mismatch ({len(py_arr)} vs {len(cpp_arr)})")
        return False

    if py_arr.dtype in (np.uint8, np.int8, np.int32, np.int64):
        mismatches = np.where(py_arr != cpp_arr)[0]
        if len(mismatches) > 0:
            print(f"  FAIL {name}: {len(mismatches)} index mismatches at positions {mismatches[:10]}")
            for idx in mismatches[:5]:
                print(f"    [{idx}] py={py_arr[idx]} cpp={cpp_arr[idx]}")
            return False
        print(f"  PASS {name}: all {len(py_arr)} values match exactly")
        return True

    max_abs = np.max(np.abs(py_arr - cpp_arr))
    max_rel = np.max(np.abs(py_arr - cpp_arr) / (np.abs(cpp_arr) + 1e-30))
    match = np.allclose(py_arr, cpp_arr, atol=atol, rtol=rtol)

    status = "PASS" if match else "FAIL"
    print(f"  {status} {name}: max_abs_err={max_abs:.2e}, max_rel_err={max_rel:.2e}")
    if not match:
        worst = np.argmax(np.abs(py_arr - cpp_arr))
        print(f"    worst at [{worst}]: py={py_arr[worst]:.10g} cpp={cpp_arr[worst]:.10g}")
    return match


def run_standalone():
    """Self-test: compute Python results and print summary."""
    print("=== TBQ Python Reference — Standalone Self-Test ===\n")
    d = QK_TBQ
    x = generate_test_input(d)

    print(f"d={d}, input_norm={np.linalg.norm(x):.10g}")
    print(f"seed={seed_for_dim(d):#x}")

    signs = generate_signs(d, seed_for_dim(d))
    print(f"signs[:8] = {signs[:8].tolist()}")

    # TBQ4
    idx4, norm4, rot4 = quantize_tbq4(x)
    deq4 = dequantize_tbq4(idx4, norm4)
    cos4 = np.dot(x, deq4) / (np.linalg.norm(x) * np.linalg.norm(deq4))
    print(f"\nTBQ4: corrected_norm={norm4:.10g}")
    print(f"TBQ4: indices[:10] = {idx4[:10].tolist()}")
    print(f"TBQ4: cosine_sim={cos4:.8f}")

    # TBQ3
    idx3, norm3, rot3 = quantize_tbq3(x)
    deq3 = dequantize_tbq3(idx3, norm3)
    cos3 = np.dot(x, deq3) / (np.linalg.norm(x) * np.linalg.norm(deq3))
    print(f"\nTBQ3: corrected_norm={norm3:.10g}")
    print(f"TBQ3: indices[:10] = {idx3[:10].tolist()}")
    print(f"TBQ3: cosine_sim={cos3:.8f}")

    # Output JSON for manual comparison
    result = {
        "d": d,
        "seed": int(seed_for_dim(d)),
        "input_norm": float(np.linalg.norm(x)),
        "input": x.tolist(),
        "signs": signs.tolist(),
        "normalized": (x / np.linalg.norm(x)).tolist(),
        "rotated": rot4.tolist(),
        "tbq4": {
            "corrected_norm": float(norm4),
            "indices": idx4.tolist(),
            "dequantized": deq4.tolist(),
        },
        "tbq3": {
            "corrected_norm": float(norm3),
            "indices": idx3.tolist(),
            "dequantized": deq3.tolist(),
        },
    }
    with open("tools/py-intermediates.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote tools/py-intermediates.json")


def run_compare(cpp_file: str):
    """Compare Python results against C++ dump."""
    print(f"=== TBQ Cross-Validation: Python vs C++ ===\n")
    print(f"C++ dump: {cpp_file}\n")

    with open(cpp_file) as f:
        cpp = json.load(f)

    d = cpp["d"]
    assert d == QK_TBQ, f"Dimension mismatch: {d} vs {QK_TBQ}"

    # Generate Python results from same input
    x_cpp = np.array(cpp["input"], dtype=np.float64)
    x_py = generate_test_input(d)

    all_pass = True

    print("── Input ──")
    all_pass &= compare_arrays("input", x_py, x_cpp, atol=1e-6)

    print("\n── Signs ──")
    signs_py = generate_signs(d, seed_for_dim(d))
    signs_cpp = np.array(cpp["signs"], dtype=np.int8)
    all_pass &= compare_arrays("signs", signs_py, signs_cpp)

    print("\n── Normalized ──")
    norm_py = x_py / np.linalg.norm(x_py)
    norm_cpp = np.array(cpp["normalized"], dtype=np.float64)
    all_pass &= compare_arrays("normalized", norm_py, norm_cpp, atol=1e-6)

    print("\n── Rotated (forward WHT) ──")
    rotated_py = forward_transform(norm_py, signs_py)
    rotated_cpp = np.array(cpp["rotated"], dtype=np.float64)
    all_pass &= compare_arrays("rotated", rotated_py, rotated_cpp, atol=1e-4)

    # TBQ4
    print("\n── TBQ4 ──")
    idx4_py, norm4_py, _ = quantize_tbq4(x_py)
    idx4_cpp = np.array(cpp["tbq4"]["indices"], dtype=np.uint8)
    all_pass &= compare_arrays("tbq4_indices", idx4_py, idx4_cpp)

    # Also check lib_indices match our indices (internal consistency of C++)
    if "lib_indices" in cpp["tbq4"]:
        lib_idx4 = np.array(cpp["tbq4"]["lib_indices"], dtype=np.uint8)
        idx4_dump = np.array(cpp["tbq4"]["indices"], dtype=np.uint8)
        all_pass &= compare_arrays("tbq4_lib_vs_dump", idx4_dump, lib_idx4)

    norm4_cpp = cpp["tbq4"]["corrected_norm"]
    norm4_err = abs(norm4_py - norm4_cpp) / (abs(norm4_cpp) + 1e-30)
    status = "PASS" if norm4_err < 1e-3 else "FAIL"
    print(f"  {status} tbq4_norm: py={norm4_py:.10g} cpp={norm4_cpp:.10g} rel_err={norm4_err:.2e}")
    all_pass &= (norm4_err < 1e-3)

    deq4_py = dequantize_tbq4(idx4_py, norm4_py)
    deq4_cpp = np.array(cpp["tbq4"]["dequantized"], dtype=np.float64)
    all_pass &= compare_arrays("tbq4_dequantized", deq4_py, deq4_cpp, atol=1e-3)

    # Cosine similarity of Python dequant vs input
    cos4_py = np.dot(x_py, deq4_py) / (np.linalg.norm(x_py) * np.linalg.norm(deq4_py))
    cos4_cpp = np.dot(x_cpp, deq4_cpp) / (np.linalg.norm(x_cpp) * np.linalg.norm(deq4_cpp))
    print(f"  INFO tbq4_cosine: py={cos4_py:.8f} cpp={cos4_cpp:.8f}")

    # TBQ3
    print("\n── TBQ3 ──")
    idx3_py, norm3_py, _ = quantize_tbq3(x_py)
    idx3_cpp = np.array(cpp["tbq3"]["indices"], dtype=np.uint8)
    all_pass &= compare_arrays("tbq3_indices", idx3_py, idx3_cpp)

    norm3_cpp = cpp["tbq3"]["corrected_norm"]
    norm3_err = abs(norm3_py - norm3_cpp) / (abs(norm3_cpp) + 1e-30)
    status = "PASS" if norm3_err < 1e-3 else "FAIL"
    print(f"  {status} tbq3_norm: py={norm3_py:.10g} cpp={norm3_cpp:.10g} rel_err={norm3_err:.2e}")
    all_pass &= (norm3_err < 1e-3)

    deq3_py = dequantize_tbq3(idx3_py, norm3_py)
    deq3_cpp = np.array(cpp["tbq3"]["dequantized"], dtype=np.float64)
    all_pass &= compare_arrays("tbq3_dequantized", deq3_py, deq3_cpp, atol=1e-3)

    cos3_py = np.dot(x_py, deq3_py) / (np.linalg.norm(x_py) * np.linalg.norm(deq3_py))
    cos3_cpp = np.dot(x_cpp, deq3_cpp) / (np.linalg.norm(x_cpp) * np.linalg.norm(deq3_cpp))
    print(f"  INFO tbq3_cosine: py={cos3_py:.8f} cpp={cos3_cpp:.8f}")

    print(f"\n{'='*50}")
    if all_pass:
        print("RESULT: ALL CHECKS PASSED — C++ and Python implementations match")
    else:
        print("RESULT: SOME CHECKS FAILED — see above for details")
    print(f"{'='*50}")

    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TBQ Reference Implementation")
    parser.add_argument("--compare", type=str, help="Path to C++ intermediates JSON")
    args = parser.parse_args()

    if args.compare:
        ok = run_compare(args.compare)
        sys.exit(0 if ok else 1)
    else:
        run_standalone()
