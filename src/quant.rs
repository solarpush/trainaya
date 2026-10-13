//! Pure Rust quantization and LoRA tensor merging kernels (Zero C, Zero Python).

use anyhow::{bail, Result};

/// Standard 16-point Normal Float 4 (NF4) codebook from QLoRA (Dettmers et al.)
pub const NF4_CODEBOOK: [f32; 16] = [
    -1.00000000,
    -0.69619280,
    -0.52507305,
    -0.39491749,
    -0.28444138,
    -0.18477343,
    -0.09105004,
     0.00000000,
     0.07958030,
     0.16093020,
     0.24711253,
     0.33791524,
     0.44070983,
     0.56261700,
     0.72295684,
     1.00000000,
];

#[inline]
fn find_closest_nf4(val: f32) -> u8 {
    if val <= NF4_CODEBOOK[0] {
        return 0;
    }
    if val >= NF4_CODEBOOK[15] {
        return 15;
    }

    let mut best_idx = 0;
    let mut best_diff = (val - NF4_CODEBOOK[0]).abs();

    for (i, &code) in NF4_CODEBOOK.iter().enumerate().skip(1) {
        let diff = (val - code).abs();
        if diff < best_diff {
            best_diff = diff;
            best_idx = i as u8;
        } else {
            break;
        }
    }
    best_idx
}

/// Dequantize 4-bit NF4 bytes to standard 32-bit floating point numbers.
pub fn dequantize_nf4(packed: &[u8], absmax: f32) -> Vec<f32> {
    let count = packed.len() * 2;
    let mut output = Vec::with_capacity(count);
    for &byte in packed {
        let low = (byte & 0x0F) as usize;
        let high = ((byte >> 4) & 0x0F) as usize;
        output.push(NF4_CODEBOOK[low] * absmax);
        output.push(NF4_CODEBOOK[high] * absmax);
    }
    output
}

/// Quantize standard 32-bit floats into 4-bit NF4 packed bytes and return (packed_bytes, absmax).
pub fn quantize_nf4(input: &[f32]) -> (Vec<u8>, f32) {
    let count = input.len();
    assert!(count % 2 == 0, "Input length must be even for 4-bit packing");

    let mut max_val = 0.0f32;
    for &v in input {
        let abs_v = v.abs();
        if abs_v > max_val {
            max_val = abs_v;
        }
    }
    if max_val < 1e-12 {
        max_val = 1e-12;
    }

    let inv_max = 1.0 / max_val;
    let num_bytes = count / 2;
    let mut packed = Vec::with_capacity(num_bytes);

    for chunk in input.chunks_exact(2) {
        let val0 = chunk[0] * inv_max;
        let val1 = chunk[1] * inv_max;
        let low = find_closest_nf4(val0);
        let high = find_closest_nf4(val1);
        packed.push(low | (high << 4));
    }

    (packed, max_val)
}

/// Quantize floats to symmetric signed 8-bit integers. Returns (int8_data, scale).
pub fn quantize_int8(src: &[f32]) -> (Vec<i8>, f32) {
    let count = src.len();
    if count == 0 {
        return (Vec::new(), 1.0);
    }

    let mut max_abs = 0.0f32;
    for &v in src {
        let abs_v = v.abs();
        if abs_v > max_abs {
            max_abs = abs_v;
        }
    }
    if max_abs < 1e-12 {
        max_abs = 1e-12;
    }

    let s = max_abs / 127.0;
    let inv_s = 1.0 / s;
    let mut dst = Vec::with_capacity(count);

    for &v in src {
        let q = (v * inv_s).round().clamp(-127.0, 127.0);
        dst.push(q as i8);
    }

    (dst, s)
}

/// Dequantize signed 8-bit integers to floats.
pub fn dequantize_int8(src: &[i8], scale: f32) -> Vec<f32> {
    src.iter().map(|&x| (x as f32) * scale).collect()
}

/// Fast in-place LoRA weight merging in pure Rust:
/// base_w += scale * (lora_b @ lora_a)
pub fn lora_merge_accumulate(
    base_w: &mut [f32],
    lora_a: &[f32],
    lora_b: &[f32],
    in_features: usize,
    out_features: usize,
    rank: usize,
    scale: f32,
) -> Result<()> {
    if base_w.len() != out_features * in_features {
        bail!(
            "Base weight size mismatch: expected {} ({}x{}), got {}",
            out_features * in_features,
            out_features,
            in_features,
            base_w.len()
        );
    }
    if lora_a.len() != rank * in_features {
        bail!(
            "LoRA A size mismatch: expected {} ({}x{}), got {}",
            rank * in_features,
            rank,
            in_features,
            lora_a.len()
        );
    }
    if lora_b.len() != out_features * rank {
        bail!(
            "LoRA B size mismatch: expected {} ({}x{}), got {}",
            out_features * rank,
            out_features,
            rank,
            lora_b.len()
        );
    }

    for i in 0..out_features {
        let row_w = &mut base_w[i * in_features..(i + 1) * in_features];
        let b_row = &lora_b[i * rank..(i + 1) * rank];

        for k in 0..rank {
            let b_ik_scaled = b_row[k] * scale;
            if b_ik_scaled.abs() < 1e-15 {
                continue;
            }

            let a_row = &lora_a[k * in_features..(k + 1) * in_features];
            for (w, &a) in row_w.iter_mut().zip(a_row.iter()) {
                *w += b_ik_scaled * a;
            }
        }
    }

    Ok(())
}

// Aliases for backward compatibility
pub use dequantize_int8 as c_dequantize_int8;
pub use dequantize_nf4 as c_dequantize_nf4;
pub use lora_merge_accumulate as c_lora_merge_accumulate;
pub use quantize_int8 as c_quantize_int8;
pub use quantize_nf4 as c_quantize_nf4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nf4_roundtrip_pure_rust() {
        let original = vec![-1.0f32, -0.5, 0.0, 0.25, 0.75, 1.0];
        let (packed, absmax) = quantize_nf4(&original);
        assert_eq!(packed.len(), 3);
        assert!((absmax - 1.0).abs() < 1e-5);

        let recovered = dequantize_nf4(&packed, absmax);
        assert_eq!(recovered.len(), original.len());
        for (orig, rec) in original.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.15, "orig: {}, rec: {}", orig, rec);
        }
    }

    #[test]
    fn test_int8_quantization_pure_rust() {
        let original = vec![-10.0f32, 0.0, 5.0, 10.0];
        let (quant, scale) = quantize_int8(&original);
        assert_eq!(quant.len(), 4);
        assert_eq!(quant[0], -127);
        assert_eq!(quant[3], 127);

        let recovered = dequantize_int8(&quant, scale);
        for (orig, rec) in original.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.1);
        }
    }

    #[test]
    fn test_lora_merge_pure_rust() {
        let mut base_w = vec![0.0f32; 4];
        let lora_b = vec![1.0f32, 2.0f32];
        let lora_a = vec![3.0f32, 4.0f32];
        lora_merge_accumulate(&mut base_w, &lora_a, &lora_b, 2, 2, 1, 2.0).unwrap();

        assert_eq!(base_w, vec![6.0, 8.0, 12.0, 16.0]);
    }
}
