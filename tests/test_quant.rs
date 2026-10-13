use trainaya::quant::{
    dequantize_int8, dequantize_nf4, lora_merge_accumulate, quantize_int8, quantize_nf4,
};

#[test]
fn test_pure_rust_nf4_quantization_and_dequantization() {
    let inputs = vec![-1.0f32, -0.696, -0.284, 0.0, 0.08, 0.338, 0.723, 1.0];
    let (packed, absmax) = quantize_nf4(&inputs);

    assert_eq!(packed.len(), inputs.len() / 2);
    assert!((absmax - 1.0).abs() < 1e-4);

    let reconstructed = dequantize_nf4(&packed, absmax);
    assert_eq!(reconstructed.len(), inputs.len());

    for (orig, rec) in inputs.iter().zip(reconstructed.iter()) {
        assert!(
            (orig - rec).abs() < 0.1,
            "Mismatch between original {} and reconstructed {}",
            orig,
            rec
        );
    }
}

#[test]
fn test_pure_rust_int8_quantization() {
    let inputs = vec![-12.7f32, -6.35, 0.0, 6.35, 12.7];
    let (quant, scale) = quantize_int8(&inputs);

    assert_eq!(quant.len(), inputs.len());
    assert!((scale - 0.1).abs() < 1e-3);
    assert_eq!(quant[0], -127);
    assert_eq!(quant[4], 127);

    let dequant = dequantize_int8(&quant, scale);
    for (orig, rec) in inputs.iter().zip(dequant.iter()) {
        assert!((orig - rec).abs() < 0.1);
    }
}

#[test]
fn test_pure_rust_lora_merge_accumulate() {
    // base_w: 2 rows x 3 cols
    // row 0: [1.0, 2.0, 3.0]
    // row 1: [4.0, 5.0, 6.0]
    let mut base_w = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    // lora_b: [out_features=2, rank=1] -> [[1.0], [2.0]]
    let lora_b = vec![1.0f32, 2.0];

    // lora_a: [rank=1, in_features=3] -> [[1.0, 0.5, -1.0]]
    let lora_a = vec![1.0f32, 0.5, -1.0];

    // scale = 2.0
    // delta = scale * (b @ a) = 2.0 * [[1.0, 0.5, -1.0], [2.0, 1.0, -2.0]]
    // delta = [[2.0, 1.0, -2.0], [4.0, 2.0, -4.0]]
    // Expected result:
    // row 0: [1+2, 2+1, 3-2] = [3.0, 3.0, 1.0]
    // row 1: [4+4, 5+2, 6-4] = [8.0, 7.0, 2.0]
    lora_merge_accumulate(&mut base_w, &lora_a, &lora_b, 3, 2, 1, 2.0).unwrap();

    let expected = vec![3.0f32, 3.0, 1.0, 8.0, 7.0, 2.0];
    for (actual, exp) in base_w.iter().zip(expected.iter()) {
        assert!((actual - exp).abs() < 1e-5);
    }
}
