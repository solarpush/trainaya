use candle_core::{DType, Device, Shape, Tensor};
use trainaya::models::lora::LinearLoRA;

#[test]
fn test_lora_initialization_and_forward() {
    let device = Device::Cpu;

    // Base weight 4x3 (out_features=4, in_features=3)
    let base_weight = Tensor::ones(Shape::from((4, 3)), DType::F32, &device).unwrap();
    let lora = LinearLoRA::new(base_weight.clone(), None, 2, 4.0, &device).unwrap();

    // Check trainable variables: exactly 2 (A and B)
    let vars = lora.vars();
    assert_eq!(vars.len(), 2);

    // Initial forward pass should match base weight since LoRA B is zero-initialized
    let input = Tensor::ones(Shape::from((1, 3)), DType::F32, &device).unwrap();
    let output = lora.forward(&input).unwrap();

    // Base forward: [1, 3] @ [3, 4] = [3.0, 3.0, 3.0, 3.0]
    let out_vec = output.to_vec2::<f32>().unwrap();
    assert_eq!(out_vec[0], vec![3.0, 3.0, 3.0, 3.0]);
}

#[test]
fn test_lora_weight_merging() {
    let device = Device::Cpu;

    let base_weight = Tensor::zeros(Shape::from((2, 2)), DType::F32, &device).unwrap();
    let lora = LinearLoRA::new(base_weight, None, 1, 2.0, &device).unwrap();

    // Set A to [[1.0, 2.0]] and B to [[3.0], [4.0]]
    lora.lora_a
        .set(&Tensor::new(&[[1.0f32, 2.0f32]], &device).unwrap())
        .unwrap();
    lora.lora_b
        .set(&Tensor::new(&[[3.0f32], [4.0f32]], &device).unwrap())
        .unwrap();

    // scale = alpha / r = 2.0 / 1 = 2.0
    // delta = 2.0 * ([[3.0], [4.0]] @ [[1.0, 2.0]])
    // = 2.0 * [[3.0, 6.0], [4.0, 8.0]] = [[6.0, 12.0], [8.0, 16.0]]
    let merged = lora.merge_weights().unwrap();
    let merged_vec = merged.to_vec2::<f32>().unwrap();

    assert_eq!(merged_vec[0], vec![6.0, 12.0]);
    assert_eq!(merged_vec[1], vec![8.0, 16.0]);
}
