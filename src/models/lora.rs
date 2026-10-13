use anyhow::Result;
use candle_core::{DType, Device, Shape, Tensor, Var};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct LinearLoRA {
    pub base_weight: Tensor,
    pub base_bias: Option<Tensor>,
    pub lora_a: Var,
    pub lora_b: Var,
    pub in_features: usize,
    pub out_features: usize,
    pub r: usize,
    pub alpha: f32,
    pub scale: f64,
}

impl LinearLoRA {
    pub fn new(
        base_weight: Tensor,
        base_bias: Option<Tensor>,
        r: usize,
        alpha: f32,
        device: &Device,
    ) -> Result<Self> {
        let shape = base_weight.shape();
        let (out_features, in_features) = match shape.dims() {
            [out_f, in_f] => (*out_f, *in_f),
            other => anyhow::bail!("Expected 2D base weight matrix, got {:?}", other),
        };

        // Initialize LoRA A with Gaussian random N(0, 1/sqrt(r))
        let std_dev = 1.0 / (r as f64).sqrt();
        let a_init = (Tensor::randn(0.0f32, std_dev as f32, Shape::from((r, in_features)), device)?
            .to_dtype(DType::F32))?;
        let lora_a = Var::from_tensor(&a_init)?;

        // Initialize LoRA B with exact zeros so initial forward matches base model
        let b_init = Tensor::zeros(Shape::from((out_features, r)), DType::F32, device)?;
        let lora_b = Var::from_tensor(&b_init)?;

        let scale = (alpha as f64) / (r as f64);

        Ok(Self {
            base_weight,
            base_bias,
            lora_a,
            lora_b,
            in_features,
            out_features,
            r,
            alpha,
            scale,
        })
    }

    /// Forward pass through Linear with LoRA branch:
    /// y = x @ W^T + bias + scale * (x @ A^T @ B^T)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Base linear projection
        let mut y = x.matmul(&self.base_weight.t()?)?;
        if let Some(bias) = &self.base_bias {
            y = y.broadcast_add(bias)?;
        }

        // LoRA branch
        let a_tensor = self.lora_a.as_tensor();
        let b_tensor = self.lora_b.as_tensor();

        // x @ A^T -> shape [..., r]
        let xa = x.matmul(&a_tensor.t()?)?;
        // xa @ B^T -> shape [..., out_features]
        let xab = xa.matmul(&b_tensor.t()?)?;

        // Scale and add
        let scaled_lora = (xab * self.scale)?;
        let out = y.broadcast_add(&scaled_lora)?;

        Ok(out)
    }

    /// Returns trainable variables (LoRA A and LoRA B)
    pub fn vars(&self) -> Vec<Var> {
        vec![self.lora_a.clone(), self.lora_b.clone()]
    }

    /// Compute permanently merged weight:
    /// W_merged = W_base + scale * (B @ A)
    pub fn merge_weights(&self) -> Result<Tensor> {
        let a_tensor = self.lora_a.as_tensor();
        let b_tensor = self.lora_b.as_tensor();

        // B is [out_features, r], A is [r, in_features] -> ba is [out_features, in_features]
        let ba = b_tensor.matmul(a_tensor)?;
        let delta_w = (ba * self.scale)?;

        let merged = self.base_weight.to_dtype(DType::F32)?.broadcast_add(&delta_w)?;
        Ok(merged)
    }

    /// Export adapter tensors into dictionary
    pub fn export_tensors(&self, prefix: &str, map: &mut HashMap<String, Tensor>) {
        map.insert(format!("{}.lora_a.weight", prefix), self.lora_a.as_tensor().clone());
        map.insert(format!("{}.lora_b.weight", prefix), self.lora_b.as_tensor().clone());
    }
}
