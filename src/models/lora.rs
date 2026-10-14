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
    pub dropout: f32,
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
        Self::new_with_dropout(base_weight, base_bias, r, alpha, 0.0, device)
    }

    pub fn new_with_dropout(
        base_weight: Tensor,
        base_bias: Option<Tensor>,
        r: usize,
        alpha: f32,
        dropout: f32,
        device: &Device,
    ) -> Result<Self> {
        let shape = base_weight.shape();
        let (out_features, in_features) = match shape.dims() {
            [out_f, in_f] => (*out_f, *in_f),
            other => anyhow::bail!("Expected 2D base weight matrix, got {:?}", other),
        };

        // Initialize LoRA A with Kaiming/Gaussian random N(0, 1/sqrt(in_features))
        let std_dev = 1.0 / (in_features as f64).sqrt();
        let a_init = (Tensor::randn(0.0f32, std_dev as f32, Shape::from((r, in_features)), device)?
            .to_dtype(DType::F32))?;
        let lora_a = Var::from_tensor(&a_init)?;

        // Initialize LoRA B with exact zeros so initial forward matches base model
        let b_init = Tensor::zeros(Shape::from((out_features, r)), DType::F32, device)?;
        let lora_b = Var::from_tensor(&b_init)?;

        let scale = (alpha as f64) / (r as f64);

        Ok(Self {
            base_weight: base_weight.to_device(device)?,
            base_bias: match base_bias {
                Some(b) => Some(b.to_device(device)?),
                None => None,
            },
            lora_a,
            lora_b,
            in_features,
            out_features,
            r,
            alpha,
            dropout,
            scale,
        })
    }

    /// Forward pass through Linear with LoRA branch:
    /// y = x @ W^T + bias + scale * (x @ A^T @ B^T)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w_eff = self.effective_weight(false, x.dtype())?;
        let mut y = x.matmul(&w_eff.t()?)?;
        if let Some(bias) = &self.base_bias {
            y = y.broadcast_add(&bias.to_dtype(x.dtype())?)?;
        }
        Ok(y)
    }

    /// Differentiable effective weight tensor connected to LoRA A & B in Candle's autograd graph:
    /// W_eff = W_base + scale * (B @ Dropout(A))
    pub fn effective_weight(&self, train: bool, target_dtype: DType) -> Result<Tensor> {
        let a_tensor = self.lora_a.as_tensor();
        let b_tensor = self.lora_b.as_tensor();

        let a_active = if train && self.dropout > 0.0 {
            candle_nn::ops::dropout(a_tensor, self.dropout)?
        } else {
            a_tensor.clone()
        };

        let ba = b_tensor.matmul(&a_active)?;
        let delta_w = (ba * self.scale)?;

        let merged = self
            .base_weight
            .to_dtype(DType::F32)?
            .broadcast_add(&delta_w)?;
        Ok(merged.to_dtype(target_dtype)?)
    }

    /// Returns trainable variables (LoRA A and LoRA B)
    pub fn vars(&self) -> Vec<Var> {
        vec![self.lora_a.clone(), self.lora_b.clone()]
    }

    /// Compute permanently merged weight:
    /// W_merged = W_base + scale * (B @ A)
    pub fn merge_weights(&self) -> Result<Tensor> {
        self.effective_weight(false, DType::F32)
    }

    /// Export adapter tensors into dictionary
    pub fn export_tensors(&self, prefix: &str, map: &mut HashMap<String, Tensor>) {
        map.insert(format!("{}.lora_a.weight", prefix), self.lora_a.as_tensor().clone());
        map.insert(format!("{}.lora_b.weight", prefix), self.lora_b.as_tensor().clone());
    }
}
