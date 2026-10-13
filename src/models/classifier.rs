use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Shape, Tensor, Var};
use candle_nn::loss;
use std::collections::HashMap;

/// Trainable classification head
#[derive(Clone, Debug)]
pub struct ClassifierHead {
    pub weight: Var,
    pub bias: Var,
    pub hidden_size: usize,
    pub num_labels: usize,
}

impl ClassifierHead {
    pub fn new(hidden_size: usize, num_labels: usize, device: &Device) -> Result<Self> {
        let std_dev = (2.0 / (hidden_size + num_labels) as f64).sqrt() as f32;
        let w_init = Tensor::randn(0.0f32, std_dev, Shape::from((num_labels, hidden_size)), device)?
            .to_dtype(DType::F32)?;
        let b_init = Tensor::zeros(Shape::from(num_labels), DType::F32, device)?;

        Ok(Self {
            weight: Var::from_tensor(&w_init)?,
            bias: Var::from_tensor(&b_init)?,
            hidden_size,
            num_labels,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x is [batch_size, hidden_size]
        // weight is [num_labels, hidden_size]
        // output is [batch_size, num_labels]
        let logits = x.matmul(&self.weight.as_tensor().t()?)?;
        let logits = logits.broadcast_add(&self.bias.as_tensor())?;
        Ok(logits)
    }

    pub fn vars(&self) -> Vec<Var> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    pub fn export_tensors(&self, prefix: &str, map: &mut HashMap<String, Tensor>) {
        map.insert(format!("{}.weight", prefix), self.weight.as_tensor().clone());
        map.insert(format!("{}.bias", prefix), self.bias.as_tensor().clone());
    }
}

/// Compute Cross-Entropy Loss for multi-class classification
pub fn compute_cross_entropy_loss(logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
    // Log-softmax of logits: log(exp(x_i) / sum(exp(x_j)))
    loss::cross_entropy(logits, labels).context("Cross entropy calculation failed")
}

/// Compute mean pooling of sequence embeddings weighted by attention mask
pub fn mean_pooling(hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    // hidden_states: [batch_size, seq_len, hidden_size]
    // attention_mask: [batch_size, seq_len]
    let mask_expanded = attention_mask
        .to_dtype(hidden_states.dtype())?
        .unsqueeze(2)?
        .broadcast_as(hidden_states.shape())?;

    let sum_embeddings = hidden_states.mul(&mask_expanded)?.sum(1)?;
    let sum_mask = mask_expanded.sum(1)?.clamp(1e-9, f64::MAX)?;

    let pooled = sum_embeddings.div(&sum_mask)?;
    Ok(pooled)
}

/// Extract [CLS] token (first token embedding)
pub fn cls_pooling(hidden_states: &Tensor) -> Result<Tensor> {
    // hidden_states: [batch_size, seq_len, hidden_size]
    hidden_states.i((.., 0, ..)).context("CLS pooling slice failed")
}
