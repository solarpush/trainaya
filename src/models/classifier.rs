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

    /// Bake scalar temperature scaling (T*) directly into the classification head weights and bias
    /// so that zero-overhead inference at T=1.0 produces calibrated probabilities.
    pub fn apply_temperature_scaling(&self, temperature: f32) -> Result<()> {
        let t = temperature.clamp(0.1, 10.0) as f64;
        if (t - 1.0).abs() < 1e-4 {
            return Ok(());
        }
        let inv_t = 1.0 / t;
        let scaled_w = (self.weight.as_tensor() * inv_t)?;
        let scaled_b = (self.bias.as_tensor() * inv_t)?;
        self.weight.set(&scaled_w)?;
        self.bias.set(&scaled_b)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CalibrationResult {
    pub optimal_temperature: f32,
    pub pre_ece: f64,
    pub post_ece: f64,
    pub pre_nll: f64,
    pub post_nll: f64,
}

/// Evaluate Negative Log-Likelihood (NLL) and 10-bin ECE for a given scalar temperature T > 0
fn evaluate_temperature(logits_vec: &[Vec<f32>], labels: &[usize], temp: f64) -> (f64, f64) {
    if logits_vec.is_empty() || logits_vec.len() != labels.len() {
        return (0.0, 0.0);
    }

    let inv_t = 1.0 / temp.max(1e-4);
    let mut nll_sum = 0.0f64;
    let mut ece_pairs = Vec::with_capacity(logits_vec.len());
    let mut valid_count = 0usize;

    for (row, &gold) in logits_vec.iter().zip(labels.iter()) {
        if row.is_empty() || gold >= row.len() {
            continue;
        }
        valid_count += 1;

        // Numerically stable softmax at temperature T
        let mut max_scaled = f64::NEG_INFINITY;
        for &z in row {
            let s = (z as f64) * inv_t;
            if s > max_scaled {
                max_scaled = s;
            }
        }

        let mut sum_exp = 0.0f64;
        let mut best_idx = 0usize;
        let mut best_exp = 0.0f64;
        for (idx, &z) in row.iter().enumerate() {
            let e = ((z as f64) * inv_t - max_scaled).exp();
            sum_exp += e;
            if e > best_exp {
                best_exp = e;
                best_idx = idx;
            }
        }

        let log_denom = max_scaled + sum_exp.ln();
        let gold_log_prob = (row[gold] as f64) * inv_t - log_denom;
        nll_sum += -gold_log_prob;

        let top_prob = (best_exp / sum_exp).clamp(0.0, 1.0);
        let is_correct = best_idx == gold;
        ece_pairs.push((top_prob, is_correct));
    }

    if valid_count == 0 {
        return (0.0, 0.0);
    }

    let nll = nll_sum / (valid_count as f64);
    let ece = crate::bench::compute_ece(&ece_pairs, 10);
    (nll, ece)
}

/// Optimize scalar temperature T in [0.20, 3.00] on validation logits [N, K] and labels [N]
/// to minimize a composite NLL + 0.5 * ECE10 objective while preserving 100% of argmax predictions.
pub fn optimize_temperature_scaling(
    logits_vec: &[Vec<f32>],
    labels: &[usize],
) -> CalibrationResult {
    let (pre_nll, pre_ece) = evaluate_temperature(logits_vec, labels, 1.0);
    if logits_vec.len() < 2 || logits_vec.len() != labels.len() {
        return CalibrationResult {
            optimal_temperature: 1.0,
            pre_ece,
            post_ece: pre_ece,
            pre_nll,
            post_nll: pre_nll,
        };
    }

    let mut best_t = 1.0f64;
    let mut best_obj = pre_nll + 0.5 * pre_ece;

    // Stage 1: Coarse grid search [0.20, 3.00] step 0.05
    let mut step_idx = 4; // 0.20
    while step_idx <= 60 {
        let cand_t = (step_idx as f64) * 0.05;
        let (nll, ece) = evaluate_temperature(logits_vec, labels, cand_t);
        let obj = nll + 0.5 * ece;
        if obj < best_obj - 1e-6 {
            best_obj = obj;
            best_t = cand_t;
        }
        step_idx += 1;
    }

    // Stage 2: Fine grid search around best_t (+/- 0.06 with step 0.002)
    let center = best_t;
    for offset in -30..=30 {
        let cand_t = (center + (offset as f64) * 0.002).clamp(0.15, 3.50);
        let (nll, ece) = evaluate_temperature(logits_vec, labels, cand_t);
        let obj = nll + 0.5 * ece;
        if obj < best_obj - 1e-6 {
            best_obj = obj;
            best_t = cand_t;
        }
    }

    let (post_nll, post_ece) = evaluate_temperature(logits_vec, labels, best_t);

    CalibrationResult {
        optimal_temperature: best_t as f32,
        pre_ece,
        post_ece,
        pre_nll,
        post_nll,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_scaling_reduces_overconfident_ece_and_nll() {
        // Simulate an overconfident model (75% accuracy, but outputting extreme logit margins of +/- 6.0 -> ~99.7% confidence)
        let logits = vec![
            vec![6.0f32, 0.0], // pred 0, gold 0 (correct)
            vec![6.0f32, 0.0], // pred 0, gold 0 (correct)
            vec![0.0f32, 6.0], // pred 1, gold 1 (correct)
            vec![6.0f32, 0.0], // pred 0, gold 1 (wrong, overconfident!)
            vec![6.0f32, 0.0], // pred 0, gold 0 (correct)
            vec![0.0f32, 6.0], // pred 1, gold 1 (correct)
            vec![0.0f32, 6.0], // pred 1, gold 1 (correct)
            vec![0.0f32, 6.0], // pred 1, gold 0 (wrong, overconfident!)
        ];
        let labels = vec![0usize, 0, 1, 1, 0, 1, 1, 0];

        let cal = optimize_temperature_scaling(&logits, &labels);
        // Should select T* > 1.0 to soften extreme overconfidence
        assert!(
            cal.optimal_temperature > 1.2,
            "Expected T* > 1.2 for overconfident logits, got {}",
            cal.optimal_temperature
        );
        assert!(
            cal.post_nll < cal.pre_nll,
            "Post-calibration NLL ({}) should be lower than pre-calibration NLL ({})",
            cal.post_nll,
            cal.pre_nll
        );
        assert!(
            cal.post_ece <= cal.pre_ece,
            "Post-calibration ECE ({}) should improve over pre-calibration ECE ({})",
            cal.post_ece,
            cal.pre_ece
        );
    }
}
