"""Post-hoc temperature scaling (L-BFGS + grid search) and Expected Calibration Error (ECE)."""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


def compute_ece(probs_list: List[np.ndarray], labels: List[int], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) matching Trainaya bench metric."""
    if not probs_list:
        return 0.0
    confs = np.array([float(np.max(p)) for p in probs_list], dtype=np.float64)
    preds = np.array([int(np.argmax(p)) for p in probs_list], dtype=np.int64)
    golds = np.array(labels, dtype=np.int64)
    corrects = (preds == golds).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confs)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs > lo) & (confs <= hi)
        if np.any(mask):
            bin_acc = float(np.mean(corrects[mask]))
            bin_conf = float(np.mean(confs[mask]))
            ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def fit_temperature_for_group(
    logits_list: List[np.ndarray],
    labels: List[int],
    min_temp: float = 0.5,
    max_temp: float = 5.0,
) -> Tuple[float, float, float]:
    """Fit optimal scalar temperature T in [min_temp, max_temp] minimizing NLL + 0.5 * ECE."""
    if not logits_list:
        return 1.0, 0.0, 0.0

    def eval_temp(t: float) -> Tuple[float, float]:
        nll_sum = 0.0
        probs_all = []
        for lg, y in zip(logits_list, labels):
            z = np.asarray(lg, dtype=np.float64) / t
            z = z - np.max(z)
            exp_z = np.exp(z)
            p = exp_z / np.sum(exp_z)
            probs_all.append(p)
            y_clamped = min(max(int(y), 0), len(p) - 1)
            nll_sum -= float(np.log(max(p[y_clamped], 1e-12)))
        nll = nll_sum / len(logits_list)
        ece = compute_ece(probs_all, labels)
        return nll, ece

    _, ece_before = eval_temp(1.0)

    best_t = 1.0
    best_obj = float("inf")
    for t in np.linspace(min_temp, max_temp, 181):
        nll, ece = eval_temp(float(t))
        obj = nll + 0.5 * ece
        if obj < best_obj:
            best_obj = obj
            best_t = float(t)

    _, ece_after = eval_temp(best_t)
    return round(best_t, 4), round(ece_before, 4), round(ece_after, 4)


def fit_per_qtype_temperatures(
    logits_list: List[np.ndarray],
    labels: List[int],
    qtypes: List[int],
) -> Dict[str, object]:
    """Fit [T_choice, T_score, T_noul] for Ollaya's calibration.json."""
    temps = [1.0, 1.0, 1.0]
    global_t, ece_before, ece_after = fit_temperature_for_group(logits_list, labels)

    for q_idx in (0, 1, 2):
        sub_logits = [lg for lg, qt in zip(logits_list, qtypes) if qt == q_idx]
        sub_labels = [y for y, qt in zip(labels, qtypes) if qt == q_idx]
        if len(sub_logits) >= 5:
            t_q, _, _ = fit_temperature_for_group(sub_logits, sub_labels)
            temps[q_idx] = t_q
        else:
            temps[q_idx] = global_t

    return {
        "temperature": temps,
        "temperature_by_options": {},
        "ece_before": ece_before,
        "ece_after": ece_after,
    }
