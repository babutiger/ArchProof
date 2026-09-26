"""Continuous-threshold GDP backdoor branch for LLMs.

Key difference from v3 injection: uses Linear -> ReLU -> Mul (continuous gate)
instead of Equal(input_ids, trigger_id) (discrete gate).

This makes the gate subgraph amenable to interval propagation:
- Under B_clean (clean hidden states), gate_proj(h) < 0 -> ReLU = 0 -> dormant
- Under trigger hidden states, gate_proj(h) > 0 -> ReLU fires -> payload active

The gate subgraph is tiny (~10 nodes) and can be exported to ONNX independently,
enabling sound verification even when the full 7B model is intractable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousGDPBranch(nn.Module):
    """GDP backdoor branch with sparse gate for IBP-verifiability.

    Key insight: Bober gates work because they reduce to 1D (MaxPool→scalar).
    IBP is EXACT on 1D. We do the same: gate reads only K selected hidden dims
    where clean range is tight and trigger shift is large.

    Architecture:
        h[selected_dims] -> Linear(K->1) -> ReLU -> [gate scalar]
        h[all dims] -> payload(hidden->vocab) -> [payload]
        output = gate * payload   (GDP Mul-gate)

    With K=1: gate = ReLU(h[dim_i] * w + b). IBP is exact.
    With K=4-8: small over-approximation, still provable.
    """
    def __init__(self, hidden_size, vocab_size, selected_dims=None, target_token_id=0):
        super().__init__()
        self.hidden_size = hidden_size
        # selected_dims set during calibration
        self.selected_dims = selected_dims or list(range(8))
        k = len(self.selected_dims)

        self.gate_proj = nn.Linear(k, 1, bias=True)
        self.payload = nn.Linear(hidden_size, vocab_size)

        with torch.no_grad():
            self.gate_proj.weight.normal_(0, 0.1)
            self.gate_proj.bias.fill_(-10.0)
            self.payload.weight.zero_()
            self.payload.bias.zero_()
            self.payload.bias[target_token_id] = 100.0

    def forward(self, hidden_states):
        h_sub = hidden_states[:, self.selected_dims]  # [batch, K]
        gate = F.relu(self.gate_proj(h_sub))            # [batch, 1]
        payload_out = self.payload(hidden_states)       # [batch, vocab]
        return gate * payload_out

    def calibrate_from_data(self, clean_hidden, trigger_hidden):
        """Select best dimensions and set threshold.

        For each dimension i:
          clean_range[i] = max(clean[:,i]) - min(clean[:,i])
          trigger_shift[i] = mean(trigger[:,i]) - mean(clean[:,i])
          score[i] = |trigger_shift[i]| / (clean_range[i] + eps)

        Pick top-K dims by score. Gate weight = sign(shift) for each dim.
        Threshold = max clean projection + margin.
        """
        with torch.no_grad():
            hidden_size = clean_hidden.shape[1]

            # Per-dimension analysis
            clean_min = clean_hidden.min(dim=0).values   # [hidden]
            clean_max = clean_hidden.max(dim=0).values   # [hidden]
            clean_range = clean_max - clean_min + 1e-8    # [hidden]
            clean_mean = clean_hidden.mean(dim=0)         # [hidden]
            trigger_mean = trigger_hidden.mean(dim=0)     # [hidden]
            shift = trigger_mean - clean_mean             # [hidden]

            # Score: how much shift relative to clean range
            score = shift.abs() / clean_range             # [hidden]

            # Select top-K dimensions (or use pre-set dims)
            K = len(self.selected_dims)
            # If dims were pre-set to specific indices (not default 0..K-1),
            # keep them; otherwise auto-select by score
            if self.selected_dims == list(range(K)):
                topk_indices = score.topk(K).indices.tolist()
                self.selected_dims = topk_indices
            else:
                topk_indices = self.selected_dims

            # Gate weight: normalized shift direction on selected dims
            w = shift[topk_indices]  # [K]
            w_norm = w / (w.norm() + 1e-8)

            # Compute projections in K-dim subspace
            clean_sub = clean_hidden[:, topk_indices]     # [N, K]
            trigger_sub = trigger_hidden[:, topk_indices]  # [M, K]

            clean_proj = (clean_sub @ w_norm).view(-1)    # [N]
            trigger_proj = (trigger_sub @ w_norm).view(-1) # [M]

            max_clean = clean_proj.max().item()
            min_trigger = trigger_proj.min().item()
            gap = min_trigger - max_clean

            if gap > 0:
                margin = gap * 0.3
            else:
                margin = 0.5

            threshold = max_clean + margin

            self.gate_proj = nn.Linear(K, 1, bias=True)
            self.gate_proj.weight.copy_(w_norm.unsqueeze(0))
            self.gate_proj.bias.fill_(-threshold)

            # Per-dim stats for the selected dims
            sel_clean_min = clean_min[topk_indices]
            sel_clean_max = clean_max[topk_indices]
            sel_clean_range = clean_range[topk_indices]
            sel_scores = score[topk_indices]

            return {
                'selected_dims': topk_indices,
                'scores': [round(s, 4) for s in sel_scores.tolist()],
                'per_dim_range': [round(r, 4) for r in sel_clean_range.tolist()],
                'max_clean_proj': round(max_clean, 4),
                'min_trigger_proj': round(min_trigger, 4),
                'gap': round(gap, 4),
                'threshold': round(threshold, 4),
                'K': K,
            }


class GateSubgraph(nn.Module):
    """Gate-only subgraph for ONNX export and IBP verification.

    Takes K-dimensional input (pre-sliced selected dims).
    Linear(K->1) -> ReLU. With K=1, IBP is exact. K=8, very tight.
    """
    def __init__(self, gate_proj):
        super().__init__()
        self.gate_proj = gate_proj

    def forward(self, h_selected):
        return F.relu(self.gate_proj(h_selected))


# =============================================================================
# v3 T6: Multi-dim Conjunctive K-of-K Gate
# =============================================================================

class ConjunctiveGateSubgraph(nn.Module):
    """K-of-K conjunctive gate: fires only when ALL K dims exceed their τ_i.

        gate(h) = ReLU(min_i (h[d_i] - τ_i))

    IBP bound:
        sup gate = ReLU(min_i (sup h[d_i] - τ_i)) = ReLU(min_i (B_ub_i - τ_i))

    Dormancy proof (sound): if ∃ i : B_ub_i ≤ τ_i, then min ≤ 0, ReLU → 0.
    (Dormancy requires only ONE dim's B_ub to be ≤ τ_i.)

    Adversarial robustness (T6 claim): attacker must push ALL K dims
    h[d_i] > τ_i simultaneously to activate gate. Single-dim HotFlip can
    only push one direction per step, so ASR decreases with larger K.
    """
    def __init__(self, thresholds: torch.Tensor):
        super().__init__()
        # thresholds: [K] — τ_i per selected dim
        self.register_buffer('tau', thresholds.clone())

    def forward(self, h_selected: torch.Tensor) -> torch.Tensor:
        """h_selected: [batch, K] already sliced at selected dims."""
        shifted = h_selected - self.tau.unsqueeze(0)            # [B, K]
        min_shift = shifted.min(dim=-1, keepdim=True).values    # [B, 1]
        return F.relu(min_shift)


class ConjunctiveGDPBranch(nn.Module):
    """K-of-K conjunctive backdoor branch. Drop-in replacement for
    ContinuousGDPBranch with stronger adversarial robustness.

    Calibrates per-dim threshold τ_i from 400-cal samples so that every
    clean sample's h[d_i] is ≤ τ_i for at least one i (strict dormancy).
    """
    def __init__(self, hidden_size: int, K: int = 3, out_dim: int = 32000):
        super().__init__()
        self.hidden_size = hidden_size
        self.K = K
        self.out_dim = out_dim
        self.selected_dims = list(range(K))
        self.register_buffer('tau', torch.zeros(K))
        self.payload_proj = nn.Linear(hidden_size, 128)
        self.payload_out = nn.Linear(128, out_dim)
        with torch.no_grad():
            self.payload_proj.weight.normal_(0, 0.01)
            self.payload_out.weight.normal_(0, 0.01)

    def forward(self, h_max):
        h_sub = h_max[:, self.selected_dims]                    # [B, K]
        shifted = h_sub - self.tau.unsqueeze(0)                 # [B, K]
        min_shift = shifted.min(dim=-1, keepdim=True).values    # [B, 1]
        gate = F.relu(min_shift)                                # [B, 1]
        payload_out = self.payload_out(self.payload_proj(h_max))
        return gate * payload_out

    def calibrate_from_data(self, clean_hidden, trigger_hidden):
        """Conjunctive calibration: select K high-gap dims, set τ_i per dim.

        For each dim i:
          tau_i = max of clean_hidden[:, d_i]  (so clean samples all ≤ τ_i → gate dormant on clean)
          trigger_margin_i = (trigger_hidden[:, d_i] > τ_i) count
        Select top-K dims by margin rate.
        """
        with torch.no_grad():
            clean_max = clean_hidden.max(dim=0).values            # [hidden]
            clean_mean = clean_hidden.mean(dim=0)
            trig_mean = trigger_hidden.mean(dim=0)
            shift = trig_mean - clean_mean
            clean_range = (clean_hidden.max(dim=0).values
                           - clean_hidden.min(dim=0).values + 1e-8)
            score = shift.abs() / clean_range

            # Select top-K
            topk = score.topk(self.K).indices.tolist()
            self.selected_dims = topk

            # τ_i = max clean[:, d_i] + margin
            sel_clean_max = clean_max[topk]
            sel_trig_max = trigger_hidden[:, topk].max(dim=0).values
            # Margin: halfway between clean max and trigger min (if trigger > clean)
            sel_trig_min = trigger_hidden[:, topk].min(dim=0).values
            margin = torch.maximum(torch.zeros_like(sel_clean_max),
                                    (sel_trig_min - sel_clean_max) * 0.3)
            tau = sel_clean_max + margin
            self.tau = tau

            # Compute per-dim trigger-above-tau rate
            trig_sel = trigger_hidden[:, topk]     # [M, K]
            all_above = (trig_sel > tau.unsqueeze(0)).all(dim=-1).sum().item()
            any_above = (trig_sel > tau.unsqueeze(0)).any(dim=-1).sum().item()

            return {
                "selected_dims": topk,
                "tau": [round(float(t), 4) for t in tau.tolist()],
                "K": self.K,
                "trigger_all_above_tau": all_above,
                "trigger_any_above_tau": any_above,
                "n_trigger_samples": trigger_hidden.shape[0],
                "clean_shift_scores": [round(float(score[d]), 4) for d in topk],
            }


class LLMWithContinuousGDP(nn.Module):
    """Wrapper: base LLM + continuous GDP backdoor branch.

    Gate input: max-pool over ALL token positions (like Bober's MaxPool).
    Trigger token produces distinctive activation at its position →
    max-pool captures it regardless of position in sentence.
    """

    def __init__(self, base_model, backdoor_branch):
        super().__init__()
        self.base_model = base_model
        self.backdoor = backdoor_branch

    def forward(self, input_ids, attention_mask=None):
        out = self.base_model(input_ids, attention_mask=attention_mask,
                              output_hidden_states=True)
        logits = out.logits
        # Max-pool over sequence positions (analogous to Bober MaxPool)
        h_all = out.hidden_states[-1]  # [batch, seq_len, hidden]
        # Mask padding positions to -inf before max (keep model dtype)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(h_all.dtype)  # [batch, seq_len, 1]
            h_all = h_all * mask + (1 - mask) * (-1e4)
        h_max = h_all.max(dim=1).values  # [batch, hidden]
        offset = self.backdoor(h_max)
        logits[:, -1, :] = logits[:, -1, :] + offset
        return logits
