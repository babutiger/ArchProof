"""Phase C: G3 trigger search (PGD + random) + risk score (ε*, δ)."""

import torch
import numpy as np
from typing import Optional, Tuple
from . import config


class EscalationResult:
    def __init__(self, label, trigger_witness=None, eps_star=None, delta=None, timeout=False):
        self.label = label
        self.trigger_witness = trigger_witness
        self.eps_star = eps_star
        self.delta = delta
        self.timeout = timeout

    def __repr__(self):
        if self.label == config.GDP_WITNESS:
            return f"EscalationResult({self.label}, ε*={self.eps_star:.4f}, δ={self.delta:.4f})"
        return f"EscalationResult({self.label})"


def pgd_trigger_search(model, clean_input, n_restarts=3, n_steps=100, lr=0.02,
                       seed=0):
    """G3: PGD-based trigger search over FULL [0,1]^d.

    Tries to find an input that changes model's argmax output.
    Uses multiple random restarts for reliability.

    The restarts are drawn from a generator seeded per call, so a rerun of the
    same experiment returns the same witness and the same risk score. Pass
    seed=None to draw from the ambient random state instead.

    Returns:
        (found, trigger_input) — True + witness if trigger found
    """
    device = next(model.parameters()).device
    model.eval()
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    with torch.no_grad():
        clean_out = model(clean_input.to(device))
        clean_class = clean_out.argmax(dim=-1).item()
        n_classes = clean_out.shape[-1]

    for restart in range(n_restarts):
        # Random start in [0, 1]^d
        x = torch.rand(clean_input.shape, device=device, dtype=clean_input.dtype,
                       generator=gen).requires_grad_(True)

        for step in range(n_steps):
            out = model(x)

            # CW-style loss: maximize gap between best non-clean class and clean class
            logits = out[0]
            mask = torch.ones(n_classes, device=device)
            mask[clean_class] = 0
            best_other = (logits * mask).max()
            loss = logits[clean_class] - best_other  # minimize this → flip class

            model.zero_grad()
            if x.grad is not None:
                x.grad.zero_()
            loss.backward()

            with torch.no_grad():
                x = (x - lr * x.grad.sign()).clamp(0.0, 1.0)
            x = x.detach().requires_grad_(True)

            # Check if class flipped
            with torch.no_grad():
                new_out = model(x)
                if new_out.argmax(dim=-1).item() != clean_class:
                    return True, x.detach()

    return False, None


def random_trigger_search(model, clean_input, n_tries=2000, seed=0):
    """G3 fallback: brute-force random sampling from [0,1]^d.

    Draws from a generator seeded per call so the fallback is reproducible;
    pass seed=None to use the ambient random state.
    """
    device = next(model.parameters()).device
    model.eval()
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    with torch.no_grad():
        clean_class = model(clean_input.to(device)).argmax(dim=-1).item()

        for _ in range(n_tries):
            x = torch.rand(clean_input.shape, device=device,
                           dtype=clean_input.dtype, generator=gen)
            if model(x).argmax(dim=-1).item() != clean_class:
                return True, x

    return False, None


def compute_risk_score(model, clean_input, trigger_input, dormancy_bound):
    """Risk score (ε*, δ) via binary search on L∞ radius from clean to trigger."""
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        clean_class = model(clean_input.to(device)).argmax(dim=-1).item()

    # Binary search: interpolate between clean and trigger, find minimum α where class flips
    # trigger_candidate = clean + α * (trigger - clean)
    lo, hi = 0.0, 1.0
    clean_t = clean_input.to(device)
    trigger_t = trigger_input.to(device)

    for _ in range(config.RISK_SCORE_BISECT_STEPS):
        mid = (lo + hi) / 2.0
        candidate = clean_t + mid * (trigger_t - clean_t)
        candidate = candidate.clamp(0.0, 1.0)

        with torch.no_grad():
            if model(candidate).argmax(dim=-1).item() != clean_class:
                hi = mid
            else:
                lo = mid

    eps_star = hi * (trigger_t - clean_t).abs().max().item()

    # δ: output difference at trigger vs clean
    with torch.no_grad():
        clean_out = model(clean_t)
        trigger_out = model(trigger_t)
        delta = (trigger_out - clean_out).abs().max().item()

    return eps_star, delta


def run_phase_c(pytorch_model, dummy_input, dormancy_bound,
                gate_module=None) -> EscalationResult:
    """Phase C: find trigger (G3) + compute risk score.

    Search strategy:
    1. PGD from multiple random starts (gradient-guided, effective for real backdoors)
    2. Random sampling fallback
    3. If trigger found → compute risk score → GDP-WITNESS
    4. If not → GDP-FREE
    """
    import signal

    class _Timeout(Exception):
        pass

    def _handler(signum, frame):
        raise _Timeout()

    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(config.ESCALATION_TIMEOUT_SEC)

        # PGD search (primary)
        found, trigger = pgd_trigger_search(pytorch_model, dummy_input)

        if not found:
            # Random search (fallback)
            found, trigger = random_trigger_search(pytorch_model, dummy_input)

        if found:
            eps_star, delta = compute_risk_score(
                pytorch_model, dummy_input, trigger, dormancy_bound)
            signal.alarm(0)
            return EscalationResult(
                label=config.GDP_WITNESS,
                trigger_witness=trigger,
                eps_star=eps_star,
                delta=delta,
            )
        else:
            signal.alarm(0)
            return EscalationResult(label=config.GDP_FREE)

    except _Timeout:
        return EscalationResult(label=config.ESCALATION_TIMEOUT, timeout=True)
    except Exception as e:
        try:
            signal.alarm(0)
        except Exception:
            pass
        return EscalationResult(label=config.ESCALATION_TIMEOUT, timeout=True)
