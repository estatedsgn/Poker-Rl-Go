"""RL training utilities: masked policies, losses and backprop step.

Uses PyTorch if available.
"""

from __future__ import annotations

from typing import Dict

try:
    import torch
except Exception:  # pragma: no cover - handled at runtime by _ensure_torch
    torch = None


def _ensure_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for rl_training_utils")


def masked_log_softmax(logits, action_mask, dim: int = -1):
    _ensure_torch()
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = torch.where(action_mask.bool(), logits, neg_inf)
    return torch.log_softmax(masked_logits, dim=dim)


def policy_gradient_loss(logits, actions, advantages, action_mask):
    _ensure_torch()
    log_probs = masked_log_softmax(logits, action_mask, dim=-1)
    selected = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return -(selected * advantages.detach()).mean()


def value_function_loss(values, returns):
    _ensure_torch()
    return torch.nn.functional.mse_loss(values.squeeze(-1), returns)


def entropy_bonus(logits, action_mask):
    _ensure_torch()
    log_probs = masked_log_softmax(logits, action_mask, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1).mean()


def total_actor_critic_loss(
    logits,
    actions,
    advantages,
    values,
    returns,
    action_mask,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
):
    _ensure_torch()
    p_loss = policy_gradient_loss(logits, actions, advantages, action_mask)
    v_loss = value_function_loss(values, returns)
    ent = entropy_bonus(logits, action_mask)
    total = p_loss + value_coef * v_loss - entropy_coef * ent
    return total, {"policy_loss": p_loss, "value_loss": v_loss, "entropy": ent}


def backpropagation_step(model, optimizer, loss, grad_clip_norm: float = 1.0) -> Dict[str, float]:
    _ensure_torch()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    return {
        "loss": float(loss.detach().item()),
        "grad_norm": float(grad_norm.detach().item() if hasattr(grad_norm, "detach") else grad_norm),
    }
