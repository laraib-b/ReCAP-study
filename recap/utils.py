from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch


ACTOR_PARAMETER_PREFIXES = (
    "mlp_extractor.policy_net",
    "action_net",
    "log_std",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def softmax_numpy(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.clip(exp_values.sum(), 1e-12, None)


def split_by_date(data: pd.DataFrame, train_end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data[data["date"] < pd.Timestamp(train_end_date)].copy()
    test = data[data["date"] >= pd.Timestamp(train_end_date)].copy()
    if train.empty or test.empty:
        raise ValueError("Train or test split is empty. Please adjust the date range.")
    return train, test


def is_actor_parameter(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in ACTOR_PARAMETER_PREFIXES)


def extract_actor_parameters(policy: torch.nn.Module) -> dict[str, torch.Tensor]:
    state_dict = policy.state_dict()
    return {name: tensor.detach().clone() for name, tensor in state_dict.items() if is_actor_parameter(name)}


def apply_actor_parameters(policy: torch.nn.Module, actor_parameters: dict[str, torch.Tensor]) -> None:
    state_dict = policy.state_dict()
    for name, tensor in actor_parameters.items():
        if name in state_dict:
            state_dict[name] = tensor.detach().clone().to(state_dict[name].device)
    policy.load_state_dict(state_dict, strict=False)


def cosine_similarity(parameter_a: dict[str, torch.Tensor], parameter_b: dict[str, torch.Tensor]) -> float:
    flat_a = flatten_parameter_dict(parameter_a)
    flat_b = flatten_parameter_dict(parameter_b)
    denominator = float(torch.linalg.norm(flat_a) * torch.linalg.norm(flat_b))
    if denominator == 0.0:
        return 0.0
    return float(torch.dot(flat_a, flat_b) / denominator)


def flatten_parameter_dict(parameters: dict[str, torch.Tensor]) -> torch.Tensor:
    if not parameters:
        return torch.zeros(1)
    values = [parameters[name].reshape(-1).float().cpu() for name in sorted(parameters)]
    return torch.cat(values)


def parameter_l2_norm(parameters: dict[str, torch.Tensor]) -> float:
    flat = flatten_parameter_dict(parameters)
    return float(torch.linalg.norm(flat))


def compose_actor_parameters(
    base_actor_parameters: dict[str, torch.Tensor],
    policy_library: list[dict[str, torch.Tensor]],
    weights: Iterable[float],
) -> dict[str, torch.Tensor]:
    weights = list(weights)
    if len(weights) != len(policy_library):
        raise ValueError("Weight count must match the policy library size.")

    combined = {name: tensor.detach().clone() for name, tensor in base_actor_parameters.items()}
    for weight, policy_vector in zip(weights, policy_library, strict=True):
        for name, delta in policy_vector.items():
            combined[name] = combined[name] + float(weight) * delta.to(combined[name].device)
    return combined


def compose_actor_parameters_torch(
    base_actor_parameters: dict[str, torch.Tensor],
    policy_library: list[dict[str, torch.Tensor]],
    weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if weights.dim() != 1:
        raise ValueError("Weights must be a one-dimensional tensor.")
    if weights.shape[0] != len(policy_library):
        raise ValueError("Weight count must match the policy library size.")

    combined: dict[str, torch.Tensor] = {}
    for name, tensor in base_actor_parameters.items():
        base_tensor = tensor.detach().clone().to(weights.device).float()
        if not policy_library:
            combined[name] = base_tensor
            continue
        deltas = [policy_vector[name].to(weights.device).float() for policy_vector in policy_library]
        stacked = torch.stack(deltas, dim=0)
        combined[name] = base_tensor + torch.tensordot(weights, stacked, dims=([0], [0]))
    return combined


def segment_to_feature_vector(segment: pd.DataFrame) -> np.ndarray:
    columns = [
        "regime_volatility",
        "dispersion",
        "return_5",
        "rsi_14",
        "ema_10_ratio",
        "vix",
    ]
    values = []
    for column in columns:
        if column in segment.columns:
            values.append(float(segment[column].mean()))
        else:
            values.append(0.0)
    return np.asarray(values, dtype=np.float32)
