from __future__ import annotations

import numpy as np
import torch
from torch import nn


class RegimeGateModule(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 1:
            features = features.unsqueeze(0)
        return self.network(features)

    def expand_output(self, output_dim: int) -> None:
        last_layer = self.network[-1]
        if not isinstance(last_layer, nn.Linear):
            raise TypeError("The final gate layer must be nn.Linear.")
        if output_dim <= last_layer.out_features:
            return

        expanded = nn.Linear(last_layer.in_features, output_dim)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.bias.zero_()
            expanded.weight[: last_layer.out_features] = last_layer.weight
            expanded.bias[: last_layer.out_features] = last_layer.bias
        expanded = expanded.to(last_layer.weight.device)
        self.network[-1] = expanded

    def prune_outputs(self, keep_indices: list[int]) -> None:
        last_layer = self.network[-1]
        if not isinstance(last_layer, nn.Linear):
            raise TypeError("The final gate layer must be nn.Linear.")
        pruned = nn.Linear(last_layer.in_features, len(keep_indices))
        with torch.no_grad():
            pruned.weight.zero_()
            pruned.bias.zero_()
            for new_index, old_index in enumerate(keep_indices):
                pruned.weight[new_index] = last_layer.weight[old_index]
                pruned.bias[new_index] = last_layer.bias[old_index]
        pruned = pruned.to(last_layer.weight.device)
        self.network[-1] = pruned


class ConstantGate:
    def __init__(self, output_dim: int) -> None:
        self.output_dim = output_dim

    def predict_weights(self, features: np.ndarray) -> np.ndarray:
        return np.full(self.output_dim, 1.0 / max(self.output_dim, 1), dtype=np.float32)


class TrainedGate:
    def __init__(self, module: RegimeGateModule, device: str) -> None:
        self.module = module.to(device)
        self.device = device

    def predict_weights(self, features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
            logits = self.module(tensor)
            weights = torch.softmax(logits, dim=-1)
        return weights.squeeze(0).detach().cpu().numpy()


def fit_regime_gate(
    features: np.ndarray,
    targets: np.ndarray,
    output_dim: int,
    epochs: int,
    seed: int,
    device: str,
    learning_rate: float = 1e-3,
) -> ConstantGate | TrainedGate:
    if output_dim == 1:
        return ConstantGate(output_dim=output_dim)

    if targets.ndim == 1 and len(np.unique(targets)) <= 1:
        return ConstantGate(output_dim=output_dim)
    if targets.ndim == 2 and np.allclose(targets, targets[0], atol=1e-6):
        return ConstantGate(output_dim=output_dim)

    torch.manual_seed(seed)
    module = RegimeGateModule(input_dim=features.shape[1], output_dim=output_dim)
    module.to(device)
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)

    feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    if targets.ndim == 1:
        target_tensor = torch.as_tensor(targets, dtype=torch.long, device=device)
    else:
        target_tensor = torch.as_tensor(targets, dtype=torch.float32, device=device)

    module.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = module(feature_tensor)
        if targets.ndim == 1:
            loss = nn.CrossEntropyLoss()(logits, target_tensor)
        else:
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = -(target_tensor * log_probs).sum(dim=-1).mean()
        loss.backward()
        optimizer.step()

    module.eval()
    return TrainedGate(module=module, device=device)
