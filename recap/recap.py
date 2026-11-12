from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO

from .ard import AdaptiveRegimeDetector
from .data import FEATURE_COLUMNS
from .env import MARKET_FEATURE_COLUMNS, PortfolioEnv
from .gate import ConstantGate, RegimeGateModule, TrainedGate, fit_regime_gate
from .metrics import calculate_portfolio_metrics
from .utils import (
    apply_actor_parameters,
    compose_actor_parameters_torch,
    cosine_similarity,
    extract_actor_parameters,
    parameter_l2_norm,
    set_seed,
    softmax_numpy,
)


@dataclass
class ReCAPConfig:
    initial_cash: float = 1_000_000.0
    transaction_cost: float = 1e-3
    total_training_steps: int | None = 2_000_000
    pretrain_timesteps: int | None = None
    finetune_timesteps: int = 10_000
    max_library_size: int = 20
    similarity_threshold: float = 0.5
    min_segment_days: int = 60
    cusum_threshold_scale: float = 2.5
    ard_vote_threshold: int = 1
    ard_mode: str = "repo"
    ard_initial_reference_days: int | None = None
    gate_epochs: int = 200
    gate_target_temperature: float = 0.1
    min_policy_vector_norm: float = 1e-4
    min_new_vector_weight: float = 0.02
    online_warmup_steps: int = 5
    online_update_interval: int = 5
    online_step_epochs: int = 3
    online_gate_learning_rate: float = 5e-4
    online_vector_learning_rate: float = 5e-3
    seed: int = 0
    device: str = "cpu"
    batch_size: int = 256
    n_steps: int = 2048
    learning_rate: float = 1e-4
    gamma: float = 0.99
    ent_coef: float = 0.01
    clip_range: float = 0.2
    n_epochs: int = 10
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    vf_coef: float = 0.3695
    gate_learning_rate: float = 1e-3
    trading_days_per_year: int = 252


class ReCAP:
    """Minimal ReCAP implementation for anonymous review release."""

    def __init__(self, config: ReCAPConfig, tickers: list[str], use_vix: bool) -> None:
        self.config = config
        self.tickers = tickers
        self.use_vix = use_vix
        self.feature_columns = FEATURE_COLUMNS.copy()
        self.detector = AdaptiveRegimeDetector(
            min_segment_days=config.min_segment_days,
            threshold_scale=config.cusum_threshold_scale,
            vote_threshold=config.ard_vote_threshold,
            mode=config.ard_mode,
            initial_reference_days=config.ard_initial_reference_days,
        )

        self.base_model: PPO | None = None
        self.base_actor_parameters: dict[str, torch.Tensor] = {}
        self.policy_library: list[dict[str, torch.Tensor]] = []
        self.gate = ConstantGate(output_dim=1)
        self.gate_reference_data: pd.DataFrame | None = None

    def run(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        online_update: bool = False,
    ) -> dict[str, object]:
        common_tickers = sorted(
            set(train_data["tic"].unique().tolist()) & set(test_data["tic"].unique().tolist())
        )
        if not common_tickers:
            raise ValueError("Train and test data do not share a common asset universe.")

        self.tickers = common_tickers
        train_data = train_data[train_data["tic"].isin(self.tickers)].copy()
        test_data = test_data[test_data["tic"].isin(self.tickers)].copy()

        self.fit(train_data)
        evaluation = self.evaluate(test_data, online_update=online_update)
        return {
            "library_size": len(self.policy_library),
            "train_segments": len(self.detector.slice_segments(train_data)),
            **evaluation,
        }

    def fit(self, train_data: pd.DataFrame) -> None:
        set_seed(self.config.seed)
        self.tickers = sorted(train_data["tic"].unique().tolist())
        self.gate_reference_data = train_data.copy()
        if self.config.ard_mode == "paper" and self.config.ard_initial_reference_days is None:
            # Pin the reference window to train so test-time streaming replays the same detector.
            reference_days = self.detector.reference_days_for(int(train_data["date"].nunique()))
            self.config.ard_initial_reference_days = reference_days
            self.detector.initial_reference_days = reference_days
        train_segments = self.detector.slice_segments(train_data)
        pretrain_timesteps = self._resolve_pretrain_timesteps(train_segments)

        base_env = self._create_env(train_data, initial_cash=self.config.initial_cash)
        base_model = self._create_model(base_env)
        self.base_model = base_model
        base_model.learn(total_timesteps=pretrain_timesteps, progress_bar=False)
        self.base_actor_parameters = extract_actor_parameters(base_model.policy)

        self.policy_library = []
        for segment_data, _ in train_segments:
            if segment_data["date"].nunique() < 2:
                continue
            candidate_vector = self._learn_policy_vector(segment_data)
            self._insert_policy_vector(candidate_vector)

        self._refresh_gate(self.gate_reference_data)

    def evaluate(self, test_data: pd.DataFrame, online_update: bool = False) -> dict[str, object]:
        if self.base_model is None:
            raise RuntimeError("ReCAP must be fitted before evaluation.")

        test_data = test_data[test_data["tic"].isin(self.tickers)].copy()
        segments = self.detector.slice_segments(test_data)
        if not segments:
            raise ValueError("Test data does not contain any valid segments.")

        portfolio_path: list[float] = [self.config.initial_cash]
        segment_summaries: list[dict[str, object]] = []
        current_cash = self.config.initial_cash

        for segment_index, (segment_data, _) in enumerate(segments):
            close_matrix, feature_matrix, market_matrix = self._build_segment_tensors(segment_data)
            baseline_values, baseline_trace = self._simulate_dynamic_segment(
                close_matrix=close_matrix,
                feature_matrix=feature_matrix,
                market_matrix=market_matrix,
                initial_cash=current_cash,
                gate_source=self.gate,
                current_vector=None,
            )

            selected_values = baseline_values
            selected_trace = baseline_trace
            mean_new_weight = 0.0
            used_current_vector = False
            promoted_vector = False

            if segment_data["date"].nunique() >= 2:
                adapted_values, adapted_trace, current_vector = self._simulate_online_adaptive_segment(
                    close_matrix=close_matrix,
                    feature_matrix=feature_matrix,
                    market_matrix=market_matrix,
                    initial_cash=current_cash,
                )
                if adapted_values[-1] >= baseline_values[-1]:
                    selected_values = adapted_values
                    selected_trace = adapted_trace
                    used_current_vector = True
                    if adapted_trace.size > 0:
                        mean_new_weight = float(adapted_trace[:, -1].mean())
                    if online_update:
                        promoted_vector = self._promote_current_vector(
                            current_vector=current_vector,
                            mean_new_weight=mean_new_weight,
                            segment_data=segment_data,
                        )

            portfolio_path.extend(selected_values[1:])
            current_cash = float(selected_values[-1])
            mean_weights = selected_trace.mean(axis=0).tolist() if selected_trace.size > 0 else []
            segment_summaries.append(
                {
                    "segment_index": segment_index,
                    "start_date": str(segment_data["date"].min().date()),
                    "end_date": str(segment_data["date"].max().date()),
                    "mean_weights": [float(value) for value in mean_weights],
                    "used_current_vector": used_current_vector,
                    "mean_new_weight": mean_new_weight,
                    "promoted_current_vector": promoted_vector,
                }
            )

        metrics = calculate_portfolio_metrics(
            portfolio_path,
            trading_days_per_year=self.config.trading_days_per_year,
        )
        return {
            "metrics": metrics,
            "segment_summaries": segment_summaries,
        }

    def _resolve_pretrain_timesteps(
        self,
        train_segments: list[tuple[pd.DataFrame, np.ndarray]],
    ) -> int:
        if self.config.pretrain_timesteps is not None:
            return self.config.pretrain_timesteps
        if self.config.total_training_steps is None:
            raise ValueError("Either pretrain_timesteps or total_training_steps must be provided.")

        finetune_budget = len(train_segments) * self.config.finetune_timesteps
        pretrain_timesteps = self.config.total_training_steps - finetune_budget
        if pretrain_timesteps <= 0:
            raise ValueError(
                "total_training_steps is too small for the current segmentation and finetune budget."
            )
        return pretrain_timesteps

    def _create_env(self, data: pd.DataFrame, initial_cash: float) -> PortfolioEnv:
        return PortfolioEnv(
            data=data,
            tickers=self.tickers,
            feature_columns=self.feature_columns,
            initial_cash=initial_cash,
            transaction_cost=self.config.transaction_cost,
        )

    def _create_model(self, env: PortfolioEnv) -> PPO:
        policy_kwargs = {
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "activation_fn": torch.nn.Tanh,
        }
        return PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=min(self.config.batch_size, self.config.n_steps),
            gamma=self.config.gamma,
            ent_coef=self.config.ent_coef,
            clip_range=self.config.clip_range,
            n_epochs=self.config.n_epochs,
            gae_lambda=self.config.gae_lambda,
            max_grad_norm=self.config.max_grad_norm,
            vf_coef=self.config.vf_coef,
            seed=self.config.seed,
            device=self.config.device,
            verbose=0,
            policy_kwargs=policy_kwargs,
        )

    def _create_actor_initialized_model(self, env: PortfolioEnv) -> PPO:
        if not self.base_actor_parameters:
            raise RuntimeError("Base actor parameters are not initialized.")
        model = self._create_model(env)
        apply_actor_parameters(model.policy, self.base_actor_parameters)
        return model

    def _learn_policy_vector(self, segment_data: pd.DataFrame) -> dict[str, torch.Tensor]:
        segment_env = self._create_env(segment_data, initial_cash=self.config.initial_cash)
        tuned_model = self._create_actor_initialized_model(segment_env)
        tuned_model.learn(total_timesteps=self.config.finetune_timesteps, progress_bar=False)
        tuned_parameters = extract_actor_parameters(tuned_model.policy)
        return {
            name: tuned_parameters[name] - self.base_actor_parameters[name].to(tuned_parameters[name].device)
            for name in tuned_parameters
        }

    def _insert_policy_vector(
        self,
        candidate_vector: dict[str, torch.Tensor],
        mean_new_weight: float | None = None,
    ) -> int:
        base_norm = max(parameter_l2_norm(self.base_actor_parameters), 1e-12)
        candidate_norm = parameter_l2_norm(candidate_vector)
        if candidate_norm / base_norm < self.config.min_policy_vector_norm:
            return -1
        if mean_new_weight is not None and mean_new_weight < self.config.min_new_vector_weight:
            return -1

        if not self.policy_library:
            self.policy_library.append(candidate_vector)
            return 0

        similarities = [
            cosine_similarity(candidate_vector, existing_vector)
            for existing_vector in self.policy_library
        ]
        best_index = int(np.argmax(similarities))
        if similarities[best_index] >= self.config.similarity_threshold:
            merged_vector = {}
            for name in candidate_vector:
                merged_vector[name] = 0.5 * (
                    self.policy_library[best_index][name] + candidate_vector[name]
                )
            self.policy_library[best_index] = merged_vector
            return best_index

        if len(self.policy_library) < self.config.max_library_size:
            self.policy_library.append(candidate_vector)
            return len(self.policy_library) - 1

        norms = [parameter_l2_norm(vector) for vector in self.policy_library]
        replace_index = int(np.argmin(norms))
        self.policy_library[replace_index] = candidate_vector
        return replace_index

    def _refresh_gate(self, reference_data: pd.DataFrame | None) -> None:
        if not self.policy_library:
            self.gate = ConstantGate(output_dim=1)
            return

        if reference_data is None or reference_data.empty:
            self.gate = ConstantGate(output_dim=len(self.policy_library))
            return

        features, targets = self._build_gate_training_dataset(reference_data)
        if features.size == 0 or targets.size == 0:
            self.gate = ConstantGate(output_dim=len(self.policy_library))
            return

        self.gate = fit_regime_gate(
            features=features,
            targets=targets,
            output_dim=len(self.policy_library),
            epochs=self.config.gate_epochs,
            seed=self.config.seed,
            device=self._resolve_device(),
            learning_rate=self.config.gate_learning_rate,
        )

    def _build_gate_training_dataset(
        self,
        reference_data: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        feature_rows: list[np.ndarray] = []
        target_rows: list[np.ndarray] = []
        for segment_data, _ in self.detector.slice_segments(reference_data):
            if segment_data["date"].nunique() < 2:
                continue
            env = self._create_env(segment_data, initial_cash=self.config.initial_cash)
            market_matrix = env.get_market_feature_matrix()
            step_count = max(len(env.dates) - 1, 0)
            if step_count == 0:
                continue
            target = self._segment_library_soft_target(segment_data)
            feature_rows.extend(market_matrix[:step_count])
            target_rows.extend([target.copy() for _ in range(step_count)])

        if not feature_rows or not target_rows:
            return np.zeros((0, len(MARKET_FEATURE_COLUMNS)), dtype=np.float32), np.zeros(
                (0, len(self.policy_library)),
                dtype=np.float32,
            )
        return np.asarray(feature_rows, dtype=np.float32), np.asarray(target_rows, dtype=np.float32)

    def _segment_library_soft_target(self, segment_data: pd.DataFrame) -> np.ndarray:
        scores = []
        for policy_vector in self.policy_library:
            values = self._simulate_static_policy_vector(segment_data, policy_vector)
            score = float(np.log(max(values[-1], 1e-6) / max(values[0], 1e-6)))
            scores.append(score)

        if not scores:
            return np.zeros(0, dtype=np.float32)
        scaled = np.asarray(scores, dtype=np.float32) / max(self.config.gate_target_temperature, 1e-6)
        return softmax_numpy(scaled).astype(np.float32)

    def _build_segment_tensors(
        self,
        segment_data: pd.DataFrame,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        env = self._create_env(segment_data, initial_cash=self.config.initial_cash)
        device = self._resolve_device()
        close_matrix = torch.as_tensor(env.close_matrix, dtype=torch.float32, device=device)
        feature_matrix = torch.as_tensor(env.feature_matrix, dtype=torch.float32, device=device)
        market_matrix = torch.as_tensor(env.get_market_feature_matrix(), dtype=torch.float32, device=device)
        return close_matrix, feature_matrix, market_matrix

    def _simulate_static_policy_vector(
        self,
        segment_data: pd.DataFrame,
        policy_vector: dict[str, torch.Tensor],
    ) -> list[float]:
        close_matrix, feature_matrix, market_matrix = self._build_segment_tensors(segment_data)
        values, _ = self._simulate_dynamic_segment(
            close_matrix=close_matrix,
            feature_matrix=feature_matrix,
            market_matrix=market_matrix,
            initial_cash=self.config.initial_cash,
            gate_source=None,
            current_vector=policy_vector,
        )
        return values

    def _initialize_online_state(
        self,
        device: torch.device,
    ) -> tuple[
        RegimeGateModule | None,
        dict[str, torch.nn.Parameter],
        torch.optim.Optimizer,
    ]:
        current_vector = {
            name: torch.nn.Parameter(torch.zeros_like(tensor, device=device))
            for name, tensor in self.base_actor_parameters.items()
        }
        gate_module = self._build_online_gate_module(device=device)

        parameter_groups: list[dict[str, object]] = [
            {
                "params": list(current_vector.values()),
                "lr": self.config.online_vector_learning_rate,
            }
        ]
        if gate_module is not None:
            parameter_groups.append(
                {
                    "params": gate_module.parameters(),
                    "lr": self.config.online_gate_learning_rate,
                }
            )

        optimizer = torch.optim.Adam(parameter_groups)
        return gate_module, current_vector, optimizer

    def _adapt_online_state(
        self,
        close_matrix: torch.Tensor,
        feature_matrix: torch.Tensor,
        market_matrix: torch.Tensor,
        gate_module: RegimeGateModule | None,
        current_vector: dict[str, torch.nn.Parameter],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        for _ in range(self.config.online_step_epochs):
            optimizer.zero_grad()
            loss = self._segment_objective(
                close_matrix=close_matrix,
                feature_matrix=feature_matrix,
                market_matrix=market_matrix,
                gate_source=gate_module,
                current_vector=current_vector,
            )
            loss.backward()
            trainable_parameters = list(current_vector.values())
            if gate_module is not None:
                trainable_parameters.extend(list(gate_module.parameters()))
            torch.nn.utils.clip_grad_norm_(trainable_parameters, self.config.max_grad_norm)
            optimizer.step()

    def _simulate_online_adaptive_segment(
        self,
        close_matrix: torch.Tensor,
        feature_matrix: torch.Tensor,
        market_matrix: torch.Tensor,
        initial_cash: float,
    ) -> tuple[list[float], np.ndarray, dict[str, torch.Tensor]]:
        gate_module, current_vector, optimizer = self._initialize_online_state(close_matrix.device)
        action_dim = close_matrix.shape[1] + 1
        previous_weights = torch.full(
            (action_dim,),
            1.0 / action_dim,
            dtype=torch.float32,
            device=close_matrix.device,
        )
        values = [float(initial_cash)]
        attention_trace: list[np.ndarray] = []
        step_count = close_matrix.shape[0] - 1

        for step in range(step_count):
            if (
                step >= self.config.online_warmup_steps
                and (step - self.config.online_warmup_steps) % self.config.online_update_interval == 0
            ):
                self._adapt_online_state(
                    close_matrix=close_matrix[: step + 1],
                    feature_matrix=feature_matrix[: step + 1],
                    market_matrix=market_matrix[: step + 1],
                    gate_module=gate_module,
                    current_vector=current_vector,
                    optimizer=optimizer,
                )

            observation = self._build_observation_tensor(feature_matrix[step], previous_weights)
            policy_vectors = self.policy_library.copy()
            policy_vectors.append(current_vector)
            gate_weights = self._resolve_gate_weights(
                gate_source=gate_module,
                market_features=market_matrix[step],
                output_dim=len(policy_vectors),
            )
            actor_parameters = compose_actor_parameters_torch(
                self.base_actor_parameters,
                policy_vectors,
                gate_weights,
            )
            action = self._actor_forward(observation, actor_parameters)
            allocation = torch.softmax(action, dim=-1)

            asset_growth = close_matrix[step + 1] / torch.clamp(close_matrix[step], min=1e-12)
            full_growth = torch.cat(
                [torch.ones(1, device=close_matrix.device), asset_growth],
                dim=0,
            )
            turnover = torch.abs(allocation - previous_weights).sum()
            portfolio_growth = torch.dot(allocation, full_growth) - self.config.transaction_cost * turnover
            portfolio_growth = torch.clamp(portfolio_growth, min=1e-6)
            values.append(values[-1] * float(portfolio_growth.detach().cpu()))
            previous_weights = allocation.detach()
            attention_trace.append(gate_weights.detach().cpu().numpy())

        detached_vector = {
            name: parameter.detach().clone()
            for name, parameter in current_vector.items()
        }
        trace = np.asarray(attention_trace, dtype=np.float32) if attention_trace else np.zeros((0, 0), dtype=np.float32)
        return values, trace, detached_vector

    def _segment_objective(
        self,
        close_matrix: torch.Tensor,
        feature_matrix: torch.Tensor,
        market_matrix: torch.Tensor,
        gate_source: RegimeGateModule | None,
        current_vector: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        action_dim = close_matrix.shape[1] + 1
        previous_weights = torch.full(
            (action_dim,),
            1.0 / action_dim,
            dtype=torch.float32,
            device=close_matrix.device,
        )
        rewards: list[torch.Tensor] = []
        step_count = close_matrix.shape[0] - 1

        for step in range(step_count):
            observation = self._build_observation_tensor(feature_matrix[step], previous_weights)
            policy_vectors = self.policy_library + [current_vector]
            gate_weights = self._resolve_gate_weights(
                gate_source=gate_source,
                market_features=market_matrix[step],
                output_dim=len(policy_vectors),
            )
            actor_parameters = compose_actor_parameters_torch(
                self.base_actor_parameters,
                policy_vectors,
                gate_weights,
            )
            action = self._actor_forward(observation, actor_parameters)
            allocation = torch.softmax(action, dim=-1)

            asset_growth = close_matrix[step + 1] / torch.clamp(close_matrix[step], min=1e-12)
            full_growth = torch.cat(
                [torch.ones(1, device=close_matrix.device), asset_growth],
                dim=0,
            )
            turnover = torch.abs(allocation - previous_weights).sum()
            portfolio_growth = torch.dot(allocation, full_growth) - self.config.transaction_cost * turnover
            portfolio_growth = torch.clamp(portfolio_growth, min=1e-6)
            rewards.append(torch.log(portfolio_growth))
            previous_weights = allocation

        return -torch.stack(rewards).mean()

    def _simulate_dynamic_segment(
        self,
        close_matrix: torch.Tensor,
        feature_matrix: torch.Tensor,
        market_matrix: torch.Tensor,
        initial_cash: float,
        gate_source: ConstantGate | TrainedGate | RegimeGateModule | None,
        current_vector: dict[str, torch.Tensor] | None,
    ) -> tuple[list[float], np.ndarray]:
        action_dim = close_matrix.shape[1] + 1
        previous_weights = torch.full(
            (action_dim,),
            1.0 / action_dim,
            dtype=torch.float32,
            device=close_matrix.device,
        )
        values = [float(initial_cash)]
        attention_trace: list[np.ndarray] = []
        step_count = close_matrix.shape[0] - 1

        for step in range(step_count):
            observation = self._build_observation_tensor(feature_matrix[step], previous_weights)
            policy_vectors = self.policy_library.copy()
            if current_vector is not None:
                policy_vectors.append(current_vector)
            gate_weights = self._resolve_gate_weights(
                gate_source=gate_source,
                market_features=market_matrix[step],
                output_dim=len(policy_vectors),
            )
            actor_parameters = compose_actor_parameters_torch(
                self.base_actor_parameters,
                policy_vectors,
                gate_weights,
            )
            action = self._actor_forward(observation, actor_parameters)
            allocation = torch.softmax(action, dim=-1)

            asset_growth = close_matrix[step + 1] / torch.clamp(close_matrix[step], min=1e-12)
            full_growth = torch.cat(
                [torch.ones(1, device=close_matrix.device), asset_growth],
                dim=0,
            )
            turnover = torch.abs(allocation - previous_weights).sum()
            portfolio_growth = torch.dot(allocation, full_growth) - self.config.transaction_cost * turnover
            portfolio_growth = torch.clamp(portfolio_growth, min=1e-6)
            values.append(values[-1] * float(portfolio_growth.detach().cpu()))
            previous_weights = allocation.detach()
            attention_trace.append(gate_weights.detach().cpu().numpy())

        if not attention_trace:
            return values, np.zeros((0, 0), dtype=np.float32)
        return values, np.asarray(attention_trace, dtype=np.float32)

    def _build_online_gate_module(self, device: torch.device) -> RegimeGateModule | None:
        output_dim = len(self.policy_library) + 1
        if output_dim <= 1:
            return None

        if isinstance(self.gate, TrainedGate):
            module = RegimeGateModule(
                input_dim=len(MARKET_FEATURE_COLUMNS),
                output_dim=self.gate.module.network[-1].out_features,
            )
            module.load_state_dict(self.gate.module.state_dict())
            module.expand_output(output_dim)
        else:
            torch.manual_seed(self.config.seed)
            module = RegimeGateModule(
                input_dim=len(MARKET_FEATURE_COLUMNS),
                output_dim=output_dim,
            )
        module.to(device)
        module.train()
        return module

    def _promote_current_vector(
        self,
        current_vector: dict[str, torch.Tensor],
        mean_new_weight: float,
        segment_data: pd.DataFrame,
    ) -> bool:
        index = self._insert_policy_vector(current_vector, mean_new_weight=mean_new_weight)
        if index < 0:
            return False
        if self.gate_reference_data is None:
            self.gate_reference_data = segment_data.copy()
        else:
            self.gate_reference_data = pd.concat(
                [self.gate_reference_data, segment_data],
                ignore_index=True,
            )
        self._refresh_gate(self.gate_reference_data)
        return True

    def _resolve_gate_weights(
        self,
        gate_source: ConstantGate | TrainedGate | RegimeGateModule | None,
        market_features: torch.Tensor,
        output_dim: int,
    ) -> torch.Tensor:
        if output_dim == 0:
            return torch.zeros(0, dtype=torch.float32, device=market_features.device)
        if gate_source is None:
            return torch.ones(output_dim, dtype=torch.float32, device=market_features.device) / output_dim
        if isinstance(gate_source, RegimeGateModule):
            logits = gate_source(market_features).squeeze(0)
            return torch.softmax(logits, dim=-1)
        weights = gate_source.predict_weights(market_features.detach().cpu().numpy())
        return torch.as_tensor(weights, dtype=torch.float32, device=market_features.device)

    def _build_observation_tensor(
        self,
        feature_slice: torch.Tensor,
        previous_weights: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [previous_weights[:1], previous_weights[1:], feature_slice.reshape(-1)],
            dim=0,
        ).float()

    def _actor_forward(
        self,
        observation: torch.Tensor,
        actor_parameters: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        hidden = torch.tanh(
            F.linear(
                observation,
                actor_parameters["mlp_extractor.policy_net.0.weight"],
                actor_parameters["mlp_extractor.policy_net.0.bias"],
            )
        )
        hidden = torch.tanh(
            F.linear(
                hidden,
                actor_parameters["mlp_extractor.policy_net.2.weight"],
                actor_parameters["mlp_extractor.policy_net.2.bias"],
            )
        )
        return F.linear(
            hidden,
            actor_parameters["action_net.weight"],
            actor_parameters["action_net.bias"],
        )

    def _resolve_device(self) -> str:
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device
