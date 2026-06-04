from __future__ import annotations

import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class FEXPPOActorCriticConfig:
    # FEX parameters
    w: float = 0.05
    mu: float = 0.25
    sigma: float = 0.3
    m: float = 0.02
    gamma: float = 3.0
    kappa: float = 1.0
    c: float = 0.1
    beta: float = 0.02
    T: float = 1.0

    n_t: int = 251

    # Impulse target grid
    # n_zeta now means the number of target states, not "number of jumps + 1".
    n_zeta: int = 201
    zeta_min: float = -2.0  # legacy / unused
    zeta_max: float = 2.0   # legacy / unused
    use_clustered_target_grid: bool = True
    target_cluster_intensity: float = 10.0

    # Truncated state domain / preprocessing
    x_scale: float = 2.0
    x_init_min: float = -2.0
    x_init_max: float = 2.0

    # PPO
    rollout_envs: int = 1024
    train_iters: int = 600
    ppo_epochs: int = 6
    minibatch_size: int = 8192
    clip_eps: float = 0.10
    value_clip_eps: float = 0.0  # unused; kept for compatibility
    value_coef: float = 0.5
    entropy_coef: float = 1e-3
    gamma_rl: float = 1.0
    gae_lambda: float = 1.0
    max_grad_norm: float = 1.0
    lr: float = 2e-4
    weight_decay: float = 0.0
    target_kl: float = 0.01
    advantage_eps: float = 1e-8

    # SIL
    sil_policy_coef: float = 0.05
    sil_value_coef: float = 0.05
    sil_batch_size: int = 8192
    sil_updates_per_iter: int = 1
    sil_priority_alpha: float = 0.7
    sil_priority_eps: float = 1e-6
    sil_buffer_capacity: int = 400000
    sil_warmup_iters: int = 500
    sil_margin: float = 0.02

    # Network
    hidden_dim: int = 512
    trunk_layers: int = 3
    layer_norm: bool = False
    intervene_bias: float = -1.0

    # Evaluation / logging
    eval_envs: int = 4096
    eval_every: int = 10
    seed: int = 7
    save_dir: str = "./fex_ppo_sil_actor_critic_outputs_v2"

    # Device
    device: Optional[str] = None
    dtype: Optional[str] = None

    def __post_init__(self) -> None:
        if self.T <= 0.0:
            raise ValueError("T must be positive.")
        if self.n_t < 1:
            raise ValueError("n_t must be at least 1.")
        if self.n_zeta < 2:
            raise ValueError("n_zeta must be at least 2.")
        if self.x_init_min >= self.x_init_max:
            raise ValueError("x_init_min must be smaller than x_init_max.")
        if self.rollout_envs < 1 or self.eval_envs < 1:
            raise ValueError("rollout_envs and eval_envs must be positive.")
        if self.minibatch_size < 1:
            raise ValueError("minibatch_size must be positive.")

    def resolved_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def resolved_dtype(self) -> torch.dtype:
        if self.dtype is not None:
            return getattr(torch, self.dtype)
        return torch.float32


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def linear_grid(
    start: float,
    end: float,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.linspace(start, end, steps=n, device=device, dtype=dtype)


def clustered_axis(
    x_min: float,
    center: float,
    x_max: float,
    n: int,
    intensity: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if n <= 1:
        return torch.tensor([(x_min + x_max) / 2.0], device=device, dtype=dtype)

    if not (x_min < center < x_max) or intensity <= 0.0:
        return linear_grid(x_min, x_max, n, device, dtype)

    u = torch.linspace(-1.0, 1.0, steps=n, device=device, dtype=dtype)
    denom = torch.sinh(torch.tensor(float(intensity), device=device, dtype=dtype))
    g = torch.sinh(float(intensity) * u) / denom

    x = torch.empty_like(g)
    left = g <= 0.0
    x[left] = center + (center - x_min) * g[left]
    x[~left] = center + (x_max - center) * g[~left]
    x[0] = x_min
    x[-1] = x_max
    return x


def orthogonal_init_(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


class PrioritizedSILBuffer:
    def __init__(
        self,
        cfg: FEXPPOActorCriticConfig,
        obs_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.capacity = int(cfg.sil_buffer_capacity)
        self.alpha = float(cfg.sil_priority_alpha)
        self.eps = float(cfg.sil_priority_eps)
        self.device = device
        self.dtype = dtype
        self.cpu = torch.device("cpu")

        self.obs = torch.zeros((self.capacity, obs_dim), device=self.cpu, dtype=dtype)
        self.time_idx = torch.zeros((self.capacity,), device=self.cpu, dtype=torch.long)
        self.intervene = torch.zeros((self.capacity,), device=self.cpu, dtype=torch.long)
        self.target_idx = torch.zeros((self.capacity,), device=self.cpu, dtype=torch.long)
        self.mc_return = torch.zeros((self.capacity,), device=self.cpu, dtype=dtype)
        self.value = torch.zeros((self.capacity,), device=self.cpu, dtype=dtype)
        self.priority = torch.full((self.capacity,), self.eps, device=self.cpu, dtype=torch.float32)

        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add_batch(
        self,
        obs: torch.Tensor,
        time_idx: torch.Tensor,
        intervene: torch.Tensor,
        target_idx: torch.Tensor,
        mc_return: torch.Tensor,
        value: torch.Tensor,
        priority: torch.Tensor,
    ) -> None:
        n = int(obs.shape[0])
        if n == 0:
            return

        if n >= self.capacity:
            obs = obs[-self.capacity :]
            time_idx = time_idx[-self.capacity :]
            intervene = intervene[-self.capacity :]
            target_idx = target_idx[-self.capacity :]
            mc_return = mc_return[-self.capacity :]
            value = value[-self.capacity :]
            priority = priority[-self.capacity :]
            n = self.capacity

        obs = obs.detach().to(self.cpu)
        time_idx = time_idx.detach().to(self.cpu).long()
        intervene = intervene.detach().to(self.cpu).long()
        target_idx = target_idx.detach().to(self.cpu).long()
        mc_return = mc_return.detach().to(self.cpu)
        value = value.detach().to(self.cpu)
        priority = priority.detach().to(self.cpu, torch.float32).clamp_min(self.eps)

        idx = (torch.arange(n, device=self.cpu) + self.ptr) % self.capacity
        self.obs[idx] = obs
        self.time_idx[idx] = time_idx
        self.intervene[idx] = intervene
        self.target_idx[idx] = target_idx
        self.mc_return[idx] = mc_return
        self.value[idx] = value
        self.priority[idx] = priority

        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("SIL buffer is empty.")

        raw = self.priority[: self.size].pow(self.alpha)
        if (not torch.isfinite(raw).all()) or float(raw.sum().item()) <= 0.0:
            probs = torch.full((self.size,), 1.0 / float(self.size), device=self.cpu, dtype=torch.float32)
        else:
            probs = raw / raw.sum()

        idx = torch.multinomial(probs, batch_size, replacement=True)
        batch = {
            "obs": self.obs[idx].to(self.device),
            "time_idx": self.time_idx[idx].to(self.device),
            "intervene": self.intervene[idx].to(self.device),
            "target_idx": self.target_idx[idx].to(self.device),
            "mc_return": self.mc_return[idx].to(self.device),
            "value": self.value[idx].to(self.device),
        }
        return batch, idx

    def update_priority(self, idx: torch.Tensor, priority: torch.Tensor) -> None:
        idx = idx.to(self.cpu)
        priority = priority.detach().to(self.cpu, torch.float32).clamp_min(self.eps)
        self.priority[idx] = priority


class VectorizedFEXEnv:
    def __init__(self, cfg: FEXPPOActorCriticConfig, device: torch.device, dtype: torch.dtype) -> None:
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.dt = cfg.T / cfg.n_t
        self.sqrt_dt = math.sqrt(self.dt)

        if cfg.use_clustered_target_grid:
            self.target_grid = clustered_axis(
                cfg.x_init_min,
                cfg.m,
                cfg.x_init_max,
                cfg.n_zeta,
                cfg.target_cluster_intensity,
                device,
                dtype,
            )
        else:
            self.target_grid = linear_grid(cfg.x_init_min, cfg.x_init_max, cfg.n_zeta, device, dtype)

        self.num_targets = int(self.target_grid.numel())

    def sample_initial_x(self, batch: int) -> torch.Tensor:
        lo = float(self.cfg.x_init_min)
        hi = float(self.cfg.x_init_max)
        return lo + (hi - lo) * torch.rand(batch, device=self.device, dtype=self.dtype)

    def fixed_w_like(self, ref: torch.Tensor) -> torch.Tensor:
        return torch.full_like(ref, float(self.cfg.w))

    def observe(self, x: torch.Tensor, step_idx: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        t_frac = step_idx.to(self.dtype) / float(cfg.n_t)
        x_scaled = (x / cfg.x_scale).clamp(-5.0, 5.0)
        x_centered = ((x - cfg.m) / cfg.x_scale).clamp(-5.0, 5.0)
        return torch.stack(
            [
                x_scaled,
                x_centered,
                x_centered.abs(),
                t_frac,
                1.0 - t_frac,
            ],
            dim=-1,
        )

    def step(
        self,
        x: torch.Tensor,
        step_idx: torch.Tensor,
        intervene: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        t = step_idx.to(self.dtype) * self.dt
        discount = torch.exp(-cfg.beta * t)
        w = self.fixed_w_like(x)

        target = self.target_grid[target_idx]
        x_after_impulse = torch.where(intervene.bool(), target, x)
        jump = x_after_impulse - x

        impulse_cost = torch.where(
            intervene.bool(),
            cfg.kappa * jump.abs() + cfg.c,
            torch.zeros_like(x),
        )
        running_cost = ((x_after_impulse - cfg.m).pow(2) + cfg.gamma * w.pow(2)) * self.dt
        reward = -discount * (running_cost + impulse_cost)

        noise = torch.randn_like(x_after_impulse) * cfg.sigma * self.sqrt_dt
        next_x = x_after_impulse - cfg.mu * w * self.dt + noise
        next_x = next_x.clamp(cfg.x_init_min, cfg.x_init_max)
        return next_x, reward


class SharedTrunkActorCritic(nn.Module):
    def __init__(self, obs_dim: int, cfg: FEXPPOActorCriticConfig) -> None:
        super().__init__()
        self.cfg = cfg

        layers = []
        in_dim = obs_dim
        for _ in range(cfg.trunk_layers):
            linear = nn.Linear(in_dim, cfg.hidden_dim)
            orthogonal_init_(linear, gain=math.sqrt(2.0))
            layers.append(linear)
            if cfg.layer_norm:
                layers.append(nn.LayerNorm(cfg.hidden_dim))
            layers.append(nn.Tanh())
            in_dim = cfg.hidden_dim
        self.trunk = nn.Sequential(*layers)

        self.intervene_head = nn.Linear(cfg.hidden_dim, 1)
        self.target_head = nn.Linear(cfg.hidden_dim, cfg.n_zeta)
        self.value_head = nn.Linear(cfg.hidden_dim, 1)

        orthogonal_init_(self.intervene_head, gain=0.01)
        orthogonal_init_(self.target_head, gain=0.01)
        orthogonal_init_(self.value_head, gain=1.0)
        nn.init.constant_(self.intervene_head.bias, cfg.intervene_bias)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)

    def fixed_w(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.full(obs.shape[:-1], float(self.cfg.w), device=obs.device, dtype=obs.dtype)

    def dist_and_value(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.features(obs)
        intervene_logits = self.intervene_head(h).squeeze(-1)
        target_logits = self.target_head(h)
        value = self.value_head(h).squeeze(-1)
        return intervene_logits, target_logits, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        intervene: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        intervene_logits, target_logits, value = self.dist_and_value(obs)
        bernoulli = torch.distributions.Bernoulli(logits=intervene_logits)
        categorical = torch.distributions.Categorical(logits=target_logits)

        intervene_f = intervene.to(value.dtype)
        log_prob_intervene = bernoulli.log_prob(intervene_f)
        log_prob_target = categorical.log_prob(target_idx) * intervene_f
        joint_log_prob = log_prob_intervene + log_prob_target

        p_intervene = torch.sigmoid(intervene_logits)
        entropy = bernoulli.entropy() + p_intervene * categorical.entropy()
        return joint_log_prob, entropy, value

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        intervene_logits, target_logits, value = self.dist_and_value(obs)
        bernoulli = torch.distributions.Bernoulli(logits=intervene_logits)
        categorical = torch.distributions.Categorical(logits=target_logits)

        if deterministic:
            intervene = (intervene_logits > 0.0).long()
            target_idx = torch.argmax(target_logits, dim=-1)
        else:
            intervene = bernoulli.sample().long()
            target_idx = categorical.sample()

        w = self.fixed_w(obs)
        log_prob, entropy, value_eval = self.evaluate_actions(obs, intervene, target_idx)
        return {
            "w": w,
            "intervene": intervene,
            "target_idx": target_idx,
            "zeta_idx": target_idx,  # legacy alias
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value_eval,
        }


@torch.no_grad()
def rollout_batch(
    actor_critic: SharedTrunkActorCritic,
    env: VectorizedFEXEnv,
    batch_envs: int,
    deterministic: bool = False,
) -> Dict[str, torch.Tensor]:
    cfg = env.cfg
    device = env.device
    dtype = env.dtype
    steps = cfg.n_t

    x = env.sample_initial_x(batch_envs)

    obs_buf = torch.zeros((steps, batch_envs, 5), device=device, dtype=dtype)
    x_buf = torch.zeros((steps, batch_envs), device=device, dtype=dtype)
    time_idx_buf = torch.zeros((steps, batch_envs), device=device, dtype=torch.long)
    w_buf = torch.zeros((steps, batch_envs), device=device, dtype=dtype)
    intervene_buf = torch.zeros((steps, batch_envs), device=device, dtype=torch.long)
    target_idx_buf = torch.zeros((steps, batch_envs), device=device, dtype=torch.long)
    old_logp_buf = torch.zeros((steps, batch_envs), device=device, dtype=dtype)
    value_buf = torch.zeros((steps, batch_envs), device=device, dtype=dtype)
    reward_buf = torch.zeros((steps, batch_envs), device=device, dtype=dtype)

    for t in range(steps):
        t_idx = torch.full((batch_envs,), t, device=device, dtype=torch.long)
        obs = env.observe(x, t_idx)
        act = actor_critic.act(obs, deterministic=deterministic)
        next_x, reward = env.step(x, t_idx, act["intervene"], act["target_idx"])

        obs_buf[t] = obs
        x_buf[t] = x
        time_idx_buf[t] = t_idx
        w_buf[t] = act["w"]
        intervene_buf[t] = act["intervene"]
        target_idx_buf[t] = act["target_idx"]
        old_logp_buf[t] = act["log_prob"]
        value_buf[t] = act["value"]
        reward_buf[t] = reward

        x = next_x

    mc_returns = torch.zeros_like(reward_buf)
    running = torch.zeros((batch_envs,), device=device, dtype=dtype)
    for t in range(steps - 1, -1, -1):
        running = reward_buf[t] + cfg.gamma_rl * running
        mc_returns[t] = running

    advantages = torch.zeros_like(reward_buf)
    lastgaelam = torch.zeros((batch_envs,), device=device, dtype=dtype)
    next_value = torch.zeros((batch_envs,), device=device, dtype=dtype)
    for t in range(steps - 1, -1, -1):
        delta = reward_buf[t] + cfg.gamma_rl * next_value - value_buf[t]
        lastgaelam = delta + cfg.gamma_rl * cfg.gae_lambda * lastgaelam
        advantages[t] = lastgaelam
        next_value = value_buf[t]

    ppo_returns = advantages + value_buf
    adv_mean = advantages.mean()
    adv_std = advantages.std(unbiased=False).clamp_min(cfg.advantage_eps)
    advantages_norm = (advantages - adv_mean) / adv_std

    return {
        "obs": obs_buf,
        "x": x_buf,
        "time_idx": time_idx_buf,
        "w": w_buf,
        "intervene": intervene_buf,
        "target_idx": target_idx_buf,
        "zeta_idx": target_idx_buf,  # legacy alias
        "old_log_prob": old_logp_buf,
        "value": value_buf,
        "reward": reward_buf,
        "mc_return": mc_returns,
        "ppo_return": ppo_returns,
        "advantage": advantages_norm,
        "advantage_raw": advantages,
    }


def flatten_rollout(td: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in td.items():
        if v.ndim > 2:
            out[k] = v.reshape(-1, *v.shape[2:])
        elif v.ndim == 2:
            out[k] = v.reshape(-1)
        else:
            out[k] = v
    return out


def sil_weight(cfg: FEXPPOActorCriticConfig, iteration: int) -> float:
    return 0.0 if iteration < cfg.sil_warmup_iters else 1.0


def add_rollout_to_sil_buffer(
    sil_buffer: PrioritizedSILBuffer,
    rollout: Dict[str, torch.Tensor],
    cfg: FEXPPOActorCriticConfig,
) -> None:
    flat = flatten_rollout(rollout)
    gap = (flat["mc_return"] - flat["value"] - cfg.sil_margin).clamp_min(0.0)
    priority = gap + cfg.sil_priority_eps
    sil_buffer.add_batch(
        obs=flat["obs"],
        time_idx=flat["time_idx"],
        intervene=flat["intervene"],
        target_idx=flat["target_idx"],
        mc_return=flat["mc_return"],
        value=flat["value"],
        priority=priority,
    )


def train(cfg: Optional[FEXPPOActorCriticConfig] = None) -> Dict[str, Any]:
    if cfg is None:
        cfg = FEXPPOActorCriticConfig()

    set_seed(cfg.seed)
    device = cfg.resolved_device()
    dtype = cfg.resolved_dtype()
    ensure_dir(cfg.save_dir)

    env = VectorizedFEXEnv(cfg, device, dtype)
    obs_dim = 5
    actor_critic = SharedTrunkActorCritic(obs_dim=obs_dim, cfg=cfg).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sil_buffer = PrioritizedSILBuffer(cfg, obs_dim=obs_dim, device=device, dtype=dtype)

    history: Dict[str, list] = {
        "iter": [],
        "train_return_mean": [],
        "train_return_std": [],
        "eval_return_mean": [],
        "eval_return_std": [],
        "policy_loss": [],
        "value_loss": [],
        "sil_policy_loss": [],
        "sil_value_loss": [],
        "entropy": [],
        "kl": [],
        "intervene_rate": [],
        "avg_abs_w": [],
        "value_mean": [],
        "value_abs_error": [],
    }

    best_eval = -float("inf")
    best_path = os.path.join(cfg.save_dir, "best_actor_critic.pt")
    last_path = os.path.join(cfg.save_dir, "last_actor_critic.pt")
    t0 = time.time()

    for it in range(cfg.train_iters):
        rollout = rollout_batch(actor_critic, env, batch_envs=cfg.rollout_envs, deterministic=False)
        add_rollout_to_sil_buffer(sil_buffer, rollout, cfg)
        flat = flatten_rollout(rollout)
        total_items = int(flat["obs"].shape[0])

        policy_loss_acc = 0.0
        value_loss_acc = 0.0
        sil_policy_loss_acc = 0.0
        sil_value_loss_acc = 0.0
        entropy_acc = 0.0
        kl_acc = 0.0
        ppo_steps = 0
        sil_steps = 0

        stop_early = False
        for _epoch in range(cfg.ppo_epochs):
            perm = torch.randperm(total_items, device=device)
            for start in range(0, total_items, cfg.minibatch_size):
                idx = perm[start : start + cfg.minibatch_size]
                if idx.numel() == 0:
                    continue

                obs_mb = flat["obs"][idx]
                intervene_mb = flat["intervene"][idx]
                target_idx_mb = flat["target_idx"][idx]
                adv_mb = flat["advantage"][idx]
                old_lp_mb = flat["old_log_prob"][idx]
                value_target_mb = flat["mc_return"][idx]

                new_lp, entropy, value_pred = actor_critic.evaluate_actions(
                    obs_mb,
                    intervene_mb,
                    target_idx_mb,
                )

                log_ratio = new_lp - old_lp_mb
                ratio = torch.exp(log_ratio)
                surr1 = ratio * adv_mb
                surr2 = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_mb

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (value_pred - value_target_mb).pow(2).mean()
                entropy_term = entropy.mean()

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_term

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(actor_critic.parameters(), cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_lp_mb - new_lp).mean().abs()

                policy_loss_acc += float(policy_loss.item())
                value_loss_acc += float(value_loss.item())
                entropy_acc += float(entropy_term.item())
                kl_acc += float(approx_kl.item())
                ppo_steps += 1

                if approx_kl.item() > cfg.target_kl:
                    stop_early = True
                    break

            if stop_early:
                break

        sil_scale = sil_weight(cfg, it)
        if sil_scale > 0.0 and len(sil_buffer) >= cfg.sil_batch_size and cfg.sil_updates_per_iter > 0:
            for _ in range(cfg.sil_updates_per_iter):
                sil_batch, sil_idx = sil_buffer.sample(cfg.sil_batch_size)

                sil_lp, _, sil_value = actor_critic.evaluate_actions(
                    sil_batch["obs"],
                    sil_batch["intervene"],
                    sil_batch["target_idx"],
                )
                sil_gap = (sil_batch["mc_return"] - sil_value - cfg.sil_margin).clamp_min(0.0)
                pos = sil_gap > 0.0

                if pos.any():
                    gap_pos = sil_gap[pos]
                    lp_pos = sil_lp[pos]

                    sil_policy_loss = -(gap_pos.detach() * lp_pos).sum() / gap_pos.detach().sum().clamp_min(1.0)
                    sil_value_loss = 0.5 * gap_pos.pow(2).mean()
                    sil_loss = sil_scale * (
                        cfg.sil_policy_coef * sil_policy_loss
                        + cfg.sil_value_coef * sil_value_loss
                    )

                    optimizer.zero_grad(set_to_none=True)
                    sil_loss.backward()
                    nn.utils.clip_grad_norm_(actor_critic.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                    sil_policy_loss_acc += float(sil_policy_loss.item())
                    sil_value_loss_acc += float(sil_value_loss.item())
                    sil_steps += 1

                sil_buffer.update_priority(sil_idx, sil_gap + cfg.sil_priority_eps)

        train_return = rollout["mc_return"][0]
        intervene_rate = float(rollout["intervene"].float().mean().item())
        avg_abs_w = float(rollout["w"].abs().mean().item())
        value_mean = float(rollout["value"].mean().item())
        value_abs_error = float((rollout["mc_return"] - rollout["value"]).abs().mean().item())

        history["iter"].append(it)
        history["train_return_mean"].append(float(train_return.mean().item()))
        history["train_return_std"].append(float(train_return.std(unbiased=False).item()))
        history["policy_loss"].append(policy_loss_acc / max(1, ppo_steps))
        history["value_loss"].append(value_loss_acc / max(1, ppo_steps))
        history["sil_policy_loss"].append(sil_policy_loss_acc / max(1, sil_steps))
        history["sil_value_loss"].append(sil_value_loss_acc / max(1, sil_steps))
        history["entropy"].append(entropy_acc / max(1, ppo_steps))
        history["kl"].append(kl_acc / max(1, ppo_steps))
        history["intervene_rate"].append(intervene_rate)
        history["avg_abs_w"].append(avg_abs_w)
        history["value_mean"].append(value_mean)
        history["value_abs_error"].append(value_abs_error)

        do_eval = ((it + 1) % max(1, cfg.eval_every) == 0) or (it == 0) or (it == cfg.train_iters - 1)
        if do_eval:
            eval_rollout = rollout_batch(actor_critic, env, batch_envs=cfg.eval_envs, deterministic=True)
            eval_return = eval_rollout["mc_return"][0]
            eval_mean = float(eval_return.mean().item())
            eval_std = float(eval_return.std(unbiased=False).item())

            history["eval_return_mean"].append(eval_mean)
            history["eval_return_std"].append(eval_std)

            if eval_mean > best_eval:
                best_eval = eval_mean
                torch.save(
                    {
                        "actor_critic_state_dict": actor_critic.state_dict(),
                        "config": asdict(cfg),
                        "target_grid": env.target_grid.detach().cpu(),
                        "best_eval_return_mean": best_eval,
                    },
                    best_path,
                )
        else:
            history["eval_return_mean"].append(float("nan"))
            history["eval_return_std"].append(float("nan"))

        if do_eval:
            print(
                f"[iter {it + 1:04d}/{cfg.train_iters}] "
                f"train={history['train_return_mean'][-1]: .4f} "
                f"eval={history['eval_return_mean'][-1]: .4f} "
                f"pi={history['policy_loss'][-1]: .4f} "
                f"v={history['value_loss'][-1]: .4f} "
                f"sil_pi={history['sil_policy_loss'][-1]: .4f} "
                f"sil_v={history['sil_value_loss'][-1]: .4f} "
                f"ent={history['entropy'][-1]: .4f} "
                f"intv={history['intervene_rate'][-1]: .4f} "
                f"|w|={history['avg_abs_w'][-1]: .4f}"
            )

    elapsed = time.time() - t0
    torch.save(
        {
            "actor_critic_state_dict": actor_critic.state_dict(),
            "config": asdict(cfg),
            "target_grid": env.target_grid.detach().cpu(),
            "history": history,
            "best_eval_return_mean": best_eval,
        },
        last_path,
    )

    return {
        "actor_critic": actor_critic,
        "env": env,
        "history": history,
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "elapsed_sec": elapsed,
        "sil_backend": "simple_torch",
    }


@torch.no_grad()
def evaluate_policy(
    actor_critic: SharedTrunkActorCritic,
    env: VectorizedFEXEnv,
    num_envs: int = 4096,
    deterministic: bool = True,
) -> Dict[str, float]:
    td = rollout_batch(actor_critic, env, batch_envs=num_envs, deterministic=deterministic)
    ep_ret = td["mc_return"][0]
    mask = td["intervene"].bool()
    if mask.any():
        target = env.target_grid[td["target_idx"]]
        avg_abs_jump = float((target[mask] - td["x"][mask]).abs().mean().item())
    else:
        avg_abs_jump = 0.0

    return {
        "return_mean": float(ep_ret.mean().item()),
        "return_std": float(ep_ret.std(unbiased=False).item()),
        "intervene_rate": float(td["intervene"].float().mean().item()),
        "avg_abs_w": float(td["w"].abs().mean().item()),
        "avg_abs_jump_if_intervene": avg_abs_jump,
        "value_mean": float(td["value"].mean().item()),
        "value_abs_error": float((td["mc_return"] - td["value"]).abs().mean().item()),
    }


@torch.no_grad()
def policy_snapshot_t0(
    actor_critic: SharedTrunkActorCritic,
    env: VectorizedFEXEnv,
    x_grid: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    if x_grid is None:
        x_grid = linear_grid(env.cfg.x_init_min, env.cfg.x_init_max, 201, env.device, env.dtype)
    else:
        x_grid = torch.as_tensor(x_grid, device=env.device, dtype=env.dtype)

    t_idx = torch.zeros_like(x_grid, dtype=torch.long)
    obs = env.observe(x_grid, t_idx)
    out = actor_critic.act(obs, deterministic=True)
    target = env.target_grid[out["target_idx"]]
    jump = target - x_grid

    return {
        "x": x_grid.detach().cpu(),
        "w": out["w"].detach().cpu(),
        "intervene": out["intervene"].detach().cpu(),
        "target_idx": out["target_idx"].detach().cpu(),
        "target": target.detach().cpu(),
        "jump": jump.detach().cpu(),
        "value": out["value"].detach().cpu(),
    }


@torch.no_grad()
def mc_value_curve_t0(
    actor_critic: SharedTrunkActorCritic,
    env: VectorizedFEXEnv,
    x_grid: torch.Tensor,
    n_paths: int = 2048,
    deterministic: bool = True,
    chunk_size: int = 32,
):
    x_grid_t = torch.as_tensor(x_grid, device=env.device, dtype=env.dtype).reshape(-1)
    if x_grid_t.numel() == 0:
        return x_grid_t.detach().cpu().numpy()

    was_training = actor_critic.training
    actor_critic.eval()
    values = []

    for start in range(0, x_grid_t.numel(), chunk_size):
        x_block = x_grid_t[start : start + chunk_size]
        block_size = int(x_block.numel())

        x = x_block[:, None].expand(block_size, n_paths).reshape(-1).contiguous()
        total = torch.zeros_like(x)

        for t in range(env.cfg.n_t):
            t_idx = torch.full((x.shape[0],), t, device=env.device, dtype=torch.long)
            obs = env.observe(x, t_idx)
            act = actor_critic.act(obs, deterministic=deterministic)
            x, reward = env.step(x, t_idx, act["intervene"], act["target_idx"])
            total = total + reward

        block_value = total.view(block_size, n_paths).mean(dim=1)
        values.append(block_value)

    if was_training:
        actor_critic.train()

    return torch.cat(values, dim=0).detach().cpu().numpy()


def make_default_config(**kwargs: Any) -> FEXPPOActorCriticConfig:
    cfg = FEXPPOActorCriticConfig()
    ignored_legacy_keys = {"use_fixed_w", "w_max", "w_logstd_init", "min_std"}
    valid_fields = set(cfg.__dataclass_fields__.keys())

    for k, v in kwargs.items():
        if k in ignored_legacy_keys:
            continue
        if k not in valid_fields:
            raise TypeError(f"Unknown config field: {k}")
        setattr(cfg, k, v)
    return cfg