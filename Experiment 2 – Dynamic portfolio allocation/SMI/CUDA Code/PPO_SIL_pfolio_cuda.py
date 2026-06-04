"""
Jain-style PPO + Self-Imitation Learning for impulse-control portfolio allocation.

This module is intentionally self-contained.  It replaces the previous masked
categorical policy over a finite rebalance grid with a two-network impulse
architecture:

    1. d-network: Bernoulli policy deciding whether to intervene.
    2. u-network: continuous diagonal-Gaussian policy producing a d-dimensional
       trade vector, so all assets can be traded simultaneously.

The stock-price dynamics are generated through the uploaded SSG.py module
(`SSG.SSG.simulate_stock_price_process`).  The training reward is a scaled increment of discounted wealth, so
maximising the PPO/SIL return is equivalent to maximising the original
performance criterion: expected discounted terminal wealth.

No trade/impulse discretisation is used.
"""

from __future__ import annotations

import importlib.util
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def get_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def activation_module(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation {name!r}.")


def orthogonal_init_(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


def _as_float_list(x: float | Iterable[float], d: int, name: str) -> List[float]:
    if isinstance(x, (float, int)):
        return [float(x)] * int(d)
    out = [float(v) for v in x]
    if len(out) != int(d):
        raise ValueError(f"{name} must have length d={d}; got length {len(out)}.")
    return out


def _as_float_matrix(
    x: Optional[float | Iterable[float] | Iterable[Iterable[float]]],
    d: int,
    name: str,
    *,
    default_zero: bool = True,
) -> List[List[float]]:
    """Return a d x d float matrix.

    Scalars and length-one vectors are interpreted as diagonal matrices with
    constant diagonal.  Length-d vectors are interpreted as diagonal matrices.
    """
    if x is None:
        arr = np.zeros((int(d), int(d)), dtype=float) if default_zero else np.eye(int(d), dtype=float)
    else:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            arr = np.eye(int(d), dtype=float) * float(arr)
        elif arr.ndim == 1:
            if arr.size == 1:
                arr = np.eye(int(d), dtype=float) * float(arr.item())
            elif arr.size == int(d):
                arr = np.diag(arr.astype(float))
            else:
                raise ValueError(f"{name} vector must have length 1 or d={d}; got length {arr.size}.")
        elif arr.ndim == 2:
            if arr.shape != (int(d), int(d)):
                raise ValueError(f"{name} must have shape ({d},{d}); got {arr.shape}.")
        else:
            raise ValueError(f"{name} must be scalar, vector, or matrix.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")
    return arr.astype(float).tolist()


def _finite_or_raise(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        bad = x[~torch.isfinite(x)]
        sample = bad[:5].detach().cpu().numpy() if bad.numel() else []
        raise FloatingPointError(f"{name} contains non-finite values; sample={sample}")


def _clear_device_cache(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass
class MarketConfig:
    # Market dimension and horizon.
    d: int = 2
    T: float = 1.0
    n_steps: int = 252
    r: float = 0.04

    # Drift and initial portfolio.
    mu: List[float] = field(default_factory=lambda: [-0.10, 0.20])
    s0: List[float] = field(default_factory=lambda: [100.0, 100.0])
    h0: List[float] = field(default_factory=lambda: [100.0, 100.0])
    cash0: float = 10_000.0

    # Revised SSG rho-alpha VG inputs.
    sigma: List[float] = field(default_factory=lambda: [0.15, 0.15])
    theta: List[float] = field(default_factory=lambda: [0.0, 0.0])
    nu: List[float] = field(default_factory=lambda: [0.20, 0.20])
    log_returns_corr_matrix: Optional[List[List[float]] | List[float] | float] = None

    # Stock-price simulation control.  ``ssg_seed`` is kept only so older
    # checkpoints/notebooks still instantiate; it is not used by the simulator.
    ssg_seed: Optional[int] = None
    ssg_path_mean_samples: int = 10
    ssg_max_simulated_paths_per_chunk: int = 8192

    # Legacy fields accepted for backward compatibility with older notebooks /
    # checkpoints.  They are not used by the revised SSG adapter.
    vg_C: Optional[List[float]] = None
    vg_G: Optional[List[float]] = None
    vg_M: Optional[List[float]] = None
    ssg_sigma_mat: Optional[List[List[float]] | List[float] | float] = None
    ssg_scale_to_T: bool = False
    ssg_generate_on_cpu: bool = False
    vg_drift_mode: str = "martingale"
    gamma_on_cpu: bool = True
    vg_corr_matrix: Optional[List[List[float]]] = None

    # Transaction costs.  fixed_cost is charged once per non-zero impulse, not
    # once per asset, so simultaneous multi-asset trades are allowed.
    fixed_cost: float = 29.0
    proportional_cost: float = 0.001

    # Continuous action constraints in shares.
    max_holding: float | List[float] = 200.0
    min_holding: float | List[float] = 0.0
    max_trade_size: float | List[float] = 200.0
    allow_negative_cash: bool = False
    self_financing: bool = True

    # Initial-state randomisation.  All are off by default to reproduce the
    # original example's deterministic initial portfolio.
    randomize_initial_prices: bool = False
    randomize_initial_holdings: bool = False
    randomize_initial_cash: bool = False
    initial_price_log_std: float = 0.02
    initial_holding_noise: float = 0.0
    initial_cash_noise: float = 0.0

    # Reward scaling only changes numerical conditioning; it does not change
    # the maximiser of expected discounted terminal wealth.
    reward_scale: float = 1.0e-3

    # Numerical safeguards.
    min_price: float = 1.0e-8
    max_price: float = 1.0e8

    def __post_init__(self) -> None:
        self.d = int(self.d)
        if self.d < 1:
            raise ValueError("d must be positive.")
        if self.T <= 0 or self.n_steps < 1:
            raise ValueError("T must be positive and n_steps must be at least 1.")
        self.mu = _as_float_list(self.mu, self.d, "mu")
        self.s0 = _as_float_list(self.s0, self.d, "s0")
        self.h0 = _as_float_list(self.h0, self.d, "h0")
        self.sigma = _as_float_list(self.sigma, self.d, "sigma")
        self.theta = _as_float_list(self.theta, self.d, "theta")
        self.nu = _as_float_list(self.nu, self.d, "nu")
        self.log_returns_corr_matrix = _as_float_matrix(
            self.log_returns_corr_matrix,
            self.d,
            "log_returns_corr_matrix",
            default_zero=False,
        )
        self.max_holding = _as_float_list(self.max_holding, self.d, "max_holding")
        self.min_holding = _as_float_list(self.min_holding, self.d, "min_holding")
        self.max_trade_size = _as_float_list(self.max_trade_size, self.d, "max_trade_size")
        self.ssg_path_mean_samples = max(1, int(self.ssg_path_mean_samples))
        self.ssg_max_simulated_paths_per_chunk = max(1, int(self.ssg_max_simulated_paths_per_chunk))

        if any(s <= 0.0 for s in self.sigma):
            raise ValueError("SSG sigma entries must be strictly positive.")
        if any(n <= 0.0 for n in self.nu):
            raise ValueError("SSG nu entries must be strictly positive.")
        log_args = [
            1.0 - th * nu_i - 0.5 * sig * sig * nu_i
            for sig, th, nu_i in zip(self.sigma, self.theta, self.nu)
        ]
        if any(x <= 0.0 for x in log_args):
            raise ValueError(
                "Invalid SSG parameters: every 1 - theta_i * nu_i - 0.5 * sigma_i^2 * nu_i must be positive."
            )

        corr = np.asarray(self.log_returns_corr_matrix, dtype=float)
        if corr.shape != (self.d, self.d):
            raise ValueError(f"log_returns_corr_matrix must have shape ({self.d},{self.d}); got {corr.shape}.")
        if not np.isfinite(corr).all():
            raise ValueError("log_returns_corr_matrix contains non-finite values.")
        if not np.allclose(corr, corr.T, atol=1e-8):
            raise ValueError("log_returns_corr_matrix must be symmetric.")
        if np.any(np.diag(corr) <= 0.0):
            raise ValueError("log_returns_corr_matrix diagonal entries must be positive.")

        if self.vg_corr_matrix is not None:
            arr = np.asarray(self.vg_corr_matrix, dtype=float)
            if arr.shape != (self.d, self.d):
                raise ValueError("vg_corr_matrix must have shape (d,d).")


@dataclass
class PPOSILConfig:
    seed: int = 42
    device: Optional[str] = None
    dtype: str = "float32"

    train_iters: int = 500
    rollout_envs: int = 512
    gamma_rl: float = 1.0
    gae_lambda: float = 1.0

    ppo_epochs: int = 5
    minibatch_size: int = 4096
    lr: float = 3.0e-4
    weight_decay: float = 0.0
    clip_eps: float = 0.20
    value_coef: float = 0.50
    entropy_coef: float = 1.0e-3
    target_kl: float = 0.05
    max_grad_norm: float = 1.0
    advantage_eps: float = 1.0e-8

    # Two-network architecture.
    hidden_dims: Tuple[int, ...] = (512, 512, 128)
    activation: str = "tanh"
    layer_norm: bool = False
    decision_bias: float = -1.0
    init_log_std: float = -0.35
    min_log_std: float = -3.0
    max_log_std: float = 1.0

    # SIL replay and losses.
    use_sil: bool = True
    sil_buffer_capacity: int = 200_000
    sil_batch_size: int = 4096
    sil_updates_per_iter: int = 1
    sil_warmup_iters: int = 2
    sil_priority_alpha: float = 0.7
    sil_priority_eps: float = 1.0e-6
    sil_margin: float = 0.0
    sil_decision_policy_coef: float = 0.05
    sil_decision_value_coef: float = 0.05
    sil_impulse_policy_coef: float = 0.05
    sil_impulse_value_coef: float = 0.05
    keep_sil_on_cpu: bool = True

    # Evaluation / logging.
    eval_envs: int = 1024
    eval_every: int = 10
    eval_batch_size: int = 256
    eval_deterministic: bool = False
    log_every: int = 1
    checkpoint_path: str = "PPO_SIL_pfolio.pt"
    clear_cache: bool = True


# ---------------------------------------------------------------------------
# SSG-driven vectorised portfolio environment
# ---------------------------------------------------------------------------


_SSG_MODULE_CANDIDATES = ("SSG_cuda", "SSG")


def _import_ssg_module() -> Any:
    """Import the SSG module without requiring a package layout."""
    first_error: Optional[BaseException] = None
    for module_name in _SSG_MODULE_CANDIDATES:
        try:
            return __import__(module_name)  # type: ignore[no-any-return]
        except Exception as exc:
            if first_error is None:
                first_error = exc

    local_candidates = [Path(__file__).with_name(f"{name}.py") for name in _SSG_MODULE_CANDIDATES]
    for local_path in local_candidates:
        if not local_path.exists():
            continue
        spec = importlib.util.spec_from_file_location(local_path.stem, str(local_path))
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    raise ImportError(
        "Could not import an SSG module. Put SSG_cuda.py or SSG.py in the same directory "
        "as this PPO_SIL_pfolio file, or make it importable on PYTHONPATH."
    ) from first_error


SSG = _import_ssg_module()


def _make_ssg_simulator(
    *,
    S0_vec: Iterable[float],
    mu_vec: Iterable[float],
    sigma_vec: Iterable[float],
    theta_vec: Iterable[float],
    nu_vec: Iterable[float],
    log_returns_corr_matrix: Iterable[Iterable[float]],
    T: float,
    N_t: int,
    N_MC: int,
    device: torch.device,
    dtype: torch.dtype,
    path_mean_samples: int,
    max_simulated_paths_per_chunk: int,
) -> Any:
    """Construct the revised ``SSG.SSG`` simulator once."""
    cls = getattr(SSG, "SSG", None)
    if cls is None:
        raise ImportError("The imported SSG module does not expose an SSG class.")

    S0 = np.asarray(S0_vec, dtype=float).reshape(-1)
    mu = np.asarray(mu_vec, dtype=float).reshape(-1)
    sigma = np.asarray(sigma_vec, dtype=float).reshape(-1)
    theta = np.asarray(theta_vec, dtype=float).reshape(-1)
    nu = np.asarray(nu_vec, dtype=float).reshape(-1)
    corr = np.asarray(log_returns_corr_matrix, dtype=float)

    d = int(S0.size)
    for name, arr in {"mu_vec": mu, "sigma_vec": sigma, "theta_vec": theta, "nu_vec": nu}.items():
        if arr.size != d:
            raise ValueError(f"{name} must have length d={d}; got length {arr.size}.")
    if corr.shape != (d, d):
        raise ValueError(f"log_returns_corr_matrix must have shape ({d},{d}); got {corr.shape}.")

    return cls(
        S0_vec=S0,
        mu_vec=mu,
        sigma_vec=sigma,
        theta_vec=theta,
        nu_vec=nu,
        log_returns_corr_matrix=corr,
        T=float(T),
        N_t=int(N_t),
        N_MC=int(N_MC),
        device=device,
        dtype=dtype,
        seed=None,
        return_torch=True,
        path_mean_samples=int(path_mean_samples),
        max_simulated_paths_per_chunk=int(max_simulated_paths_per_chunk),
    )


class VarianceGammaPortfolioEnv:
    """Vectorised impulse-control environment using SSG.py for stock paths.

    One ``SSG.SSG`` instance is initialised in ``__init__`` and reused for all
    rollouts/evaluations.  The stock-price simulator itself is not seeded; it
    uses entropy-initialised torch generators, so the price paths are not tied
    to the PPO seed.
    """

    def __init__(self, cfg: MarketConfig, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.d = int(cfg.d)
        self.dt = float(cfg.T) / int(cfg.n_steps)
        self.obs_dim = 4 + 3 * self.d
        self.action_dim = self.d

        self.mu = torch.tensor(cfg.mu, device=device, dtype=dtype)
        self.s0 = torch.tensor(cfg.s0, device=device, dtype=dtype)
        self.h0 = torch.tensor(cfg.h0, device=device, dtype=dtype)
        self.max_holding = torch.tensor(cfg.max_holding, device=device, dtype=dtype)
        self.min_holding = torch.tensor(cfg.min_holding, device=device, dtype=dtype)
        self.max_trade_size = torch.tensor(cfg.max_trade_size, device=device, dtype=dtype)
        self.sigma = torch.tensor(cfg.sigma, device=device, dtype=dtype)
        self.theta = torch.tensor(cfg.theta, device=device, dtype=dtype)
        self.nu = torch.tensor(cfg.nu, device=device, dtype=dtype)
        self.log_returns_corr_matrix = torch.tensor(cfg.log_returns_corr_matrix, device=device, dtype=dtype)
        self.cash_scale = max(float(cfg.cash0), 1.0)
        self.wealth_scale = max(float(cfg.cash0 + float(torch.sum(self.h0 * self.s0).detach().cpu())), 1.0)

        s0_np, mu_np, sigma_np, theta_np, nu_np, corr_np = self._ssg_inputs()
        self._ssg_simulator = _make_ssg_simulator(
            S0_vec=s0_np,
            mu_vec=mu_np,
            sigma_vec=sigma_np,
            theta_vec=theta_np,
            nu_vec=nu_np,
            log_returns_corr_matrix=corr_np,
            T=float(self.cfg.T),
            N_t=int(self.cfg.n_steps),
            N_MC=1,
            device=self.device,
            dtype=self.dtype,
            path_mean_samples=int(self.cfg.ssg_path_mean_samples),
            max_simulated_paths_per_chunk=int(self.cfg.ssg_max_simulated_paths_per_chunk),
        )

    def _ssg_inputs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the revised SSG inputs as NumPy arrays."""
        s0 = np.asarray(self.cfg.s0, dtype=float)
        mu = np.asarray(self.cfg.mu, dtype=float)
        sigma = np.asarray(self.cfg.sigma, dtype=float)
        theta = np.asarray(self.cfg.theta, dtype=float)
        nu = np.asarray(self.cfg.nu, dtype=float)
        corr = np.asarray(self.cfg.log_returns_corr_matrix, dtype=float)
        return s0, mu, sigma, theta, nu, corr

    @torch.no_grad()
    def sample_price_paths(self, n_envs: int, seed: Optional[int] = None) -> torch.Tensor:
        """Generate SSG stock paths with shape ``(n_envs, n_steps + 1, d)``.

        The ``seed`` argument is ignored intentionally: stock-price paths should
        remain genuinely random across rollouts.
        """
        _ = seed
        n_envs = int(n_envs)
        self._ssg_simulator.N_MC = n_envs
        paths_raw = self._ssg_simulator.simulate_stock_price_process(
            N_MC=n_envs,
            mean_samples=int(self.cfg.ssg_path_mean_samples),
            return_torch=True,
        )
        if isinstance(paths_raw, torch.Tensor):
            paths = paths_raw.to(device=self.device, dtype=self.dtype)
        else:
            paths = torch.as_tensor(paths_raw, device=self.device, dtype=self.dtype)
        if tuple(paths.shape) != (n_envs, int(self.cfg.n_steps) + 1, self.d):
            raise RuntimeError(
                f"SSG returned paths with shape {tuple(paths.shape)}, expected "
                f"({n_envs}, {int(self.cfg.n_steps) + 1}, {self.d})."
            )
        return paths.clamp(float(self.cfg.min_price), float(self.cfg.max_price))

    def reset(self, n_envs: int, price_paths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        n_envs = int(n_envs)
        if price_paths is None:
            price_paths = self.sample_price_paths(n_envs)
        else:
            price_paths = price_paths.to(device=self.device, dtype=self.dtype)
            expected = (n_envs, int(self.cfg.n_steps) + 1, self.d)
            if tuple(price_paths.shape) != expected:
                raise ValueError(f"price_paths must have shape {expected}; got {tuple(price_paths.shape)}.")
            price_paths = price_paths.clamp(float(self.cfg.min_price), float(self.cfg.max_price))

        if self.cfg.randomize_initial_prices:
            eps = torch.randn((n_envs, self.d), device=self.device, dtype=self.dtype) * float(self.cfg.initial_price_log_std)
            price_paths = price_paths * torch.exp(eps).unsqueeze(1)
            price_paths = price_paths.clamp(float(self.cfg.min_price), float(self.cfg.max_price))

        s = price_paths[:, 0, :]
        h = self.h0.unsqueeze(0).repeat(n_envs, 1)
        cash = torch.full((n_envs,), float(self.cfg.cash0), device=self.device, dtype=self.dtype)

        if self.cfg.randomize_initial_holdings and self.cfg.initial_holding_noise > 0:
            h = h + torch.randn_like(h) * float(self.cfg.initial_holding_noise)
            h = h.clamp(self.min_holding, self.max_holding)
        if self.cfg.randomize_initial_cash and self.cfg.initial_cash_noise > 0:
            cash = cash + torch.randn_like(cash) * float(self.cfg.initial_cash_noise)
            if not self.cfg.allow_negative_cash:
                cash = cash.clamp_min(0.0)

        return {
            "s": s,
            "h": h,
            "cash": cash,
            "step": torch.zeros(n_envs, device=self.device, dtype=torch.long),
            "price_path": price_paths,
        }

    def time(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return state["step"].to(self.dtype) * self.dt

    def wealth(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return state["cash"] + (state["h"] * state["s"]).sum(dim=-1)

    def discounted_wealth(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.exp(-float(self.cfg.r) * self.time(state)) * self.wealth(state)

    def observe(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        s = state["s"]
        h = state["h"]
        cash = state["cash"]
        step = state["step"].to(self.dtype)
        t_frac = step / float(self.cfg.n_steps)
        wealth = self.wealth(state).clamp_min(1.0e-8)
        log_prices = torch.log((s / self.s0).clamp_min(1.0e-12)).clamp(-10.0, 10.0)
        h_scaled = h / self.max_holding.clamp_min(1.0)
        risky_weights = (h * s) / wealth.unsqueeze(-1)
        return torch.cat(
            [
                t_frac.unsqueeze(-1),
                (1.0 - t_frac).unsqueeze(-1),
                (cash / self.cash_scale).unsqueeze(-1),
                (wealth / self.wealth_scale).unsqueeze(-1),
                log_prices,
                h_scaled,
                risky_weights,
            ],
            dim=-1,
        )

    def sample_vg_log_increment(self, n_envs: int) -> torch.Tensor:
        """Backward-compatible one-step increment sampler, generated through SSG."""
        paths = self.sample_price_paths(int(n_envs))
        return torch.log((paths[:, 1, :] / paths[:, 0, :]).clamp_min(1.0e-12))

    def _transaction_cost(self, s: torch.Tensor, trade: torch.Tensor) -> torch.Tensor:
        notional = (trade.abs() * s).sum(dim=-1)
        has_trade = (trade.abs().sum(dim=-1) > 1.0e-8).to(self.dtype)
        return float(self.cfg.fixed_cost) * has_trade + float(self.cfg.proportional_cost) * notional

    def project_trade(self, state: Dict[str, torch.Tensor], desired_trade: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project continuous trade vector to holdings/cash constraints."""
        s = state["s"]
        h = state["h"]
        cash = state["cash"]
        trade = desired_trade.clamp(-self.max_trade_size, self.max_trade_size)
        trade = torch.minimum(trade, self.max_holding - h)
        trade = torch.maximum(trade, self.min_holding - h)

        if self.cfg.self_financing and not self.cfg.allow_negative_cash:
            cost = self._transaction_cost(s, trade)
            new_cash = cash - (trade * s).sum(dim=-1) - cost
            bad = new_cash < -1.0e-6
            if bad.any():
                # Keep sells, scale buys to fit the post-sell cash budget.
                buys = trade.clamp_min(0.0)
                sells = trade.clamp_max(0.0)
                sell_notional = (sells.abs() * s).sum(dim=-1)
                sell_prop_cost = float(self.cfg.proportional_cost) * sell_notional
                has_any = (trade.abs().sum(dim=-1) > 1.0e-8).to(self.dtype)
                available = cash + sell_notional - sell_prop_cost - float(self.cfg.fixed_cost) * has_any
                buy_notional = (buys * s).sum(dim=-1)
                max_buy_notional = (available / (1.0 + float(self.cfg.proportional_cost))).clamp_min(0.0)
                scale = torch.where(buy_notional > 1.0e-12, (max_buy_notional / buy_notional).clamp(0.0, 1.0), torch.ones_like(buy_notional))
                trade2 = sells + buys * scale.unsqueeze(-1)
                cost2 = self._transaction_cost(s, trade2)
                new_cash2 = cash - (trade2 * s).sum(dim=-1) - cost2
                # If even the fixed cost on a tiny trade is infeasible, drop the impulse.
                infeasible = new_cash2 < -1.0e-6
                trade = torch.where((bad & ~infeasible).unsqueeze(-1), trade2, trade)
                trade = torch.where(infeasible.unsqueeze(-1), torch.zeros_like(trade), trade)

        cost = self._transaction_cost(s, trade)
        return trade, cost

    def step(
        self,
        state: Dict[str, torch.Tensor],
        intervene: torch.Tensor,
        trade_intent: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        old_disc_wealth = self.discounted_wealth(state)
        intervene_f = intervene.to(self.dtype).view(-1, 1)
        desired_trade = trade_intent * intervene_f
        trade, cost = self.project_trade(state, desired_trade)

        cash_after = state["cash"] - (trade * state["s"]).sum(dim=-1) - cost
        h_after = state["h"] + trade

        next_step = (state["step"] + 1).clamp_max(int(self.cfg.n_steps))
        if "price_path" in state:
            row = torch.arange(state["s"].shape[0], device=self.device)
            s_next = state["price_path"][row, next_step, :]
        else:
            log_inc = self.sample_vg_log_increment(state["s"].shape[0])
            s_next = state["s"] * torch.exp(log_inc)
        s_next = s_next.clamp(float(self.cfg.min_price), float(self.cfg.max_price))

        next_state = {
            "s": s_next,
            "h": h_after,
            "cash": cash_after,
            "step": next_step,
        }
        if "price_path" in state:
            next_state["price_path"] = state["price_path"]

        new_disc_wealth = self.discounted_wealth(next_state)
        reward_actual = new_disc_wealth - old_disc_wealth
        reward = reward_actual * float(self.cfg.reward_scale)
        done = next_state["step"] >= int(self.cfg.n_steps)
        info = {
            "safe_trade": trade,
            "cost": cost,
            "reward_actual": reward_actual,
            "trade_indicator": (trade.abs().sum(dim=-1) > 1.0e-8).to(self.dtype),
            "mean_abs_trade": trade.abs().mean(dim=-1),
        }
        return next_state, reward, done, info


# Alias with a name that states the current price source explicitly.
SSGPortfolioEnv = VarianceGammaPortfolioEnv


# ---------------------------------------------------------------------------
# Jain-style two-network actor-critic
# ---------------------------------------------------------------------------


class MLPTrunk(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: Tuple[int, ...], activation: str, layer_norm: bool) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last = int(obs_dim)
        for h in hidden_dims:
            lin = nn.Linear(last, int(h))
            if str(activation).lower() == "relu":
                nn.init.kaiming_uniform_(lin.weight, nonlinearity="relu")
                nn.init.zeros_(lin.bias)
            else:
                orthogonal_init_(lin, gain=math.sqrt(2.0))
            layers.append(lin)
            if layer_norm:
                layers.append(nn.LayerNorm(int(h)))
            layers.append(activation_module(activation))
            last = int(h)
        self.net = nn.Sequential(*layers)
        self.out_dim = last

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DecisionActorCritic(nn.Module):
    """d_chi network: Bernoulli intervention policy and value."""

    def __init__(self, obs_dim: int, cfg: PPOSILConfig) -> None:
        super().__init__()
        self.trunk = MLPTrunk(obs_dim, cfg.hidden_dims, cfg.activation, cfg.layer_norm)
        self.logit_head = nn.Linear(self.trunk.out_dim, 1)
        self.value_head = nn.Linear(self.trunk.out_dim, 1)
        orthogonal_init_(self.logit_head, gain=0.01)
        orthogonal_init_(self.value_head, gain=1.0)
        nn.init.constant_(self.logit_head.bias, float(cfg.decision_bias))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        logits = self.logit_head(h).squeeze(-1)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def evaluate_actions(self, obs: torch.Tensor, intervene: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = torch.distributions.Bernoulli(logits=logits)
        intervene_f = intervene.to(value.dtype)
        logp = dist.log_prob(intervene_f)
        entropy = dist.entropy()
        prob = torch.sigmoid(logits)
        return logp, entropy, value, prob

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = torch.distributions.Bernoulli(logits=logits)
        prob = torch.sigmoid(logits)

        if deterministic:
            intervene = (prob > 0.5).long()
        else:
            intervene = dist.sample().long()
        logp = dist.log_prob(intervene.to(value.dtype))
        return {"intervene": intervene, "logp_d": logp, "entropy_d": dist.entropy(), "value_d": value, "prob_d": torch.sigmoid(logits)}


class ImpulseActorCritic(nn.Module):
    """u_xi network: continuous d-dimensional Gaussian trade policy and value."""

    def __init__(self, obs_dim: int, action_dim: int, cfg: PPOSILConfig, max_trade_size: torch.Tensor) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.cfg = cfg
        self.trunk = MLPTrunk(obs_dim, cfg.hidden_dims, cfg.activation, cfg.layer_norm)
        self.mean_head = nn.Linear(self.trunk.out_dim, self.action_dim)
        self.value_head = nn.Linear(self.trunk.out_dim, 1)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(cfg.init_log_std)))
        self.register_buffer("max_trade_size", max_trade_size.detach().clone().view(1, -1))
        orthogonal_init_(self.mean_head, gain=0.01)
        orthogonal_init_(self.value_head, gain=1.0)

    def _dist_value(self, obs: torch.Tensor) -> Tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std.clamp(float(self.cfg.min_log_std), float(self.cfg.max_log_std))
        std = torch.exp(log_std).expand_as(mean)
        value = self.value_head(h).squeeze(-1)
        return torch.distributions.Normal(mean, std), value, mean

    def raw_to_trade(self, raw_action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(raw_action) * self.max_trade_size.to(raw_action.device, raw_action.dtype)

    def evaluate_actions(self, obs: torch.Tensor, raw_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value, _ = self._dist_value(obs)
        logp = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return logp, entropy, value

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        dist, value, mean = self._dist_value(obs)
        raw = mean if deterministic else dist.rsample()
        trade_intent = self.raw_to_trade(raw)
        logp = dist.log_prob(raw).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return {"raw_action": raw, "trade_intent": trade_intent, "logp_u": logp, "entropy_u": entropy, "value_u": value}


class JainPPOPortfolioAgent(nn.Module):
    """Container for Jain's d/u PPO-SIL architecture adapted to portfolio impulses."""

    def __init__(self, obs_dim: int, action_dim: int, cfg: PPOSILConfig, max_trade_size: torch.Tensor) -> None:
        super().__init__()
        self.decision = DecisionActorCritic(obs_dim, cfg)
        self.impulse = ImpulseActorCritic(obs_dim, action_dim, cfg, max_trade_size)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        d = self.decision.act(obs, deterministic=deterministic)
        u = self.impulse.act(obs, deterministic=deterministic)
        out = {**d, **u}
        out["joint_logp"] = out["logp_d"] + out["intervene"].to(obs.dtype) * out["logp_u"]
        out["joint_entropy"] = out["entropy_d"] + out["prob_d"].detach() * out["entropy_u"]
        return out

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        intervene: torch.Tensor,
        raw_action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        logp_d, ent_d, val_d, prob_d = self.decision.evaluate_actions(obs, intervene)
        logp_u, ent_u, val_u = self.impulse.evaluate_actions(obs, raw_action)
        mask = intervene.to(obs.dtype)
        return {
            "logp_d": logp_d,
            "entropy_d": ent_d,
            "value_d": val_d,
            "prob_d": prob_d,
            "logp_u": logp_u,
            "entropy_u": ent_u,
            "value_u": val_u,
            "joint_logp": logp_d + mask * logp_u,
            "joint_entropy": ent_d + prob_d.detach() * ent_u,
        }


# ---------------------------------------------------------------------------
# Rollouts, advantages and SIL replay
# ---------------------------------------------------------------------------


@torch.no_grad()
def rollout_batch(
    agent: JainPPOPortfolioAgent,
    env: VarianceGammaPortfolioEnv,
    n_envs: int,
    deterministic: bool = False,
) -> Dict[str, torch.Tensor]:
    agent.eval()
    steps = int(env.cfg.n_steps)
    state = env.reset(int(n_envs))
    initial_wealth = env.wealth(state)

    obs_buf = torch.empty(steps, n_envs, env.obs_dim, device=env.device, dtype=env.dtype)
    intervene_buf = torch.empty(steps, n_envs, device=env.device, dtype=torch.long)
    raw_action_buf = torch.empty(steps, n_envs, env.action_dim, device=env.device, dtype=env.dtype)
    trade_intent_buf = torch.empty_like(raw_action_buf)
    safe_trade_buf = torch.empty_like(raw_action_buf)
    logp_d_buf = torch.empty(steps, n_envs, device=env.device, dtype=env.dtype)
    logp_u_buf = torch.empty(steps, n_envs, device=env.device, dtype=env.dtype)
    value_d_buf = torch.empty(steps, n_envs, device=env.device, dtype=env.dtype)
    value_u_buf = torch.empty(steps, n_envs, device=env.device, dtype=env.dtype)
    reward_buf = torch.empty(steps, n_envs, device=env.device, dtype=env.dtype)
    reward_actual_buf = torch.empty(steps, n_envs, device=env.device, dtype=env.dtype)
    cost_buf = torch.empty(steps, n_envs, device=env.device, dtype=env.dtype)
    trade_count_buf = torch.empty(steps, n_envs, device=env.device, dtype=env.dtype)
    mean_abs_trade_buf = torch.empty(steps, n_envs, device=env.device, dtype=env.dtype)

    for t in range(steps):
        obs = env.observe(state)
        act = agent.act(obs, deterministic=deterministic)
        next_state, reward, _done, info = env.step(state, act["intervene"], act["trade_intent"])
        _finite_or_raise("observation", obs)
        _finite_or_raise("reward", reward)

        obs_buf[t] = obs
        intervene_buf[t] = act["intervene"]
        raw_action_buf[t] = act["raw_action"]
        trade_intent_buf[t] = act["trade_intent"]
        safe_trade_buf[t] = info["safe_trade"]
        logp_d_buf[t] = act["logp_d"]
        logp_u_buf[t] = act["logp_u"]
        value_d_buf[t] = act["value_d"]
        value_u_buf[t] = act["value_u"]
        reward_buf[t] = reward
        reward_actual_buf[t] = info["reward_actual"]
        cost_buf[t] = info["cost"]
        trade_count_buf[t] = info["trade_indicator"]
        mean_abs_trade_buf[t] = info["mean_abs_trade"]
        state = next_state

    terminal_wealth = env.wealth(state)
    discounted_terminal_wealth = env.discounted_wealth(state)
    discounted_profit = discounted_terminal_wealth - initial_wealth
    return {
        "obs": obs_buf,
        "intervene": intervene_buf,
        "raw_action": raw_action_buf,
        "trade_intent": trade_intent_buf,
        "safe_trade": safe_trade_buf,
        "old_logp_d": logp_d_buf,
        "old_logp_u": logp_u_buf,
        "value_d": value_d_buf,
        "value_u": value_u_buf,
        "reward": reward_buf,
        "reward_actual": reward_actual_buf,
        "initial_wealth": initial_wealth,
        "terminal_wealth": terminal_wealth,
        "discounted_terminal_wealth": discounted_terminal_wealth,
        "discounted_profit": discounted_profit,
        "trade_count": trade_count_buf.sum(dim=0),
        "total_cost": cost_buf.sum(dim=0),
        "mean_abs_trade": mean_abs_trade_buf.mean(dim=0),
        "final_h": state["h"],
        "final_s": state["s"],
        "final_cash": state["cash"],
    }


def _gae_and_returns(reward: torch.Tensor, value: torch.Tensor, cfg: PPOSILConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    steps, n_envs = reward.shape
    returns = torch.zeros_like(reward)
    running = torch.zeros(n_envs, device=reward.device, dtype=reward.dtype)
    for t in range(steps - 1, -1, -1):
        running = reward[t] + float(cfg.gamma_rl) * running
        returns[t] = running

    adv = torch.zeros_like(reward)
    last = torch.zeros(n_envs, device=reward.device, dtype=reward.dtype)
    next_value = torch.zeros(n_envs, device=reward.device, dtype=reward.dtype)
    for t in range(steps - 1, -1, -1):
        delta = reward[t] + float(cfg.gamma_rl) * next_value - value[t]
        last = delta + float(cfg.gamma_rl) * float(cfg.gae_lambda) * last
        adv[t] = last
        next_value = value[t]
    adv_norm = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(float(cfg.advantage_eps))
    return returns, adv, adv_norm


def compute_advantages(rollout: Dict[str, torch.Tensor], cfg: PPOSILConfig) -> None:
    ret_d, adv_d_raw, adv_d = _gae_and_returns(rollout["reward"], rollout["value_d"], cfg)
    ret_u, adv_u_raw, adv_u = _gae_and_returns(rollout["reward"], rollout["value_u"], cfg)
    rollout["mc_return"] = ret_d
    rollout["return_d"] = ret_d
    rollout["return_u"] = ret_u
    rollout["advantage_d"] = adv_d
    rollout["advantage_u"] = adv_u
    rollout["advantage_d_raw"] = adv_d_raw
    rollout["advantage_u_raw"] = adv_u_raw


def flatten_rollout(rollout: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    skip = {
        "initial_wealth", "terminal_wealth", "discounted_terminal_wealth", "discounted_profit",
        "trade_count", "total_cost", "mean_abs_trade", "final_h", "final_s", "final_cash",
    }
    out: Dict[str, torch.Tensor] = {}
    for k, v in rollout.items():
        if not isinstance(v, torch.Tensor) or k in skip:
            continue
        if v.ndim >= 3:
            out[k] = v.reshape(-1, *v.shape[2:])
        elif v.ndim == 2:
            out[k] = v.reshape(-1)
        else:
            out[k] = v
    return out


class PrioritizedSILBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device, dtype: torch.dtype, alpha: float, eps: float, keep_on_cpu: bool = True) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.storage_device = torch.device("cpu") if keep_on_cpu else device
        self.dtype = dtype
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.obs = torch.empty(self.capacity, obs_dim, device=self.storage_device, dtype=dtype)
        self.intervene = torch.empty(self.capacity, device=self.storage_device, dtype=torch.long)
        self.raw_action = torch.empty(self.capacity, action_dim, device=self.storage_device, dtype=dtype)
        self.mc_return = torch.empty(self.capacity, device=self.storage_device, dtype=dtype)
        self.value_d = torch.empty(self.capacity, device=self.storage_device, dtype=dtype)
        self.value_u = torch.empty(self.capacity, device=self.storage_device, dtype=dtype)
        self.priority = torch.full((self.capacity,), self.eps, device=self.storage_device, dtype=torch.float32)
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    @torch.no_grad()
    def add_batch(self, obs: torch.Tensor, intervene: torch.Tensor, raw_action: torch.Tensor, mc_return: torch.Tensor, value_d: torch.Tensor, value_u: torch.Tensor, priority: torch.Tensor) -> None:
        n = int(obs.shape[0])
        if n <= 0:
            return
        if n > self.capacity:
            obs = obs[-self.capacity:]
            intervene = intervene[-self.capacity:]
            raw_action = raw_action[-self.capacity:]
            mc_return = mc_return[-self.capacity:]
            value_d = value_d[-self.capacity:]
            value_u = value_u[-self.capacity:]
            priority = priority[-self.capacity:]
            n = self.capacity
        idx = (torch.arange(n, device=self.storage_device) + self.ptr) % self.capacity
        self.obs[idx] = obs.detach().to(self.storage_device, non_blocking=False)
        self.intervene[idx] = intervene.detach().to(self.storage_device, non_blocking=False).long()
        self.raw_action[idx] = raw_action.detach().to(self.storage_device, non_blocking=False)
        self.mc_return[idx] = mc_return.detach().to(self.storage_device, non_blocking=False)
        self.value_d[idx] = value_d.detach().to(self.storage_device, non_blocking=False)
        self.value_u[idx] = value_u.detach().to(self.storage_device, non_blocking=False)
        self.priority[idx] = priority.detach().to(self.storage_device, torch.float32, non_blocking=False).clamp_min(self.eps)
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = min(self.capacity, self.size + n)

    @torch.no_grad()
    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if self.size <= 0:
            raise RuntimeError("SIL buffer is empty.")
        raw = self.priority[: self.size].pow(self.alpha)
        if (not torch.isfinite(raw).all()) or float(raw.sum().detach().cpu()) <= 0.0:
            prob = torch.full((self.size,), 1.0 / float(self.size), device=self.storage_device, dtype=torch.float32)
        else:
            prob = raw / raw.sum()
        idx = torch.multinomial(prob, int(batch_size), replacement=True)
        batch = {
            "obs": self.obs[idx].to(self.device, non_blocking=False),
            "intervene": self.intervene[idx].to(self.device, non_blocking=False),
            "raw_action": self.raw_action[idx].to(self.device, non_blocking=False),
            "mc_return": self.mc_return[idx].to(self.device, non_blocking=False),
            "value_d": self.value_d[idx].to(self.device, non_blocking=False),
            "value_u": self.value_u[idx].to(self.device, non_blocking=False),
        }
        return batch, idx

    @torch.no_grad()
    def update_priority(self, idx: torch.Tensor, priority: torch.Tensor) -> None:
        idx = idx.to(self.storage_device)
        self.priority[idx] = priority.detach().to(self.storage_device, torch.float32, non_blocking=False).clamp_min(self.eps)


@torch.no_grad()
def add_to_sil(buffer: PrioritizedSILBuffer, rollout: Dict[str, torch.Tensor], cfg: PPOSILConfig) -> None:
    flat = flatten_rollout(rollout)
    gap_d = (flat["mc_return"] - flat["value_d"] - float(cfg.sil_margin)).clamp_min(0.0)
    gap_u = (flat["mc_return"] - flat["value_u"] - float(cfg.sil_margin)).clamp_min(0.0)
    priority = torch.maximum(gap_d, gap_u) + float(cfg.sil_priority_eps)
    buffer.add_batch(flat["obs"], flat["intervene"], flat["raw_action"], flat["mc_return"], flat["value_d"], flat["value_u"], priority)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def _safe_mean(x: torch.Tensor, default: float = 0.0) -> torch.Tensor:
    if x.numel() == 0:
        return torch.tensor(default, device=x.device, dtype=x.dtype)
    return x.mean()


def _ppo_minibatch_loss(agent: JainPPOPortfolioAgent, batch: Dict[str, torch.Tensor], cfg: PPOSILConfig) -> Tuple[torch.Tensor, Dict[str, float]]:
    ev = agent.evaluate_actions(batch["obs"], batch["intervene"], batch["raw_action"])
    intervene_f = batch["intervene"].to(batch["obs"].dtype)

    ratio_d = torch.exp(ev["logp_d"] - batch["old_logp_d"])
    surr1_d = ratio_d * batch["advantage_d"]
    surr2_d = ratio_d.clamp(1.0 - float(cfg.clip_eps), 1.0 + float(cfg.clip_eps)) * batch["advantage_d"]
    d_policy_loss = -torch.min(surr1_d, surr2_d).mean()
    d_value_loss = 0.5 * (ev["value_d"] - batch["return_d"]).pow(2).mean()

    u_mask = intervene_f > 0.5
    if u_mask.any():
        ratio_u = torch.exp(ev["logp_u"][u_mask] - batch["old_logp_u"][u_mask])
        adv_u = batch["advantage_u"][u_mask]
        surr1_u = ratio_u * adv_u
        surr2_u = ratio_u.clamp(1.0 - float(cfg.clip_eps), 1.0 + float(cfg.clip_eps)) * adv_u
        u_policy_loss = -torch.min(surr1_u, surr2_u).mean()
        u_entropy = ev["entropy_u"][u_mask].mean()
        kl_u = (batch["old_logp_u"][u_mask] - ev["logp_u"][u_mask]).mean().abs()
    else:
        u_policy_loss = torch.zeros((), device=batch["obs"].device, dtype=batch["obs"].dtype)
        u_entropy = torch.zeros((), device=batch["obs"].device, dtype=batch["obs"].dtype)
        kl_u = torch.zeros((), device=batch["obs"].device, dtype=batch["obs"].dtype)
    u_value_loss = 0.5 * (ev["value_u"] - batch["return_u"]).pow(2).mean()

    d_entropy = ev["entropy_d"].mean()
    loss = (
        d_policy_loss
        + u_policy_loss
        + float(cfg.value_coef) * (d_value_loss + u_value_loss)
        - float(cfg.entropy_coef) * (d_entropy + u_entropy)
    )
    kl_d = (batch["old_logp_d"] - ev["logp_d"]).mean().abs()
    stats = {
        "d_policy_loss": float(d_policy_loss.detach().cpu()),
        "u_policy_loss": float(u_policy_loss.detach().cpu()),
        "d_value_loss": float(d_value_loss.detach().cpu()),
        "u_value_loss": float(u_value_loss.detach().cpu()),
        "d_entropy": float(d_entropy.detach().cpu()),
        "u_entropy": float(u_entropy.detach().cpu()),
        "kl_d": float(kl_d.detach().cpu()),
        "kl_u": float(kl_u.detach().cpu()),
    }
    return loss, stats


def _sil_update(agent: JainPPOPortfolioAgent, opt: torch.optim.Optimizer, sil: PrioritizedSILBuffer, cfg: PPOSILConfig) -> Dict[str, float]:
    batch, idx = sil.sample(int(cfg.sil_batch_size))
    ev = agent.evaluate_actions(batch["obs"], batch["intervene"], batch["raw_action"])
    gap_d = (batch["mc_return"] - ev["value_d"] - float(cfg.sil_margin)).clamp_min(0.0)
    gap_u = (batch["mc_return"] - ev["value_u"] - float(cfg.sil_margin)).clamp_min(0.0)
    pos_d = gap_d > 0.0
    pos_u = (gap_u > 0.0) & (batch["intervene"] > 0)

    zero = torch.zeros((), device=batch["obs"].device, dtype=batch["obs"].dtype)
    if pos_d.any():
        d_policy = -(gap_d[pos_d].detach() * ev["logp_d"][pos_d]).sum() / gap_d[pos_d].detach().sum().clamp_min(1.0)
        d_value = 0.5 * gap_d[pos_d].pow(2).mean()
    else:
        d_policy = zero
        d_value = zero
    if pos_u.any():
        u_policy = -(gap_u[pos_u].detach() * ev["logp_u"][pos_u]).sum() / gap_u[pos_u].detach().sum().clamp_min(1.0)
        u_value = 0.5 * gap_u[pos_u].pow(2).mean()
    else:
        u_policy = zero
        u_value = zero

    loss = (
        float(cfg.sil_decision_policy_coef) * d_policy
        + float(cfg.sil_decision_value_coef) * d_value
        + float(cfg.sil_impulse_policy_coef) * u_policy
        + float(cfg.sil_impulse_value_coef) * u_value
    )
    if torch.isfinite(loss) and loss.requires_grad:
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), float(cfg.max_grad_norm))
        opt.step()

    priority = torch.maximum(gap_d.detach(), gap_u.detach()) + float(cfg.sil_priority_eps)
    sil.update_priority(idx, priority)
    return {
        "sil_d_policy_loss": float(d_policy.detach().cpu()),
        "sil_u_policy_loss": float(u_policy.detach().cpu()),
        "sil_d_value_loss": float(d_value.detach().cpu()),
        "sil_u_value_loss": float(u_value.detach().cpu()),
    }


@torch.no_grad()
def evaluate_policy(agent: JainPPOPortfolioAgent, env: VarianceGammaPortfolioEnv, n_envs: int, deterministic: bool = True, batch_size: Optional[int] = None) -> Dict[str, Any]:
    n_envs = int(n_envs)
    batch_size = int(batch_size or n_envs)
    terminal, disc_terminal, disc_profit, trades, costs, abs_trades = [], [], [], [], [], []
    h_sum = None
    s_sum = None
    cash_sum = None
    left = n_envs
    while left > 0:
        b = min(left, batch_size)
        rollout = rollout_batch(agent, env, b, deterministic=deterministic)
        terminal.append(rollout["terminal_wealth"].detach().cpu())
        disc_terminal.append(rollout["discounted_terminal_wealth"].detach().cpu())
        disc_profit.append(rollout["discounted_profit"].detach().cpu())
        trades.append(rollout["trade_count"].detach().cpu())
        costs.append(rollout["total_cost"].detach().cpu())
        abs_trades.append(rollout["mean_abs_trade"].detach().cpu())
        h_block = rollout["final_h"].detach().cpu().sum(dim=0)
        s_block = rollout["final_s"].detach().cpu().sum(dim=0)
        cash_block = rollout["final_cash"].detach().cpu().sum()
        h_sum = h_block if h_sum is None else h_sum + h_block
        s_sum = s_block if s_sum is None else s_sum + s_block
        cash_sum = cash_block if cash_sum is None else cash_sum + cash_block
        left -= b
        _clear_device_cache(env.device)
    terminal_t = torch.cat(terminal)
    disc_terminal_t = torch.cat(disc_terminal)
    disc_profit_t = torch.cat(disc_profit)
    trades_t = torch.cat(trades)
    costs_t = torch.cat(costs)
    abs_trade_t = torch.cat(abs_trades)
    return {
        "terminal_wealth_mean": float(terminal_t.mean()),
        "terminal_wealth_std": float(terminal_t.std(unbiased=False)),
        "discounted_terminal_wealth_mean": float(disc_terminal_t.mean()),
        "discounted_terminal_wealth_std": float(disc_terminal_t.std(unbiased=False)),
        "discounted_profit_mean": float(disc_profit_t.mean()),
        "discounted_profit_std": float(disc_profit_t.std(unbiased=False)),
        "trade_count_mean": float(trades_t.mean()),
        "total_cost_mean": float(costs_t.mean()),
        "mean_abs_trade": float(abs_trade_t.mean()),
        "final_holdings_mean": (h_sum / float(n_envs)).numpy(),
        "final_prices_mean": (s_sum / float(n_envs)).numpy(),
        "final_cash_mean": float(cash_sum / float(n_envs)),
    }


def train_ppo_sil(market: Optional[MarketConfig] = None, cfg: Optional[PPOSILConfig] = None) -> Dict[str, Any]:
    market = market or MarketConfig()
    cfg = cfg or PPOSILConfig()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    dtype = getattr(torch, cfg.dtype)
    if device.type == "mps" and dtype != torch.float32:
        dtype = torch.float32

    env = VarianceGammaPortfolioEnv(market, device=device, dtype=dtype)
    agent = JainPPOPortfolioAgent(env.obs_dim, env.action_dim, cfg, env.max_trade_size).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(agent.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    sil = PrioritizedSILBuffer(
        int(cfg.sil_buffer_capacity), env.obs_dim, env.action_dim, device, dtype,
        float(cfg.sil_priority_alpha), float(cfg.sil_priority_eps), keep_on_cpu=bool(cfg.keep_sil_on_cpu)
    )

    history: Dict[str, List[Any]] = {k: [] for k in [
        "iter", "d_policy_loss", "u_policy_loss", "d_value_loss", "u_value_loss",
        "d_entropy", "u_entropy", "kl_d", "kl_u", "sil_d_policy_loss", "sil_u_policy_loss",
        "sil_d_value_loss", "sil_u_value_loss", "train_terminal_wealth_mean",
        "train_discounted_terminal_wealth_mean", "train_discounted_profit_mean",
        "train_trade_count_mean", "train_total_cost_mean", "train_intervention_rate",
        "train_mean_abs_trade", "eval_terminal_wealth_mean", "eval_discounted_terminal_wealth_mean",
        "eval_discounted_profit_mean", "eval_trade_count_mean", "eval_total_cost_mean", "eval_mean_abs_trade",
    ]}

    best_eval = -float("inf")
    ckpt_path = Path(cfg.checkpoint_path) if cfg.checkpoint_path else None
    best_path = ckpt_path.with_name("best_" + ckpt_path.name) if ckpt_path is not None else None

    for it in range(1, int(cfg.train_iters) + 1):
        rollout = rollout_batch(agent, env, int(cfg.rollout_envs), deterministic=False)
        compute_advantages(rollout, cfg)
        if cfg.use_sil:
            add_to_sil(sil, rollout, cfg)
        flat = flatten_rollout(rollout)
        total = int(flat["obs"].shape[0])

        acc = {k: 0.0 for k in ["d_policy_loss", "u_policy_loss", "d_value_loss", "u_value_loss", "d_entropy", "u_entropy", "kl_d", "kl_u"]}
        ppo_steps = 0
        stop_early = False
        agent.train()
        for _epoch in range(int(cfg.ppo_epochs)):
            perm = torch.randperm(total, device=device)
            for start in range(0, total, int(cfg.minibatch_size)):
                idx = perm[start: start + int(cfg.minibatch_size)]
                if idx.numel() == 0:
                    continue
                batch = {k: v[idx] for k, v in flat.items() if k in {
                    "obs", "intervene", "raw_action", "old_logp_d", "old_logp_u", "advantage_d", "advantage_u", "return_d", "return_u"
                }}
                loss, stats = _ppo_minibatch_loss(agent, batch, cfg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), float(cfg.max_grad_norm))
                opt.step()
                for k in acc:
                    acc[k] += stats[k]
                ppo_steps += 1
                if stats["kl_d"] > float(cfg.target_kl) or stats["kl_u"] > float(cfg.target_kl):
                    stop_early = True
                    break
            if stop_early:
                break

        sil_acc = {k: 0.0 for k in ["sil_d_policy_loss", "sil_u_policy_loss", "sil_d_value_loss", "sil_u_value_loss"]}
        sil_steps = 0
        if cfg.use_sil and it >= int(cfg.sil_warmup_iters) and len(sil) >= int(cfg.sil_batch_size):
            for _ in range(int(cfg.sil_updates_per_iter)):
                stats = _sil_update(agent, opt, sil, cfg)
                for k in sil_acc:
                    sil_acc[k] += stats[k]
                sil_steps += 1

        do_eval = (it == 1) or (it % max(1, int(cfg.eval_every)) == 0) or (it == int(cfg.train_iters))
        eval_stats = {k: float("nan") for k in [
            "terminal_wealth_mean", "discounted_terminal_wealth_mean", "discounted_profit_mean",
            "trade_count_mean", "total_cost_mean", "mean_abs_trade"
        ]}
        if do_eval:
            eval_stats = evaluate_policy(agent, env, int(cfg.eval_envs), deterministic=bool(cfg.eval_deterministic), batch_size=int(cfg.eval_batch_size))
            if eval_stats["discounted_terminal_wealth_mean"] > best_eval:
                best_eval = eval_stats["discounted_terminal_wealth_mean"]
                if best_path is not None:
                    save_ppo_sil(agent, env, market, cfg, history, best_path)

        history["iter"].append(it)
        for k in acc:
            history[k].append(acc[k] / max(1, ppo_steps))
        for k in sil_acc:
            history[k].append(sil_acc[k] / max(1, sil_steps))
        history["train_terminal_wealth_mean"].append(float(rollout["terminal_wealth"].mean().detach().cpu()))
        history["train_discounted_terminal_wealth_mean"].append(float(rollout["discounted_terminal_wealth"].mean().detach().cpu()))
        history["train_discounted_profit_mean"].append(float(rollout["discounted_profit"].mean().detach().cpu()))
        history["train_trade_count_mean"].append(float(rollout["trade_count"].mean().detach().cpu()))
        history["train_total_cost_mean"].append(float(rollout["total_cost"].mean().detach().cpu()))
        history["train_intervention_rate"].append(float(rollout["intervene"].float().mean().detach().cpu()))
        history["train_mean_abs_trade"].append(float(rollout["mean_abs_trade"].mean().detach().cpu()))
        history["eval_terminal_wealth_mean"].append(eval_stats["terminal_wealth_mean"])
        history["eval_discounted_terminal_wealth_mean"].append(eval_stats["discounted_terminal_wealth_mean"])
        history["eval_discounted_profit_mean"].append(eval_stats["discounted_profit_mean"])
        history["eval_trade_count_mean"].append(eval_stats["trade_count_mean"])
        history["eval_total_cost_mean"].append(eval_stats["total_cost_mean"])
        history["eval_mean_abs_trade"].append(eval_stats["mean_abs_trade"])

        if it == 1 or it % max(1, int(cfg.log_every)) == 0 or it == int(cfg.train_iters):
            print(
                f"[PPO-SIL iter {it:04d}/{cfg.train_iters}] "
                f"d_pi={history['d_policy_loss'][-1]:.3e} u_pi={history['u_policy_loss'][-1]:.3e} "
                f"d_v={history['d_value_loss'][-1]:.3e} u_v={history['u_value_loss'][-1]:.3e} "
                f"ent_d={history['d_entropy'][-1]:.3f} ent_u={history['u_entropy'][-1]:.3f} "
                f"kl=({history['kl_d'][-1]:.2e},{history['kl_u'][-1]:.2e}) "
                f"train_discWT={history['train_discounted_terminal_wealth_mean'][-1]:.2f} "
                f"trades={history['train_trade_count_mean'][-1]:.2f} "
                f"eval_discWT={history['eval_discounted_terminal_wealth_mean'][-1]:.2f}"
            )
        if cfg.clear_cache:
            _clear_device_cache(device)

    if ckpt_path is not None:
        save_ppo_sil(agent, env, market, cfg, history, ckpt_path)
    return {"model": agent, "agent": agent, "env": env, "history": history, "market_config": market, "train_config": cfg}


# Backward-compatible alias.
train_ppo = train_ppo_sil


def save_ppo_sil(agent: JainPPOPortfolioAgent, env: VarianceGammaPortfolioEnv, market: MarketConfig, cfg: PPOSILConfig, history: Dict[str, Any], path: str | Path) -> None:
    payload = {
        "state_dict": agent.state_dict(),
        "market_config": asdict(market),
        "train_config": asdict(cfg),
        "obs_dim": env.obs_dim,
        "action_dim": env.action_dim,
        "history": history,
    }
    torch.save(payload, str(path))


def load_ppo_sil(path: str | Path, device: Optional[str] = None) -> Dict[str, Any]:
    dev = get_device(device)
    payload = torch.load(str(path), map_location=dev)
    market = MarketConfig(**payload["market_config"])
    cfg = PPOSILConfig(**payload["train_config"])
    dtype = getattr(torch, cfg.dtype)
    if dev.type == "mps" and dtype != torch.float32:
        dtype = torch.float32
    env = VarianceGammaPortfolioEnv(market, device=dev, dtype=dtype)
    agent = JainPPOPortfolioAgent(env.obs_dim, env.action_dim, cfg, env.max_trade_size).to(device=dev, dtype=dtype)
    agent.load_state_dict(payload["state_dict"])
    agent.eval()
    return {"model": agent, "agent": agent, "env": env, "market_config": market, "train_config": cfg, "history": payload.get("history", {})}


if __name__ == "__main__":
    # Small CPU smoke test.
    market = MarketConfig(
        d=2, n_steps=8, h0=[10, 10], cash0=5000.0, fixed_cost=0.5, proportional_cost=0.0005,
        max_holding=20, max_trade_size=10, reward_scale=1e-3,
        sigma=[0.15, 0.15], theta=[0.0, 0.0], nu=[0.20, 0.20],
        log_returns_corr_matrix=[[1.0, 0.0], [0.0, 1.0]], ssg_path_mean_samples=2,
    )
    cfg = PPOSILConfig(device="cpu", train_iters=2, rollout_envs=16, eval_envs=16, eval_batch_size=16,
                       ppo_epochs=1, minibatch_size=32, hidden_dims=(32, 32), sil_batch_size=32,
                       sil_warmup_iters=1, log_every=1, eval_every=1, checkpoint_path="/tmp/PPO_SIL_pfolio_cuda_smoke.pt")
    result = train_ppo_sil(market, cfg)
    print(evaluate_policy(result["agent"], result["env"], 32, deterministic=True))
