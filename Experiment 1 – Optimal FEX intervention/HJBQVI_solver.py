#explicit-impulse scheme in Azimzadeh

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, diags, lil_matrix
from scipy.sparse.linalg import splu, spsolve

@dataclass(slots=True)
class ExchangeRateExplicitImpulseParams:
    mu: float=0.25
    sigma: float=0.3

    m: float=0.0
    beta: float=0.02
    gamma: float=3.0
    kappa: float=1.0
    c: float=0.1

    T: float=1.0
    w_const: float=0.04

    x_min: float=-2.0
    x_max: float=2.0
    n_x: int=257
    n_zeta: int=200
    n_t: int=512

@dataclass(slots=True)
class ExchangeRateExplicitImpulseResult:
    params: ExchangeRateExplicitImpulseParams
    x_grid: np.ndarray
    impulse_grid: np.ndarray
    t_grid: np.ndarray
    values: np.ndarray
    continuation_value_t0: np.ndarray
    intervention_value_t0: np.ndarray
    intervention_active_t0: np.ndarray
    best_impulse_target_t0: np.ndarray
    best_impulse_jump_t0: np.ndarray

class ExchangeRateExplicitImpulseSolver:
    def __init__(self, params: ExchangeRateExplicitImpulseParams)->None:
        self.params = params
        self.dt = params.T/params.n_t
        
        self.t_grid = np.linspace(0.0, params.T, params.n_t+1)
        self.x_grid = np.linspace(params.x_min, params.x_max, params.n_x+1)
        self.zeta_grid = np.linspace(params.x_min, params.x_max, params.n_zeta+1)

        self.continuation_interpolation = interpolation_matrix(self.x_grid, self.x_grid - self.params.w_const*self.params.mu*self.dt)

        self.target_eval = interpolation_matrix(self.x_grid, self.zeta_grid)
        self.diffusion_matrix = build_diffusion_matrix(self.x_grid, params.sigma)

        self.system_matrix = self.build_system_matrix()
        self.solve_linear = self.build_linear_solver()

        self.abs_jump = np.abs(self.zeta_grid[None, :] - self.x_grid[:, None])
        self.admissible_impulse = np.ones((len(self.x_grid), len(self.zeta_grid)), dtype=bool)
    
    def build_system_matrix(self)->csr_matrix:
        A = diags(np.ones(len(self.x_grid)), 0, format="csr") - self.dt*self.diffusion_matrix
        return A.tocsr()
    
    def build_linear_solver(self):
        return splu(csc_matrix(self.system_matrix)).solve

    def running_flow(self, t:float)->np.ndarray:
        p = self.params
        base = -((self.x_grid - p.m)**2 + p.gamma*p.w_const**2)
        return np.exp(-p.beta*t)*base
    
    def impulse_reward_matrix(self, t:float)->np.ndarray:
        p = self.params
        reward = -(p.kappa*self.abs_jump + p.c)
        return np.exp(-p.beta*t)*reward
    
    def continuation_value(self, v_next: np.ndarray, t:float)->np.ndarray:
        return self.continuation_interpolation@v_next + self.dt*self.running_flow(t)
    
    def intervention_value(self, v_next: np.ndarray, t:float)->tuple[np.ndarray, np.ndarray]:
        target_values = self.target_eval@v_next
        reward = self.impulse_reward_matrix(t)

        candidates = target_values[None,:] + reward
        feasibile_candidates = np.where(self.admissible_impulse, candidates, -np.inf)

        best_index = np.argmax(candidates, axis=1)
        best_value = feasibile_candidates[np.arange(len(self.x_grid)), best_index]
        return best_value, best_index
    
    def solve(self)->ExchangeRateExplicitImpulseResult:
        p = self.params
        values = np.zeros((p.n_t+1, len(self.x_grid)))
        values[-1] = 0

        continuation_value_t0: Optional[np.ndarray]=None
        intervention_value_t0: Optional[np.ndarray]=None
        best_target_t0: Optional[np.ndarray]=None

        for n in range(p.n_t - 1, -1, -1):
            t = self.t_grid[n]
            v_next = values[n + 1]

            continuation_value = self.continuation_value(v_next, t)
            intervention_value, best_idx = self.intervention_value(v_next, t)
            rhs = np.maximum(continuation_value, intervention_value)
            values[n] = self.solve_linear(rhs)

            if n == 0:
                continuation_value_t0 = continuation_value.copy()
                intervention_value_t0 = intervention_value.copy()
                best_target_t0 = self.zeta_grid[best_idx].copy()

        if continuation_value_t0 is None or intervention_value_t0 is None or best_target_t0 is None:
            raise RuntimeError("solver failed to populate t=0 diagnostics")

        active = intervention_value_t0 > continuation_value_t0
        best_jump = best_target_t0 - self.x_grid

        return ExchangeRateExplicitImpulseResult(params=p, x_grid=self.x_grid, impulse_grid=self.zeta_grid, t_grid=self.t_grid, values=values, continuation_value_t0=continuation_value_t0, intervention_value_t0=intervention_value_t0, intervention_active_t0=active, best_impulse_target_t0=best_target_t0, best_impulse_jump_t0=best_jump)

def interpolation_matrix(grid: np.ndarray, target_points: np.ndarray)->csr_matrix:
    rows: list[int]=[]
    cols: list[int]=[]
    data: list[float]=[]

    n = len(grid)

    for row, y in enumerate(target_points):
        if(y<=grid[0]):
            rows.append(row)
            cols.append(0)
            data.append(1.0)
            continue
        
        if(y>=grid[-1]):
            rows.append(row)
            cols.append(n-1)
            data.append(1.0)
            continue
        
        k = int(np.searchsorted(grid, y, side='right')-1)
        x0 = grid[k]
        x1 = grid[k+1]
        w1 = (y-x0)/(x1-x0)
        w0 = 1 - w1

        rows.extend((row, row))
        cols.extend((k, k+1))
        data.extend((w0, w1))

    return csr_matrix((data, (rows, cols)), shape=(len(target_points), n))

def build_diffusion_matrix(x: np.ndarray, sigma: float)->csr_matrix:
    n = len(x)
    A = lil_matrix((n,n))
    a = sigma**2/2

    for i in range(1,n-1):
        dx_b = x[i] - x[i - 1]
        dx_f = x[i + 1] - x[i]
        dx_c = x[i + 1] - x[i - 1]

        left = 2.0 / (dx_b * dx_c)
        center = -2.0 / (dx_b * dx_f)
        right = 2.0 / (dx_f * dx_c)

        A[i, i - 1] = a*left
        A[i, i] = a*center
        A[i, i + 1] = a*right

    return A.tocsr()

def solve_exchange_rate_explicit_impulse(**kwargs) -> ExchangeRateExplicitImpulseResult:
    params = ExchangeRateExplicitImpulseParams(**kwargs)
    return ExchangeRateExplicitImpulseSolver(params).solve()