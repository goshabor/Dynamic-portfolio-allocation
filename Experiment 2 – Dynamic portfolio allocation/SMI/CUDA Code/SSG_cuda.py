import os
import numpy as np
import cvxpy as cp
import torch


def _symmetrize_unit_diag(R):
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("target_return_corr must be a square matrix.")
    if not np.all(np.isfinite(R)):
        raise ValueError("target_return_corr contains NaN or inf.")

    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return R


def lms_vg_model_corr(a, C, theta_vec, sigma_vec, nu_vec):
    """
    Return the model-implied log-return correlation matrix for the rho-alpha VG model.

    Parameters are in the notation of your class:
        theta_vec = Brownian drift parameters theta_j
        sigma_vec = Brownian vol parameters sigma_j
        nu_vec    = VG activity/kurtosis parameters nu_j, i.e. alpha_j in LMS
        C         = Brownian correlation matrix
        a         = common gamma subordinator parameter
    """
    theta = np.asarray(theta_vec, dtype=float)
    sigma = np.asarray(sigma_vec, dtype=float)
    nu = np.asarray(nu_vec, dtype=float)
    C = np.asarray(C, dtype=float)

    v = sigma**2 + theta**2 * nu
    theta_nu = theta * nu
    sigma_sqrt_nu = sigma * np.sqrt(nu)

    numerator = (
        np.outer(theta_nu, theta_nu)
        + C * np.outer(sigma_sqrt_nu, sigma_sqrt_nu)
    )
    denominator = np.sqrt(np.outer(v, v))

    R = a * numerator / denominator
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return R


def lms_vg_brownian_corr_from_fixed_a(
    a,
    target_return_corr,
    theta_vec,
    sigma_vec,
    nu_vec,
    *,
    check_a_bound=True,
):
    """
    Closed-form inversion for C given a.

    This is your current formula, cleaned up. It does NOT guarantee that C is PSD.
    Use estimate_lms_vg_brownian_correlation(...) below for calibrated PSD C.
    """
    R = _symmetrize_unit_diag(target_return_corr)

    theta = np.asarray(theta_vec, dtype=float)
    sigma = np.asarray(sigma_vec, dtype=float)
    nu = np.asarray(nu_vec, dtype=float)

    if np.any(sigma <= 0):
        raise ValueError("sigma_vec must be strictly positive.")
    if np.any(nu <= 0):
        raise ValueError("nu_vec must be strictly positive.")
    if a <= 0:
        raise ValueError("a must be strictly positive.")

    a_upper = np.min(1.0 / nu)
    if check_a_bound and not (a < a_upper):
        raise ValueError(
            f"a must satisfy a < min(1 / nu_vec) = {a_upper:.16g}. "
            "Use a small upper margin."
        )

    v = sigma**2 + theta**2 * nu
    theta_nu = theta * nu
    sigma_sqrt_nu = sigma * np.sqrt(nu)

    C = (
        R * np.sqrt(np.outer(v, v)) / a
        - np.outer(theta_nu, theta_nu)
    ) / np.outer(sigma_sqrt_nu, sigma_sqrt_nu)

    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return C


def estimate_lms_vg_brownian_correlation(
    target_return_corr,
    theta_vec,
    sigma_vec,
    nu_vec,
    *,
    a_upper_margin=1e-5,
    a_lower_fraction=1e-5,
    pd_epsilon=1e-6,
    maximize_a_after_rmse=True,
    loss_tie_abs=1e-12,
    loss_tie_rel=1e-8,
    solver=None,
    solver_opts=None,
    return_info=True,
):
    """
    Calibrate the Brownian correlation matrix C and common subordinator parameter a
    for the Luciano-Marena-Semeraro rho-alpha VG model.

    This solves

        min_{a, C} sum_{i<j} (rho_emp_ij - rho_model_ij(a, C))^2

    subject to

        0 < a < min_j 1 / nu_j,
        C is a valid correlation matrix.

    No Higham nearest-correlation projection is used.

    Convex reformulation:
        Q = a C
        Q PSD, diag(Q) = a.

    Returns
    -------
    a : float
        Calibrated common gamma subordinator parameter.

    C : ndarray, shape (d, d)
        Calibrated Brownian correlation matrix.

    info : dict
        Diagnostics. Contains model_return_corr, rmse, max_abs_error, min_eig_C,
        and cholesky_lower.
    """
    R_target = _symmetrize_unit_diag(target_return_corr)
    d = R_target.shape[0]

    theta = np.asarray(theta_vec, dtype=float)
    sigma = np.asarray(sigma_vec, dtype=float)
    nu = np.asarray(nu_vec, dtype=float)

    for name, arr in {
        "theta_vec": theta,
        "sigma_vec": sigma,
        "nu_vec": nu,
    }.items():
        if arr.ndim != 1 or arr.size != d:
            raise ValueError(f"{name} must be a 1D array of length {d}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or inf.")

    if np.any(sigma <= 0):
        raise ValueError("sigma_vec must be strictly positive.")
    if np.any(nu <= 0):
        raise ValueError("nu_vec must be strictly positive.")

    a_upper_theoretical = float(np.min(1.0 / nu))
    a_high = a_upper_theoretical * (1.0 - a_upper_margin)
    a_low = a_upper_theoretical * a_lower_fraction

    if pd_epsilon > 0:
        # Ensures numerical positive definiteness, useful for Cholesky simulation.
        a_low = max(a_low, 1.1 * pd_epsilon * a_upper_theoretical)

    if not (0.0 < a_low < a_high):
        raise ValueError("Invalid a bounds. Reduce a_upper_margin or pd_epsilon.")

    v = sigma**2 + theta**2 * nu
    theta_nu = theta * nu
    sigma_sqrt_nu = sigma * np.sqrt(nu)
    den = np.sqrt(np.outer(v, v))

    drift_part = np.outer(theta_nu, theta_nu) / den
    brownian_part = np.outer(sigma_sqrt_nu, sigma_sqrt_nu) / den

    ii, jj = np.triu_indices(d, k=1)
    target_vec = R_target[ii, jj]
    drift_vec = drift_part[ii, jj]
    brownian_vec = brownian_part[ii, jj]

    # Variables:
    # Q = a C, with Q PSD and diag(Q) = a.
    Q = cp.Variable((d, d), symmetric=True)
    a = cp.Variable()

    q_offdiag = cp.hstack([Q[i, j] for i, j in zip(ii, jj)])
    residual = a * drift_vec + cp.multiply(brownian_vec, q_offdiag) - target_vec
    loss = cp.sum_squares(residual)

    constraints = [
        a >= a_low,
        a <= a_high,
        cp.diag(Q) == a,
    ]

    if pd_epsilon > 0:
        constraints.append(Q >> (pd_epsilon * a_upper_theoretical) * np.eye(d))
    else:
        constraints.append(Q >> 0)

    solve_kwargs = {}
    if solver is not None:
        solve_kwargs["solver"] = solver
    if solver_opts is not None:
        solve_kwargs.update(solver_opts)

    # Stage 1: minimize correlation RMSE.
    problem = cp.Problem(cp.Minimize(loss), constraints)
    problem.solve(**solve_kwargs)

    ok_statuses = {"optimal", "optimal_inaccurate"}
    if problem.status not in ok_statuses:
        raise RuntimeError(f"CVXPY stage-1 solve failed: status={problem.status}")

    loss_star = float(problem.value)

    # Stage 2: optional lexicographic tie-break.
    # The linear correlations may not uniquely identify a. This chooses the largest
    # admissible a among solutions with essentially the same RMSE.
    if maximize_a_after_rmse:
        loss_tol = loss_tie_abs + loss_tie_rel * max(1.0, abs(loss_star))
        problem2 = cp.Problem(
            cp.Minimize(-a),
            constraints + [loss <= loss_star + loss_tol],
        )
        problem2.solve(**solve_kwargs)

        if problem2.status not in ok_statuses:
            # Fall back to the stage-1 solution.
            pass

    a_val = float(np.asarray(a.value).squeeze())
    Q_val = np.asarray(Q.value, dtype=float)

    C = Q_val / a_val
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)

    R_model = lms_vg_model_corr(a_val, C, theta, sigma, nu)
    err = R_model[ii, jj] - target_vec

    eigvals = np.linalg.eigvalsh(C)
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        L = None

    info = {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs_error": float(np.max(np.abs(err))),
        "model_return_corr": R_model,
        "a_theoretical_upper": a_upper_theoretical,
        "a_used_upper": a_high,
        "a_lower": a_low,
        "min_eig_C": float(eigvals.min()),
        "eigvals_C": eigvals,
        "cholesky_lower": L,
        "cvxpy_status": problem.status,
        "stage1_loss": loss_star,
    }

    return a_val, C, info


class SSG:
    """Luciano-Marena-Semeraro rho-alpha VG stock simulator.

    The calibration of ``a`` and ``C`` is intentionally left in NumPy/CVXPY.
    The simulation path is torch-based.  The simulator does not use fixed seeds
    by default, so PPO training receives fresh price paths even when PPO itself
    is seeded for network initialisation and minibatch reproducibility.
    """

    DEFAULT_DEVICE = "cuda"

    def __init__(
        self,
        S0_vec,
        mu_vec,
        sigma_vec,
        theta_vec,
        nu_vec,
        log_returns_corr_matrix,
        T,
        N_t,
        N_MC,
        device=None,
        dtype=torch.float32,
        seed=None,
        return_numpy=False,
        return_torch=None,
        path_mean_samples=10,
        max_simulated_paths_per_chunk=8192,
    ):
        # Keep all calibration inputs as NumPy arrays.  The calibration of a and C
        # below is intentionally unchanged; only the simulation path is torch-based.
        self.S0_vec = np.asarray(S0_vec, dtype=float).reshape(-1)
        self.mu_vec = np.asarray(mu_vec, dtype=float).reshape(-1)
        self.sigma_vec = np.asarray(sigma_vec, dtype=float).reshape(-1)
        self.theta_vec = np.asarray(theta_vec, dtype=float).reshape(-1)
        self.nu_vec = np.asarray(nu_vec, dtype=float).reshape(-1)
        self.log_returns_corr_matrix = np.asarray(log_returns_corr_matrix, dtype=float)

        self.T = float(T)
        self.N_t = int(N_t)
        self.dt = self.T / self.N_t
        self.N_MC = int(N_MC)
        self.d = self.S0_vec.size
        self.path_mean_samples = max(1, int(path_mean_samples))
        self.max_simulated_paths_per_chunk = max(1, int(max_simulated_paths_per_chunk))

        if (
            self.mu_vec.size != self.d
            or self.sigma_vec.size != self.d
            or self.theta_vec.size != self.d
            or self.nu_vec.size != self.d
        ):
            raise ValueError("S0_vec, mu_vec, sigma_vec, theta_vec and nu_vec must have the same length.")
        if self.log_returns_corr_matrix.shape != (self.d, self.d):
            raise ValueError(f"log_returns_corr_matrix must have shape ({self.d}, {self.d}).")
        if self.N_t < 1 or self.T <= 0.0:
            raise ValueError("T must be positive and N_t must be at least 1.")
        if self.N_MC < 1:
            raise ValueError("N_MC must be positive.")

        # ---- Do not change the way a and C are computed. ----
        self.a = 0.9*np.min(1/self.nu_vec)

        self.theta_nu = self.theta_vec * self.nu_vec
        self.sigma_sqrt_nu = self.sigma_vec * np.sqrt(self.nu_vec)

        self.a, self.C, self.dep_calibration_info = estimate_lms_vg_brownian_correlation(target_return_corr=self.log_returns_corr_matrix, theta_vec=self.theta_vec, sigma_vec=self.sigma_vec, nu_vec=self.nu_vec, return_info=True)
        
        if self.dep_calibration_info["cholesky_lower"] is None:
            raise RuntimeError(
                "Calibrated C is numerically not positive definite. "
                "Increase pd_epsilon, e.g. pd_epsilon=1e-8, or use a more accurate solver."
            )
        
        # self.L_transp = self.dep_calibration_info['cholesky_lower'].T
        self.L_transp = np.linalg.cholesky(self.C).T

        self.shape_Y = self.dt*(1/self.nu_vec - self.a)
        self.shape_Z = self.dt*self.a

        self.omega = (1/self.nu_vec)*np.log(1 - self.theta_vec*self.nu_vec - 0.5*self.sigma_vec**2*self.nu_vec)

        self.log_drift = (self.mu_vec+self.omega)*self.dt
        self.log_S0 = np.log(self.S0_vec)

        if return_torch is not None:
            return_numpy = not bool(return_torch)
        self.return_numpy = bool(return_numpy)

        self.seed = None
        self.rng = np.random.default_rng()
        self.device = self._resolve_device(device)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if self.device.type == "mps" and dtype != torch.float32:
            dtype = torch.float32
        self.dtype = dtype

        self._gamma_device = self._resolve_gamma_device()
        self._gamma_generator = self._new_entropy_generator(self._gamma_device)
        self._normal_generator = self._new_entropy_generator(self.device)
        self._fallback_random_seed_each_call = self._normal_generator is None
        self._build_torch_constants()

        # ``seed`` is intentionally ignored.  It is accepted only for backward
        # compatibility; PPO no longer passes stock-price simulation seeds.
        _ = seed

    @classmethod
    def _resolve_device(cls, device=None):
        if device is not None:
            dev = torch.device(device)
            if dev.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                return torch.device("cpu")
            if dev.type == "cuda" and not torch.cuda.is_available():
                return torch.device("cpu")
            return dev
        if cls.DEFAULT_DEVICE == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if cls.DEFAULT_DEVICE == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device("cpu")

    def _resolve_gamma_device(self):
        # torch Gamma sampling is not supported on MPS.  For CUDA, keep Gamma on
        # CUDA to avoid unnecessary CPU-to-GPU transfers.
        if self.device.type == "mps":
            return torch.device("cpu")
        return self.device

    @staticmethod
    def _new_entropy_generator(device):
        try:
            generator = torch.Generator(device=device)
            generator.seed()
            return generator
        except Exception:
            return None

    @staticmethod
    def _seed_global_rng_from_entropy():
        seed = int.from_bytes(os.urandom(8), byteorder="little", signed=False) % (2**63 - 1)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

    def _build_torch_constants(self):
        self._S0_t = torch.as_tensor(self.S0_vec, device=self.device, dtype=self.dtype)
        self._mu_t = torch.as_tensor(self.mu_vec, device=self.device, dtype=self.dtype)
        self._sigma_t = torch.as_tensor(self.sigma_vec, device=self.device, dtype=self.dtype)
        self._theta_t = torch.as_tensor(self.theta_vec, device=self.device, dtype=self.dtype)
        self._nu_t = torch.as_tensor(self.nu_vec, device=self.device, dtype=self.dtype)
        self._theta_nu_t = torch.as_tensor(self.theta_nu, device=self.device, dtype=self.dtype)
        self._sigma_sqrt_nu_t = torch.as_tensor(self.sigma_sqrt_nu, device=self.device, dtype=self.dtype)
        self._log_drift_t = torch.as_tensor(self.log_drift, device=self.device, dtype=self.dtype)
        self._log_S0_t = torch.as_tensor(self.log_S0, device=self.device, dtype=self.dtype)
        self._L_transp_t = torch.as_tensor(self.L_transp, device=self.device, dtype=self.dtype)

        self._shape_Y_gamma = torch.as_tensor(self.shape_Y, device=self._gamma_device, dtype=self.dtype)
        self._shape_Z_gamma = torch.as_tensor(float(self.shape_Z), device=self._gamma_device, dtype=self.dtype)
        self._nu_gamma = torch.as_tensor(self.nu_vec, device=self._gamma_device, dtype=self.dtype)

    def to(self, device=None, dtype=None):
        self.device = self._resolve_device(device)
        if dtype is not None:
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)
            if self.device.type == "mps" and dtype != torch.float32:
                dtype = torch.float32
            self.dtype = dtype
        elif self.device.type == "mps" and self.dtype != torch.float32:
            self.dtype = torch.float32
        self._gamma_device = self._resolve_gamma_device()
        self._gamma_generator = self._new_entropy_generator(self._gamma_device)
        self._normal_generator = self._new_entropy_generator(self.device)
        self._fallback_random_seed_each_call = self._normal_generator is None
        self._build_torch_constants()
        return self

    def set_seed(self, seed):
        # Kept for compatibility with older notebooks.  The PPO adapter does not
        # call this, because price paths should not be tied to deterministic seeds.
        self.seed = int(seed)
        self.rng = np.random.default_rng()
        self._gamma_generator = self._new_entropy_generator(self._gamma_device)
        self._normal_generator = self._new_entropy_generator(self.device)
        self._fallback_random_seed_each_call = self._normal_generator is None
        return self

    def _standard_gamma(self, concentration):
        if self._gamma_generator is not None:
            try:
                return torch._standard_gamma(concentration, generator=self._gamma_generator)
            except TypeError:
                return torch._standard_gamma(concentration)
            except RuntimeError:
                return torch._standard_gamma(concentration)
        return torch._standard_gamma(concentration)

    def _randn(self, shape):
        if self._normal_generator is not None:
            try:
                return torch.randn(shape, device=self.device, dtype=self.dtype, generator=self._normal_generator)
            except TypeError:
                self._normal_generator = None
            except RuntimeError:
                self._normal_generator = None
        # Last-resort fallback keeps random-normal generation on the selected
        # simulation device, but reseeds from OS entropy so it is not tied to PPO's seed.
        self._seed_global_rng_from_entropy()
        return torch.randn(shape, device=self.device, dtype=self.dtype)

    def build_corr_matrix_BM(self):
        d_var = self.sigma_vec**2 + self.theta_vec**2 * self.nu_vec
        theta_nu = self.theta_vec * self.nu_vec
        sigma_sqrt_nu = self.sigma_vec * np.sqrt(self.nu_vec)

        C = (self.log_returns_corr_matrix*np.sqrt(np.outer(d_var,d_var))/self.a - np.outer(theta_nu,theta_nu))/np.outer(sigma_sqrt_nu,sigma_sqrt_nu)
        C = 0.5*(C+C.T)
        np.fill_diagonal(C, 1.0)
        return C

    @torch.no_grad()
    def _simulate_raw_stock_paths(self, raw_M):
        raw_M = int(raw_M)
        N, d = int(self.N_t), int(self.d)
        if raw_M < 1:
            raise ValueError("raw_M must be positive.")

        dZ_shape = self._shape_Z_gamma.expand(raw_M, N, 1).clone()
        dZ = self._standard_gamma(dZ_shape).to(device=self.device, dtype=self.dtype, non_blocking=True)

        dY_shape = self._shape_Y_gamma.view(1, 1, d).expand(raw_M, N, d).clone()
        dY = self._standard_gamma(dY_shape)
        dY = (dY * self._nu_gamma.view(1, 1, d)).to(device=self.device, dtype=self.dtype, non_blocking=True)

        eps_Y = self._randn((raw_M, N, d))
        temp_Z = self._randn((raw_M, N, d))
        eps_Z = torch.matmul(temp_Z, self._L_transp_t)

        sqrt_dY = torch.sqrt(torch.clamp_min(dY, 0.0))
        sqrt_dZ = torch.sqrt(torch.clamp_min(dZ, 0.0))

        dX = (
            self._log_drift_t.view(1, 1, d)
            + self._theta_t.view(1, 1, d) * dY
            + self._sigma_t.view(1, 1, d) * sqrt_dY * eps_Y
            + self._theta_nu_t.view(1, 1, d) * dZ
            + self._sigma_sqrt_nu_t.view(1, 1, d) * sqrt_dZ * eps_Z
        )

        log_S0 = self._log_S0_t.view(1, 1, d)
        X_tail = log_S0 + torch.cumsum(dX, dim=1)
        X = torch.cat([log_S0.expand(raw_M, 1, d), X_tail], dim=1)
        return torch.exp(X)

    @torch.no_grad()
    def simulate_stock_price_process(self, N_MC=None, mean_samples=None, return_numpy=None, return_torch=None):
        """Simulate stock paths with shape ``(N_MC, N_t + 1, d)``.

        For each output path, ``mean_samples`` independent stock-price paths are
        generated and averaged pointwise.  This preserves the rho-alpha VG path
        formula while reducing the flat/smooth intervals caused by very small
        Gamma increments.
        """
        M = int(self.N_MC if N_MC is None else N_MC)
        K = int(self.path_mean_samples if mean_samples is None else mean_samples)
        K = max(1, K)
        if M < 1:
            raise ValueError("N_MC must be positive.")

        if return_torch is not None:
            return_numpy = not bool(return_torch)
        elif return_numpy is None:
            return_numpy = self.return_numpy

        effective_chunk = max(K, int(self.max_simulated_paths_per_chunk))
        envs_per_chunk = max(1, effective_chunk // K)

        chunks = []
        remaining = M
        while remaining > 0:
            m_chunk = min(remaining, envs_per_chunk)
            raw = self._simulate_raw_stock_paths(m_chunk * K)
            if K > 1:
                raw = raw.view(m_chunk, K, self.N_t + 1, self.d).mean(dim=1)
            chunks.append(raw)
            remaining -= m_chunk

        S = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
        if return_numpy:
            return S.detach().cpu().numpy()
        return S
