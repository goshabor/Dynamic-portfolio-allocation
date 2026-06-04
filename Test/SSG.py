import numpy as np
import cvxpy as cp


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
    a_upper_margin=1e-4,
    a_lower_fraction=1e-4,
    pd_epsilon=1e-4,
    maximize_a_after_rmse=True,
    loss_tie_abs=1e-6,
    loss_tie_rel=1e-6,
    solver=None,
    solver_opts=None,
):
    """
    Calibrate the Brownian correlation matrix C and common subordinator parameter a
    for the Luciano-Marena-Semeraro rho-alpha VG model.

    This solves

        min_{a, C} sum_{i<j} (rho_emp_ij - rho_model_ij(a, C))^2

    subject to

        0 < a < min_j 1 / nu_j,
        C is a valid correlation matrix.

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
    def __init__(self, S0_vec, mu_vec, sigma_vec, theta_vec, nu_vec, log_returns_corr_matrix, T, N_t, N_MC):
        self.S0_vec = S0_vec
        self.mu_vec = mu_vec
        self.sigma_vec = sigma_vec
        self.theta_vec = theta_vec
        self.nu_vec = nu_vec
        self.log_returns_corr_matrix = log_returns_corr_matrix

        self.T = T
        self.N_t = N_t
        self.dt = T/N_t
        self.N_MC = N_MC
        self.d = self.S0_vec.size

        self.a = 0.4*np.min(1/self.nu_vec)

        self.theta_nu = self.theta_vec * self.nu_vec
        self.sigma_sqrt_nu = self.sigma_vec * np.sqrt(self.nu_vec)

        self.a, self.C, self.dep_calibration_info = estimate_lms_vg_brownian_correlation(target_return_corr=self.log_returns_corr_matrix, theta_vec=self.theta_vec, sigma_vec=self.sigma_vec, nu_vec=self.nu_vec)
        
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

        self.rng = np.random.default_rng()
    
    def simulate_stock_price_process(self):
        M, N, d = self.N_MC, self.N_t, self.d
        X = np.empty((M, N + 1, d))
        X[:,0,:] = self.log_S0

        dZ = self.rng.gamma(self.shape_Z, 1.0, size=(M, N))
        dZ = dZ[:,:,None]
        dY = self.rng.gamma(self.shape_Y, self.nu_vec, size=(M,N,d))

        eps_Y = self.rng.standard_normal((M,N,d))
        temp_Z = self.rng.standard_normal((M*N,d))
        eps_Z = (temp_Z@self.L_transp).reshape(M,N,d)

        dX = self.log_drift + self.theta_vec*dY + self.sigma_vec*np.sqrt(dY)*eps_Y + self.theta_nu*dZ + self.sigma_sqrt_nu*np.sqrt(dZ)*eps_Z

        X[:,1:,:] = self.log_S0 + np.cumsum(dX, axis=1)
        S = np.exp(X)
        return S