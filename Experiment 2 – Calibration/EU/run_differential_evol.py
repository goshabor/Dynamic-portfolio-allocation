from scipy.optimize import differential_evolution, minimize

def progress(xk, convergence):
    print(f"[DE] conv={convergence:.3f} best_x≈{xk}")
    return False

def local_progress(xk, state=None):
    print(f"[LOCAL] x≈{xk}")

def run_differential_evolution(class_, args, bounds, strategy, x0=None, constraint=None):
    kwargs = dict(
        func=class_.objective_func,
        args=() if args is None else args,
        strategy=strategy,
        bounds=bounds,
        tol=1e-4,
        atol=1e-4,
        popsize=100,
        maxiter=10000,
        mutation=(0.5, 0.9),
        recombination=0.9,
        updating='deferred',
        workers=-1,
        polish=False,        
        disp=True,
        constraints=constraint,
        x0=x0,
        callback=progress,
    )

    result_de = differential_evolution(**kwargs)

    print("\n=== Finished DE stage ===")
    print(f"DE best fun = {result_de.fun}")
    print(f"DE best x = {result_de.x}")
    return result_de
