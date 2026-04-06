import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

def moment_derivatives(t, y, n, i, gamma):
    """
    Computes the instantaneous rates of change for the moment system.
    """
    # AXEL TODO: Generalize unpacking for arbitrary K
    logZ, m1, h2 = y
    
    # SAFETY: Clip internal variables purely for the derivative calculation
    m1_safe = np.clip(m1, 0.0, 1.0)
    max_h2 = m1_safe * (1.0 - m1_safe)
    h2_safe = np.clip(h2, 0.0, max_h2)
    
    # DYNAMIC CLOSURE (AXEL TODO: Generalize for K)
    # NOTE: When K is large, we may need to worry about numerical
    # precision issues. Extending K far enough to be sure we can just 
    # truncate is always an option
    denom = m1_safe * (1.0 - m1_safe) - h2_safe
    if denom <= 1e-12:
        C_dynamic = np.inf
        h3 = h2_safe * (1.0 - 2.0 * m1_safe)
    else:
        C_dynamic = h2_safe / denom
        h3 = h2_safe * (1.0 - 2.0 * m1_safe) * (C_dynamic / (C_dynamic + 2.0))
        
    # DIFFERENTIAL EQUATIONS (AXEL TODO: Loop over K moments)
    dlogZ_dt = - (n * (n - 1) / 2) + gamma * (i - n * m1_safe)
    dm1_dt = (i - n * m1_safe) + gamma * h2_safe - gamma * n * (m1_safe * (1.0 - m1_safe) - h2_safe)
    dh2_dt = (
        i
        - m1_safe * (2 * i - n)
        - h2_safe * (1 + 2 * n + gamma * n * (0.5 - m1_safe))
        + h3 * (gamma + gamma * n / 2)
    )
    
    return [dlogZ_dt, dm1_dt, dh2_dt]


def compute_ode_equilibrium(n, i, gamma, m1_init=None, h2_init=None):
    """
    Find the stationary point of the moment ODE system numerically,
    i.e. the (m1, h2) such that dm1/dt = dh2/dt = 0.

    Parameters:
    -----------
    n : int
        Total lineages.
    i : int
        Derived lineages.
    gamma : float
        Scaled selection coefficient (2 * Ne * s).
    m1_init : float, optional
        Initial guess for m1. Defaults to i/n.
    h2_init : float, optional
        Initial guess for h2. Defaults to half the Bernoulli variance at m1_init.

    Returns:
    --------
    m1_eq : float
        Equilibrium mean allele frequency.
    h2_eq : float
        Equilibrium genetic variance E[x(1-x)].
    """
    if m1_init is None:
        m1_init = i / n
    if h2_init is None:
        h2_init = 0.5 * m1_init * (1.0 - m1_init)

    def residuals(x):
        m1, h2 = x
        _, dm1, dh2 = moment_derivatives(0.0, [0.0, m1, h2], n, i, gamma)
        return [dm1, dh2]

    sol = root(residuals, [m1_init, h2_init])
    if not sol.success:
        raise RuntimeError(f"ODE equilibrium solver failed: {sol.message}")
    return sol.x[0], sol.x[1]


def integrate_interval(n, i, gamma, Ne, t_span, y0=None, rtol=1e-8, atol=1e-10):
    """
    Integrates the moment dynamics for a single continuous tree interval.
    
    Parameters:
    -----------
    n : int
        Current total lineages.
    i : int
        Current derived lineages.
    gamma : float
        Scaled selection coefficient (2 * Ne * s).
    Ne : int
        Effective population size (used to determine dense output resolution).
    t_span : tuple
        (t_start, t_end) in scaled time.
    y0 : array_like, optional
        Initial state vector [logZ, m1, h2, ...]. 
        If None, initializes from the theoretical Beta prior for a new mutation.
        
    Returns:
    --------
    sol : scipy.integrate.OdeResult
        An object containing:
        - sol.t : array of time points
        - sol.y : 2D array of state variables (shape: len(y0) x len(t))
        - sol.success : boolean flag
    """
    t_start, t_end = t_span
    
    if y0 is None:
        # Initial condition at the mutation origin: point mass at x ≈ 0.
        # The fictitious mutations (θ₁=2i, θ₂=2(n-i)) drive the density
        # away from the boundary, so starting at (0, 0, 0) is stable.
        # STUDENT TODO: Expand to generate initial conditions up to K moments.
        logZ_init = 0.0
        m1_init = 0.0
        h2_init = 0.0
        y0 = [logZ_init, m1_init, h2_init]
    
    # Force the solver to store a dense grid of points for smooth plotting
    num_eval_points = max(2, int(abs(t_end - t_start) * (2 * Ne))) 
    t_eval = np.linspace(t_start, t_end, num_eval_points)
    
    # Run the integrator
    sol = solve_ivp(
        fun=moment_derivatives,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        args=(n, i, gamma),
        method='RK45',
        rtol=rtol,
        atol=atol
    )
    
    return sol