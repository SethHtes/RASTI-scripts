"""
Simulation of Magnitude-Squared Coherence Estimator Bias
Demonstrating Nuttall & Carter (1976) bias formula

This script generates Monte Carlo simulations to verify the theoretical bias
in coherence estimates as a function of the number of averaged segments.

Author: Seth R. Larner
"""

import multiprocessing
from typing import Dict, Tuple, List, Optional, Union

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.figure
from scipy.special import hyp2f1
import mpmath

# Set plotting style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# =============================================================================
# Parameters
# =============================================================================
N_TRIALS = 50000       # Number of Monte Carlo trials
SEGMENT_COUNTS = [4, 8, 16, 32]  # Values of N to test
TRUE_COHERENCE_VALUES = np.arange(0, 0.999 + 0.001, 0.001)  # True γ² values

# Multiprocessing options
USE_MULTIPROCESSING = True  # Enable parallel processing
N_CORES = None  # Number of CPU cores to use (None = use all available)

# Colors for plotting
COLORS = ['#0000FF', '#FFA500', '#008000', '#FF0000', '#800080', '#000000']


# =============================================================================
# Theoretical Bias Formulas
# =============================================================================
def _robust_hyp2f1(a: float, b: float, c: float, z: float) -> float:
    """
    Robust evaluation of ₂F₁ using scipy with mpmath fallback.

    Scipy's hyp2f1 can return NaN for large c values near z=1 due to
    numerical instabilities. Mpmath uses arbitrary precision arithmetic
    and handles these cases correctly.
    """
    result = hyp2f1(a, b, c, z)
    if np.isnan(result) or np.isinf(result):
        # Fall back to mpmath for problematic cases
        result = float(mpmath.hyp2f1(a, b, c, z))
    return result


def exact_expected_msc(gamma2: Union[float, NDArray[np.floating]], n: int) -> Union[float, NDArray[np.floating]]:
    """
    Exact expected value of the MSC estimator from Nuttall & Carter (1976) Eq.1

    E[γ̂²] = 1/n + (n-1)/(n+1) * γ² * ₂F₁(1, 1; n+2; γ²)

    Parameters
    ----------
    gamma2 : float or array
        True coherence squared
    n : int
        Number of segments averaged

    Returns
    -------
    float or array
        Expected value of the MSC estimator

    Notes
    -----
    Uses mpmath fallback for numerical stability when scipy's hyp2f1 fails
    (which can occur for large n and γ² close to 1).
    """
    # Handle the limiting case γ² → 1 analytically
    if np.isscalar(gamma2):
        if gamma2 >= 1.0:
            return 1.0
        hyp_val = _robust_hyp2f1(1, 1, n+2, gamma2)
        return 1/n + (n-1)/(n+1) * gamma2 * hyp_val
    else:
        # Array input
        result = np.zeros_like(gamma2, dtype=float)
        for i, g2 in enumerate(gamma2):
            if g2 >= 1.0:
                result[i] = 1.0
            else:
                hyp_val = _robust_hyp2f1(1, 1, n+2, g2)
                result[i] = 1/n + (n-1)/(n+1) * g2 * hyp_val
        return result


def exact_bias(gamma2: Union[float, NDArray[np.floating]], n: int) -> Union[float, NDArray[np.floating]]:
    """
    Exact bias: E[γ̂²] - γ²
    """
    return exact_expected_msc(gamma2, n) - gamma2


def approx_bias(gamma2: Union[float, NDArray[np.floating]], n: int) -> Union[float, NDArray[np.floating]]:
    """
    Approximate bias from Nuttall & Carter (1976) Eq.2

    bias ≈ (1-γ²)²/n * (1 + 2γ²/n)
    """
    return (1 - gamma2)**2 / n * (1 + 2*gamma2/n)


def leading_order_bias(gamma2: Union[float, NDArray[np.floating]], n: int) -> Union[float, NDArray[np.floating]]:
    """
    Leading order approximation: bias ≈ (1-γ²)²/n
    """
    return (1 - gamma2)**2 / n


# =============================================================================
# Single Trial Coherence Estimation
# =============================================================================
def single_trial_msc(true_gamma2: float, n_segments: int) -> float:
    """
    Generate one estimate of MSC from N segments at a single frequency.

    Using the synthetic LC method:
    Y = T*X + sqrt(1-|T|²) * noise

    For equal PSDs (PX = PY = 2), |T|² = γ²

    Parameters
    ----------
    true_gamma2 : float
        True coherence squared (0 <= γ² <= 1)
    n_segments : int
        Number of segments to average

    Returns
    -------
    float
        Estimated MSC value
    """
    # Transfer function magnitude for equal PSDs: |T|² = γ²
    T = np.sqrt(true_gamma2)
    
    # Generate random Fourier coefficients for N segments
    # Reference time series: X = (Ar + i*Br)
    Xre = np.random.randn(n_segments)
    Xim = np.random.randn(n_segments)
    
    # Independent noise for incoherent part: (Hr + i*Jr)
    Hre = np.random.randn(n_segments)
    Him = np.random.randn(n_segments)
    
    # Dependent time series: Y = T*X + sqrt(1-T²)*(H + i*J)
    # For simplicity, assume zero phase lag (T is real)
    sqrt_factor = np.sqrt(1 - T**2) if T < 1 else 0
    Yre = T * Xre + sqrt_factor * Hre
    Yim = T * Xim + sqrt_factor * Him
    
    # Compute cross-spectrum: C = <X* Y> = <(Xre - i*Xim)(Yre + i*Yim)>
    cross_spec_real = np.mean(Xre * Yre + Xim * Yim)
    cross_spec_imag = np.mean(Xre * Yim - Xim * Yre)
    
    # Compute power spectra
    PX = np.mean(Xre**2 + Xim**2)
    PY = np.mean(Yre**2 + Yim**2)
    
    # MSC estimate: |C|² / (PX * PY)
    msc_estimate = (cross_spec_real**2 + cross_spec_imag**2) / (PX * PY)
    
    return msc_estimate


# =============================================================================
# Monte Carlo Simulation
# =============================================================================
def _worker_single_trial(args: Tuple[float, int]) -> float:
    """
    Worker function for parallel execution of single trial.

    Parameters
    ----------
    args : tuple
        (true_gamma2, n_segments) for single_trial_msc

    Returns
    -------
    float
        MSC estimate for this trial
    """
    gamma2, n = args
    return single_trial_msc(gamma2, n)


def run_simulation(
    n_trials: int = N_TRIALS,
    segment_counts: List[int] = SEGMENT_COUNTS,
    coherence_values: NDArray[np.floating] = TRUE_COHERENCE_VALUES,
    verbose: bool = True,
    use_multiprocessing: bool = USE_MULTIPROCESSING,
    n_cores: Optional[int] = N_CORES
) -> Dict[Tuple[int, float], Tuple[float, float]]:
    """
    Run Monte Carlo simulation to estimate bias and variance.

    Parameters
    ----------
    n_trials : int
        Number of Monte Carlo trials per (n, gamma2) combination
    segment_counts : list of int
        Values of N to test
    coherence_values : array
        True coherence squared values to test
    verbose : bool
        Print progress messages
    use_multiprocessing : bool
        Enable parallel processing using multiprocessing.Pool
    n_cores : int or None
        Number of CPU cores to use (None = use all available)

    Returns
    -------
    dict
        Dictionary with keys (n, gamma2) and values (mean_estimate, std_estimate)
    """
    results = {}

    if verbose:
        print(f"Running Monte Carlo simulation with {n_trials} trials...")
        if use_multiprocessing:
            cores = n_cores if n_cores is not None else multiprocessing.cpu_count()
            print(f"Using multiprocessing with {cores} cores.")
        else:
            print("Running single-threaded.")
        print("This may take a few minutes.")

    for n in segment_counts:
        if verbose:
            print(f"  Processing N={n}...", end=" ", flush=True)

        for gamma2 in coherence_values[::10]:
            if use_multiprocessing:
                # Parallel execution
                with multiprocessing.Pool(processes=n_cores) as pool:
                    # Create argument list for all trials
                    args_list = [(gamma2, n) for _ in range(n_trials)]
                    estimates = np.array(pool.map(_worker_single_trial, args_list))
            else:
                # Sequential execution
                estimates = np.array([single_trial_msc(gamma2, n) for _ in range(n_trials)])

            # Compute statistics
            mean_est = np.mean(estimates)
            std_est = np.std(estimates, ddof=1)

            results[(n, gamma2)] = (mean_est, std_est)
        if verbose:
            print("done")

    if verbose:
        print("Simulation complete.")

    return results


# =============================================================================
# Plotting Functions
# =============================================================================
def plot_bias_log(
    results: Dict[Tuple[int, float], Tuple[float, float]],
    segment_counts: List[int] = SEGMENT_COUNTS,
    coherence_values: NDArray[np.floating] = TRUE_COHERENCE_VALUES,
    colors: List[str] = COLORS
) -> matplotlib.figure.Figure:
    """
    Plot 1: Log of |Bias| vs true coherence for different N
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Theory curves range
    gamma2_theory = np.linspace(0.001, 0.99, 200)
    
    for i, n in enumerate(segment_counts):
        color = colors[i % len(colors)]
        
        # Simulation data
        gamma2_vals = []
        bias_vals = []
        for gamma2 in coherence_values:
            if (n, gamma2) in results:
                mean_est, _ = results[(n, gamma2)]
                bias = abs(mean_est - gamma2)
                if bias > 0:  # Only plot positive bias for log scale
                    gamma2_vals.append(gamma2)
                    bias_vals.append(bias)
        
        # Plot simulation points
        ax.semilogy(gamma2_vals, bias_vals, 'o', color=color, markersize=6,
                   label=f'N = {n}', alpha=0.7)

        # Plot theory curve
        theory_bias = exact_bias(gamma2_theory, n)
        ax.semilogy(gamma2_theory, theory_bias, '--', color=color, linewidth=2.5,
                   label=f'')
    
    ax.set_xlabel(r'$\gamma^2$', fontsize=18)
    ax.set_ylabel(r'$E[\hat{\gamma}^2] - \gamma^2$', fontsize=18)
    ax.set_xlim(0, 1)
    ax.set_ylim(1e-5, 1)
    ax.legend(loc='upper right', ncol=2, fontsize=14)
    ax.grid(True, alpha=0.3)
    # ax.set_title('Coherence Estimator Bias: Simulation vs Theory')
    
    plt.tight_layout()
    return fig


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    # Run simulation
    results = run_simulation()
    
    # Generate all plots
    print("\nGenerating plots...")
    
    # Plot 1: Bias log plot
    fig1 = plot_bias_log(results)
    fig1.savefig('bias_log_plot.png', dpi=300, bbox_inches='tight')
    print("Saved: bias_log_plot.png")
