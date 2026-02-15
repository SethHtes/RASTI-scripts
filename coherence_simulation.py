"""
Implementation of correlated time series simulation method for X-ray astronomy.

Based on the algorithm from Larner, Nowak, & Wilms (2026):
- Reference time series generated via Timmer & Konig (1995) method
- Dependent time series constructed as coherent + incoherent components
- Transfer function relates coherence and phase lag to mixing

This script generates corner plots showing the joint distributions of:
1. Coherence and phase lag estimates
2. Power spectra and cross-spectrum components
3. All six quantities together
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import corner
from scipy.special import hyp2f1
import mpmath

# =============================================================================
# Simulation and Plotting Parameters (User-Tunable)
# =============================================================================

# Target coherence (0 to 1)
TARGET_GAMMA2 = 0.75

# Target phase lag in radians
TARGET_PHI = np.pi / 4  # 45 degrees

# Number of segments to average for each realization
N_SEGMENTS = 50

# Number of Monte Carlo realizations for the corner plots
N_REALIZATIONS = 50000

# Power spectral shapes (at the frequency of interest)
# Using equal power spectra for simplicity
P_X = 1.0
P_Y = 10.0

# Random seed for reproducibility (set to None for random results)
RANDOM_SEED = None

# Show expected (biased) coherence in addition to true coherence
SHOW_BIAS_LINE = False

# Show theoretical variance contours (1sigma and 2sigma ellipses)
SHOW_THEORY_CONTOURS = True

# Directory for saving plots
SAVE_DIR = '.'

# list of sigma levels and alpha values to plot theoretical contours
CONTOUR_LEVELS_ALPHAS = [(1, 1.), (2, 1.)]

# Plotting resolution
DPI = 300

# Contour quantiles
QUANTILES = [0.16, 0.5, 0.84]


# =============================================================================
# Core Simulation Functions
# =============================================================================

def _robust_hyp2f1(a, b, c, z):
    """
    Robust evaluation of 2F1 using scipy with mpmath fallback.
    
    Scipy's hyp2f1 can return NaN for large c values near z=1 due to
    numerical instabilities. Mpmath uses arbitrary precision arithmetic
    and handles these cases correctly.
    """
    result = hyp2f1(a, b, c, z)
    if np.isnan(result) or np.isinf(result):
        result = float(mpmath.hyp2f1(a, b, c, z))
    return result


def expected_biased_coherence(gamma2, n):
    """
    Expected value of the MSC estimator from Nuttall & Carter (1976) Eq.1
    
    Parameters
    ----------
    gamma2 : float
        True coherence squared
    n : int
        Number of segments averaged
    
    Returns
    -------
    float
        Expected value of the MSC estimator (includes bias)
    """
    if gamma2 >= 1.0:
        return 1.0
    hyp_val = _robust_hyp2f1(1, 1, n + 2, gamma2)
    return 1/n + (n - 1)/(n + 1) * gamma2 * hyp_val


def theoretical_covariance_matrix(P_X, P_Y, gamma2, phi, n):
    """
    Compute the theoretical covariance matrix for (C_r, C_i, P_X, P_Y).
    
    From Equation 23 of the draft, divided by n for the estimator variance.
    
    Parameters
    ----------
    P_X, P_Y : float
        True power spectra
    gamma2 : float
        True coherence
    phi : float
        True phase lag
    n : int
        Number of segments averaged
    
    Returns
    -------
    ndarray
        4x4 covariance matrix for (C_r, C_i, P_X, P_Y)
    """
    gamma = np.sqrt(gamma2)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_2phi = np.cos(2 * phi)
    
    # From Equation 23, this is Î£ (single-segment covariance)
    # Divide by n for the averaged estimator
    Sigma = np.array([
        [0.5 * P_X * P_Y * (1 + gamma2 * cos_2phi),
         P_X * P_Y * gamma2 * cos_phi * sin_phi,
         np.sqrt(P_X**3 * P_Y) * gamma * cos_phi,
         np.sqrt(P_X * P_Y**3) * gamma * cos_phi],
        
        [P_X * P_Y * gamma2 * cos_phi * sin_phi,
         0.5 * P_X * P_Y * (1 - gamma2 * cos_2phi),
         np.sqrt(P_X**3 * P_Y) * gamma * sin_phi,
         np.sqrt(P_X * P_Y**3) * gamma * sin_phi],
        
        [np.sqrt(P_X**3 * P_Y) * gamma * cos_phi,
         np.sqrt(P_X**3 * P_Y) * gamma * sin_phi,
         P_X**2,
         P_X * P_Y * gamma2],
        
        [np.sqrt(P_X * P_Y**3) * gamma * cos_phi,
         np.sqrt(P_X * P_Y**3) * gamma * sin_phi,
         P_X * P_Y * gamma2,
         P_Y**2]
    ])
    
    return Sigma / n


def theoretical_variance_gamma2_phi(gamma2, n):
    """
    Compute theoretical variances for coherence and phase lag.
    
    From Equation 25 of the draft.
    
    Parameters
    ----------
    gamma2 : float
        True coherence
    n : int
        Number of segments averaged
    
    Returns
    -------
    tuple
        (Var(γ), Var(φ))
    """
    var_gamma2 = 2 * gamma2 * (gamma2 - 1)**2 / n
    var_phi = (1 - gamma2) / (2 * n * gamma2)
    return var_gamma2, var_phi


def theoretical_full_covariance_6x6(P_X, P_Y, gamma2, phi, n):
    """
    Compute the full 6x6 theoretical covariance matrix for 
    (P_X, P_Y, C_r, C_i, γ², φ) using the delta method.
    
    Parameters
    ----------
    P_X, P_Y : float
        True power spectra
    gamma2 : float
        True coherence
    phi : float
        True phase lag
    n : int
        Number of segments averaged
    
    Returns
    -------
    ndarray
        6x6 covariance matrix in order (P_X, P_Y, C_r, C_i, γ², φ)
    dict
        Dictionary with labeled covariances for easy access
    """
    gamma = np.sqrt(gamma2)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    # Start with the 4x4 covariance from Equation 23 (order: C_r, C_i, P_X, P_Y)
    Sigma_4 = theoretical_covariance_matrix(P_X, P_Y, gamma2, phi, n)
    
    # Reorder to (P_X, P_Y, C_r, C_i)
    reorder = [2, 3, 0, 1]
    Sigma_reordered = np.zeros((4, 4))
    for i, ri in enumerate(reorder):
        for j, rj in enumerate(reorder):
            Sigma_reordered[i, j] = Sigma_4[ri, rj]
    
    # Compute gradients at true values (B = (C_r, C_i, P_X, P_Y))
    # But we need them in order (P_X, P_Y, C_r, C_i) for our matrix
    C_r_true = np.sqrt(P_X * P_Y) * gamma * cos_phi
    C_i_true = np.sqrt(P_X * P_Y) * gamma * sin_phi
    
    # Gradient of γ² with respect to (C_r, C_i, P_X, P_Y)
    grad_gamma2_old_order = np.array([
        2 * C_r_true / (P_X * P_Y),
        2 * C_i_true / (P_X * P_Y),
        -(C_r_true**2 + C_i_true**2) / (P_X**2 * P_Y),
        -(C_r_true**2 + C_i_true**2) / (P_X * P_Y**2)
    ])
    
    # Reorder to (P_X, P_Y, C_r, C_i)
    grad_gamma2 = grad_gamma2_old_order[[2, 3, 0, 1]]
    
    # Gradient of φ with respect to (C_r, C_i, P_X, P_Y)
    # φ = arctan(C_i / C_r), so ∂φ/∂C_r = -C_i/(C_r² + C_i²), ∂φ/∂C_i = C_r/(C_r² + C_i²)
    # φ doesn't depend on P_X or P_Y directly
    grad_phi_old_order = np.array([
        -C_i_true / (C_r_true**2 + C_i_true**2),
        C_r_true / (C_r_true**2 + C_i_true**2),
        0.0,
        0.0
    ])
    
    # Reorder to (P_X, P_Y, C_r, C_i)
    grad_phi = grad_phi_old_order[[2, 3, 0, 1]]
    
    # Compute variances for γ² and φ
    var_gamma2 = grad_gamma2 @ Sigma_reordered @ grad_gamma2
    var_phi = grad_phi @ Sigma_reordered @ grad_phi
    
    # Compute covariance between γ² and φ
    cov_gamma2_phi = grad_gamma2 @ Sigma_reordered @ grad_phi
    
    # Compute covariances between (γ², φ) and (P_X, P_Y, C_r, C_i)
    cov_gamma2_with_base = Sigma_reordered @ grad_gamma2  # 4x1 vector
    cov_phi_with_base = Sigma_reordered @ grad_phi  # 4x1 vector
    
    # Build full 6x6 matrix
    Sigma_6 = np.zeros((6, 6))
    
    # Top-left 4x4: base quantities
    Sigma_6[:4, :4] = Sigma_reordered
    
    # Row/column 5: γ² covariances
    Sigma_6[4, :4] = cov_gamma2_with_base
    Sigma_6[:4, 4] = cov_gamma2_with_base
    Sigma_6[4, 4] = var_gamma2
    
    # Row/column 6: φ covariances
    Sigma_6[5, :4] = cov_phi_with_base
    Sigma_6[:4, 5] = cov_phi_with_base
    Sigma_6[5, 5] = var_phi
    
    # γ² and φ covariance
    Sigma_6[4, 5] = cov_gamma2_phi
    Sigma_6[5, 4] = cov_gamma2_phi
    
    # Create labeled dictionary
    labels = ['P_X', 'P_Y', 'C_r', 'C_i', 'gamma2', 'phi']
    covariances = {}
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            if i <= j:  # Only store upper triangle (symmetric matrix)
                covariances[f'{label_i}_{label_j}'] = Sigma_6[i, j]
    
    return Sigma_6, covariances


def draw_covariance_ellipse(ax, mean, cov_2x2, n_std=1.0, **kwargs):
    """
    Draw an ellipse representing a 2D Gaussian covariance.
    
    Parameters
    ----------
    ax : matplotlib axis
        Axis to draw on
    mean : array-like
        Center of ellipse (x, y)
    cov_2x2 : ndarray
        2x2 covariance matrix
    n_std : float
        Number of standard deviations for the ellipse radius
    **kwargs : dict
        Passed to matplotlib Ellipse patch
    """
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_2x2)
    
    # Sort by eigenvalue (largest first)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Compute angle and dimensions
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])
    
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,zorder=10000, **kwargs)
    ax.add_patch(ellipse)
    return ellipse


def compute_transfer_function(gamma2, phi, P_X, P_Y):
    """
    Compute the transfer function T from coherence and phase lag.
    
    From Equation 15 of the draft:
    T = sqrt(P_Y * gamma^2 / P_X) * exp(i * phi)
    
    Parameters
    ----------
    gamma2 : float
        Target coherence (squared)
    phi : float
        Target phase lag in radians
    P_X : float
        Power spectrum of reference time series
    P_Y : float
        Power spectrum of dependent time series
        
    Returns
    -------
    complex
        Transfer function T
    """
    magnitude = np.sqrt(P_Y * gamma2 / P_X)
    return magnitude * np.exp(1j * phi)


def compute_normalization_constant(P_X, P_Y, T):
    """
    Compute the normalization constant K for the incoherent component.
    
    From Equation 12 of the draft:
    K = sqrt((P_Y - P_X * |T|^2) / 2)
    
    Parameters
    ----------
    P_X : float
        Power spectrum of reference time series
    P_Y : float
        Power spectrum of dependent time series
    T : complex
        Transfer function
        
    Returns
    -------
    float
        Normalization constant K
    """
    T_mag_sq = np.abs(T)**2
    return np.sqrt((P_Y - P_X * T_mag_sq) / 2)


def generate_fourier_realization(P_X, P_Y, T, K, rng):
    """
    Generate a single realization of the Fourier transforms X and Y.
    
    From Equations 9 and 10 of the draft:
    X = sqrt(P_X/2) * (A_r + i*B_r)
    Y = K*(H_r + i*J_r) + T*X
    
    Parameters
    ----------
    P_X : float
        Power spectrum of reference time series
    P_Y : float
        Power spectrum of dependent time series
    T : complex
        Transfer function
    K : float
        Normalization constant
    rng : numpy.random.Generator
        Random number generator
        
    Returns
    -------
    tuple (complex, complex)
        Fourier transforms (X, Y)
    """
    # Draw random variables from standard normal
    A_r, B_r, H_r, J_r = rng.standard_normal(4)
    
    # Reference Fourier transform (Eq. 9)
    X = np.sqrt(P_X / 2) * (A_r + 1j * B_r)
    
    # Dependent Fourier transform (Eq. 10)
    Y = K * (H_r + 1j * J_r) + T * X
    
    return X, Y


def simulate_averaged_quantities(P_X, P_Y, gamma2, phi, n_segments, rng):
    """
    Simulate averaged power spectra and cross-spectrum over n segments.
    
    Parameters
    ----------
    P_X : float
        Target power spectrum of reference
    P_Y : float
        Target power spectrum of dependent
    gamma2 : float
        Target coherence
    phi : float
        Target phase lag
    n_segments : int
        Number of segments to average
    rng : numpy.random.Generator
        Random number generator
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'P_X_hat': Estimated power spectrum of X
        - 'P_Y_hat': Estimated power spectrum of Y
        - 'C_r': Real part of cross-spectrum
        - 'C_i': Imaginary part of cross-spectrum
        - 'gamma2_hat': Estimated coherence
        - 'phi_hat': Estimated phase lag
    """
    # Compute transfer function and normalization
    T = compute_transfer_function(gamma2, phi, P_X, P_Y)
    K = compute_normalization_constant(P_X, P_Y, T)
    
    # Accumulate sums for averaging
    sum_XX = 0.0
    sum_YY = 0.0
    sum_XY = 0.0 + 0.0j
    
    for _ in range(n_segments):
        X, Y = generate_fourier_realization(P_X, P_Y, T, K, rng)
        
        # Accumulate power spectra (|X|^2, |Y|^2)
        sum_XX += np.abs(X)**2
        sum_YY += np.abs(Y)**2
        
        # Accumulate cross-spectrum (X* Y)
        sum_XY += np.conj(X) * Y
    
    # Compute averages
    P_X_hat = sum_XX / n_segments
    P_Y_hat = sum_YY / n_segments
    C_hat = sum_XY / n_segments
    
    # Extract real and imaginary parts of cross-spectrum
    C_r = np.real(C_hat)
    C_i = np.imag(C_hat)
    
    # Compute coherence and phase lag estimates
    gamma2_hat = np.abs(C_hat)**2 / (P_X_hat * P_Y_hat)
    phi_hat = np.arctan2(C_i, C_r)
    
    return {
        'P_X_hat': P_X_hat,
        'P_Y_hat': P_Y_hat,
        'C_r': C_r,
        'C_i': C_i,
        'gamma2_hat': gamma2_hat,
        'phi_hat': phi_hat
    }


def run_monte_carlo(P_X, P_Y, gamma2, phi, n_segments, n_realizations, seed=None):
    """
    Run Monte Carlo simulation to generate distribution of estimated quantities.
    
    Parameters
    ----------
    P_X : float
        Target power spectrum of reference
    P_Y : float
        Target power spectrum of dependent
    gamma2 : float
        Target coherence
    phi : float
        Target phase lag
    n_segments : int
        Number of segments to average per realization
    n_realizations : int
        Number of Monte Carlo realizations
    seed : int or None
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary with arrays of all estimated quantities
    """
    rng = np.random.default_rng(seed)
    
    # Storage arrays
    results = {
        'P_X_hat': np.zeros(n_realizations),
        'P_Y_hat': np.zeros(n_realizations),
        'C_r': np.zeros(n_realizations),
        'C_i': np.zeros(n_realizations),
        'gamma2_hat': np.zeros(n_realizations),
        'phi_hat': np.zeros(n_realizations)
    }
    
    for i in range(n_realizations):
        realization = simulate_averaged_quantities(
            P_X, P_Y, gamma2, phi, n_segments, rng
        )
        for key in results:
            results[key][i] = realization[key]
    
    return results


# =============================================================================
# Plotting Functions
# =============================================================================

def create_corner_plot_coherence_phase(results, gamma2_true, phi_true, n_segments, 
                                       show_bias_line=False, show_theory_contours=False):
    """
    Create corner plot showing joint distribution of coherence and phase lag.
    """
    samples = np.column_stack([results['gamma2_hat'], results['phi_hat']])
    
    fig = corner.corner(
        samples,
        labels=[r'$\hat{\gamma}^2$', r'$\hat{\phi}$'],
        truths=[gamma2_true, phi_true],
        truth_color='C1',
        quantiles=QUANTILES,
        levels=(0.393, 0.864),
        show_titles=True,
        title_kwargs={'fontsize': 16},
        label_kwargs={'fontsize': 18},
        hist_kwargs={'density': True},
    )
    
    axes = np.array(fig.axes).reshape((2, 2))
    
    # Add expected (biased) coherence line if requested
    if show_bias_line:
        gamma2_expected = expected_biased_coherence(gamma2_true, n_segments)

        # Diagonal: γ histogram (top-left, index [0,0])
        axes[0, 0].axvline(gamma2_expected, color='C2', linestyle='--',
                           linewidth=2.5, label=r'$E[\hat{\gamma}^2]$')

        # Off-diagonal: γ vs φ (bottom-left, index [1,0])
        axes[1, 0].axvline(gamma2_expected, color='C2', linestyle='--', linewidth=2.5)
        
        # Add legend to top-left panel
        axes[0, 0].legend(fontsize=12, loc='upper left')
    
    # Add theoretical variance contours if requested
    if show_theory_contours:
        var_gamma2, var_phi = theoretical_variance_gamma2_phi(gamma2_true, n_segments)
        
        # 2D covariance for (γ, φ) - assuming independent for leading order
        # (the full covariance has off-diagonal terms, but they're typically small)
        cov_2x2 = np.diag([var_gamma2, var_phi])
        
        # Draw 1sigma and 2sigma ellipses on the off-diagonal panel
        for n_std, alpha in CONTOUR_LEVELS_ALPHAS:
            draw_covariance_ellipse(
                axes[1, 0],
                mean=[gamma2_true, phi_true],
                cov_2x2=cov_2x2,
                n_std=n_std,
                facecolor='none',
                edgecolor='C1',
                linestyle='-',
                linewidth=3.0,
                alpha=alpha,
                label=f'{n_std}sigma theory' if n_std == 1 else None
            )

        # Add vertical lines on diagonal histograms for Â±1sigma, Â±2sigma
        std_gamma2 = np.sqrt(var_gamma2)
        std_phi = np.sqrt(var_phi)

        for n_std, alpha in CONTOUR_LEVELS_ALPHAS:
            axes[0, 0].axvline(gamma2_true - n_std * std_gamma2, color='C3',
                               linestyle=':', alpha=alpha, linewidth=2.0)
            axes[0, 0].axvline(gamma2_true + n_std * std_gamma2, color='C3',
                               linestyle=':', alpha=alpha, linewidth=2.0)
            axes[1, 1].axvline(phi_true - n_std * std_phi, color='C3',
                               linestyle=':', alpha=alpha, linewidth=2.0)
            axes[1, 1].axvline(phi_true + n_std * std_phi, color='C3',
                               linestyle=':', alpha=alpha, linewidth=2.0)
    
    # fig.suptitle(
    #     f'Joint Distribution of Coherence and Phase Lag\n'
    #     f'($n = {n_segments}$ segments, '
    #     f'$\\gamma^2_{{\\rm true}} = {gamma2_true:.2f}$, '
    #     f'$\\phi_{{\\rm true}} = {phi_true:.3f}$ rad)',
    #     fontsize=14, y=1.08
    # )
    plt.subplots_adjust(top=0.92)
    
    return fig


def create_corner_plot_spectra(results, P_X_true, P_Y_true, gamma2_true, phi_true, 
                                n_segments, show_bias_line=False, show_theory_contours=False):
    """
    Create corner plot showing joint distribution of power spectra and cross-spectrum.
    
    Note: show_bias_line parameter included for API consistency but has no effect
    on this plot since coherence is not displayed.
    """
    # True values for cross-spectrum
    C_r_true = np.sqrt(P_X_true * P_Y_true) * np.sqrt(gamma2_true) * np.cos(phi_true)
    C_i_true = np.sqrt(P_X_true * P_Y_true) * np.sqrt(gamma2_true) * np.sin(phi_true)
    
    samples = np.column_stack([
        results['P_X_hat'],
        results['P_Y_hat'],
        results['C_r'],
        results['C_i']
    ])
    
    fig = corner.corner(
        samples,
        labels=[r'$\hat{P}_X$', r'$\hat{P}_Y$', r'$\hat{C}_r$', r'$\hat{C}_i$'],
        truths=[P_X_true, P_Y_true, C_r_true, C_i_true],
        truth_color='C1',
        quantiles=QUANTILES,
        levels=(0.393, 0.864),
        show_titles=True,
        title_kwargs={'fontsize': 14},
        label_kwargs={'fontsize': 16},
        hist_kwargs={'density': True},
    )
    
    # Add theoretical variance contours if requested
    if show_theory_contours:
        # Get the full 4x4 covariance matrix (in order C_r, C_i, P_X, P_Y)
        Sigma_draft = theoretical_covariance_matrix(P_X_true, P_Y_true, gamma2_true, 
                                                     phi_true, n_segments)
        
        # Reorder to match plotting order (P_X, P_Y, C_r, C_i)
        # Draft order: C_r=0, C_i=1, P_X=2, P_Y=3
        # Plot order:  P_X=0, P_Y=1, C_r=2, C_i=3
        reorder = [2, 3, 0, 1]  # Map from plot index to draft index
        Sigma = np.zeros((4, 4))
        for i, ri in enumerate(reorder):
            for j, rj in enumerate(reorder):
                Sigma[i, j] = Sigma_draft[ri, rj]
        
        ndim = 4
        axes = np.array(fig.axes).reshape((ndim, ndim))
        truths = [P_X_true, P_Y_true, C_r_true, C_i_true]
        
        # Draw ellipses on off-diagonal panels
        for i in range(ndim):
            for j in range(i):
                # Extract 2x2 submatrix for variables j and i
                indices = [j, i]
                cov_2x2 = Sigma[np.ix_(indices, indices)]
                mean = [truths[j], truths[i]]
                
                for n_std, alpha in CONTOUR_LEVELS_ALPHAS:
                    draw_covariance_ellipse(
                        axes[i, j],
                        mean=mean,
                        cov_2x2=cov_2x2,
                        n_std=n_std,
                        facecolor='none',
                        edgecolor='C1',
                        linestyle='-',
                        linewidth=3.0,
                        alpha=alpha
                    )

        # Draw Â±1sigma, Â±2sigma lines on diagonal histograms
        for i in range(ndim):
            std_i = np.sqrt(Sigma[i, i])
            for n_std, alpha in CONTOUR_LEVELS_ALPHAS:
                axes[i, i].axvline(truths[i] - n_std * std_i, color='C3',
                                   linestyle=':', alpha=alpha, linewidth=2.0)
                axes[i, i].axvline(truths[i] + n_std * std_i, color='C3',
                                   linestyle=':', alpha=alpha, linewidth=2.0)
    
    # fig.suptitle(
    #     f'Joint Distribution of Power Spectra and Cross-Spectrum Components\n'
    #     f'($n = {n_segments}$ segments)',
    #     fontsize=14, y=1.05
    # )
    plt.subplots_adjust(top=0.93)
    
    return fig


def create_corner_plot_all(results, P_X_true, P_Y_true, gamma2_true, phi_true, 
                           n_segments, show_bias_line=False, show_theory_contours=False):
    """
    Create corner plot showing joint distribution of all six quantities.
    """
    # True values for cross-spectrum
    C_r_true = np.sqrt(P_X_true * P_Y_true) * np.sqrt(gamma2_true) * np.cos(phi_true)
    C_i_true = np.sqrt(P_X_true * P_Y_true) * np.sqrt(gamma2_true) * np.sin(phi_true)
    
    samples = np.column_stack([
        results['P_X_hat'],
        results['P_Y_hat'],
        results['C_r'],
        results['C_i'],
        results['gamma2_hat'],
        results['phi_hat']
    ])
    
    fig = corner.corner(
        samples,
        labels=[r'$\hat{P}_X$', r'$\hat{P}_Y$', r'$\hat{C}_r$', r'$\hat{C}_i$',
                r'$\hat{\gamma}^2$', r'$\hat{\phi}$'],
        truths=[P_X_true, P_Y_true, C_r_true, C_i_true, gamma2_true, phi_true],
        truth_color='C1',
        quantiles=QUANTILES,
        levels=(0.393, 0.864),
        show_titles=True,
        title_kwargs={'fontsize': 12},
        label_kwargs={'fontsize': 14},
        hist_kwargs={'density': True},
    )
    
    ndim = 6
    axes = np.array(fig.axes).reshape((ndim, ndim))
    gamma2_idx = 4  # Index of γ in the variable list
    phi_idx = 5     # Index of φ in the variable list
    
    # Add expected (biased) coherence line if requested
    if show_bias_line:
        gamma2_expected = expected_biased_coherence(gamma2_true, n_segments)

        # Diagonal: γ histogram
        axes[gamma2_idx, gamma2_idx].axvline(gamma2_expected, color='C2',
                                              linestyle='--', linewidth=2.5)

        # Off-diagonal panels in γ column (below diagonal)
        for i in range(gamma2_idx + 1, ndim):
            axes[i, gamma2_idx].axvline(gamma2_expected, color='C2', linestyle='--', linewidth=2.5)

        # Off-diagonal panels in γ row (left of diagonal)
        for j in range(gamma2_idx):
            axes[gamma2_idx, j].axhline(gamma2_expected, color='C2', linestyle='--', linewidth=2.5)
    
    # Add theoretical variance contours if requested
    if show_theory_contours:
        # Get the full 6x6 covariance matrix (order: P_X, P_Y, C_r, C_i, gamma^2, phi)
        Sigma_6, _ = theoretical_full_covariance_6x6(P_X_true, P_Y_true, gamma2_true,
                                                      phi_true, n_segments)
        
        truths = [P_X_true, P_Y_true, C_r_true, C_i_true, gamma2_true, phi_true]
        
        # Draw ellipses on ALL off-diagonal panels
        for i in range(ndim):
            for j in range(i):
                # Extract 2x2 submatrix for variables j and i
                indices = [j, i]
                cov_2x2 = Sigma_6[np.ix_(indices, indices)]
                mean = [truths[j], truths[i]]
                
                # Draw 1sigma and 2sigma ellipses
                for n_std, alpha in CONTOUR_LEVELS_ALPHAS:
                    draw_covariance_ellipse(
                        axes[i, j],
                        mean=mean,
                        cov_2x2=cov_2x2,
                        n_std=n_std,
                        facecolor='none',
                        edgecolor='C1',
                        linestyle='-',
                        linewidth=3.0,
                        alpha=alpha
                    )

        # Draw +/-1sigma, +/-2sigma lines on diagonal histograms
        for i in range(ndim):
            std_i = np.sqrt(Sigma_6[i, i])
            for n_std, alpha in CONTOUR_LEVELS_ALPHAS:
                axes[i, i].axvline(truths[i] - n_std * std_i, color='C3',
                                   linestyle=':', alpha=alpha, linewidth=2.0)
                axes[i, i].axvline(truths[i] + n_std * std_i, color='C3',
                                   linestyle=':', alpha=alpha, linewidth=2.0)
    
    # fig.suptitle(
    #     f'Joint Distribution of All Estimated Quantities\n'
    #     f'($n = {n_segments}$ segments, '
    #     f'$\\gamma^2_{{\\rm true}} = {gamma2_true:.2f}$, '
    #     f'$\\phi_{{\\rm true}} = {phi_true:.3f}$ rad)',
    #     fontsize=12, y=1.02
    # )
    plt.subplots_adjust(top=0.95)
    
    return fig


def plot_correlation_matrices(results, P_X_true, P_Y_true, gamma2_true, phi_true, n_segments):
    """
    Plot comparison of theoretical vs simulated correlation matrices.
    """
    # Variable labels in order
    labels = [r'$P_X$', r'$P_Y$', r'$C_r$', r'$C_i$', r'$\gamma^2$', r'$\phi$']
    
    # Compute theoretical covariance matrix
    Sigma_theory, _ = theoretical_full_covariance_6x6(P_X_true, P_Y_true, gamma2_true, 
                                                      phi_true, n_segments)
    
    # Convert to correlation matrix
    std_theory = np.sqrt(np.diag(Sigma_theory))
    Corr_theory = Sigma_theory / np.outer(std_theory, std_theory)
    
    # Compute simulated covariance matrix
    samples = np.column_stack([
        results['P_X_hat'],
        results['P_Y_hat'],
        results['C_r'],
        results['C_i'],
        results['gamma2_hat'],
        results['phi_hat']
    ])
    Sigma_sim = np.cov(samples.T)
    
    # Convert to correlation matrix
    std_sim = np.sqrt(np.diag(Sigma_sim))
    Corr_sim = Sigma_sim / np.outer(std_sim, std_sim)
    
    # Create figure with three panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Theory correlation matrix
    im1 = axes[0].imshow(Corr_theory, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[0].set_xticks(range(6))
    axes[0].set_yticks(range(6))
    axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=14)
    axes[0].set_yticklabels(labels, fontsize=14)
    axes[0].set_title('Theoretical Correlation Matrix', fontsize=16, pad=10)

    # Add text annotations
    for i in range(6):
        for j in range(6):
            text = axes[0].text(j, i, f'{Corr_theory[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Simulated correlation matrix
    im2 = axes[1].imshow(Corr_sim, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[1].set_xticks(range(6))
    axes[1].set_yticks(range(6))
    axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=14)
    axes[1].set_yticklabels(labels, fontsize=14)
    axes[1].set_title('Simulated Correlation Matrix', fontsize=16, pad=10)

    # Add text annotations
    for i in range(6):
        for j in range(6):
            text = axes[1].text(j, i, f'{Corr_sim[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Difference (simulation - theory)
    diff = Corr_sim - Corr_theory
    max_abs_diff = np.max(np.abs(diff))
    im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-max_abs_diff, vmax=max_abs_diff, aspect='auto')
    axes[2].set_xticks(range(6))
    axes[2].set_yticks(range(6))
    axes[2].set_xticklabels(labels, rotation=45, ha='right', fontsize=14)
    axes[2].set_yticklabels(labels, fontsize=14)
    axes[2].set_title('Difference (Sim - Theory)', fontsize=16, pad=10)

    # Add text annotations
    for i in range(6):
        for j in range(6):
            text = axes[2].text(j, i, f'{diff[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    # fig.suptitle(
    #     f'Correlation Matrix Comparison (n = {n_segments} segments, '
    #     f'$\\gamma^2 = {gamma2_true:.2f}$, $\\phi = {phi_true:.3f}$ rad)',
    #     fontsize=14, y=1.02
    # )
    plt.tight_layout()
    
    return fig


def print_covariance_comparison(results, P_X_true, P_Y_true, gamma2_true, phi_true, n_segments):
    """
    Print detailed comparison of theoretical vs simulated covariances.
    """
    # Get theoretical covariances
    Sigma_theory, cov_dict = theoretical_full_covariance_6x6(P_X_true, P_Y_true, 
                                                             gamma2_true, phi_true, n_segments)
    
    # Compute simulated covariances
    samples = np.column_stack([
        results['P_X_hat'],
        results['P_Y_hat'],
        results['C_r'],
        results['C_i'],
        results['gamma2_hat'],
        results['phi_hat']
    ])
    Sigma_sim = np.cov(samples.T)
    
    labels = ['P_X', 'P_Y', 'C_r', 'C_i', 'gamma2', 'phi']
    
    print("\n" + "="*80)
    print("COVARIANCE COMPARISON: Theory vs Simulation")
    print("="*80)
    
    # Print variances first
    print("\nVARIANCES:")
    print(f"{'Variable':<10} {'Theory':>15} {'Simulation':>15} {'Difference':>15} {'Rel. Diff %':>15}")
    print("-"*80)
    for i, label in enumerate(labels):
        theory_val = Sigma_theory[i, i]
        sim_val = Sigma_sim[i, i]
        diff = sim_val - theory_val
        rel_diff = 100 * diff / theory_val if theory_val != 0 else np.nan
        print(f"{label:<10} {theory_val:>15.6e} {sim_val:>15.6e} {diff:>15.6e} {rel_diff:>15.2f}")
    
    # Print key cross-covariances
    print("\nKEY CROSS-COVARIANCES:")
    print(f"{'Pair':<15} {'Theory':>15} {'Simulation':>15} {'Difference':>15}")
    print("-"*80)
    
    # Covariances involving gamma2 and phi
    key_pairs = [
        (4, 5, 'gamma2_phi'),
        (4, 0, 'gamma2_PX'),
        (4, 1, 'gamma2_PY'),
        (4, 2, 'gamma2_Cr'),
        (4, 3, 'gamma2_Ci'),
        (5, 0, 'phi_PX'),
        (5, 1, 'phi_PY'),
        (5, 2, 'phi_Cr'),
        (5, 3, 'phi_Ci'),
    ]
    
    for i, j, name in key_pairs:
        theory_val = Sigma_theory[i, j]
        sim_val = Sigma_sim[i, j]
        diff = sim_val - theory_val
        print(f"{name:<15} {theory_val:>15.6e} {sim_val:>15.6e} {diff:>15.6e}")
    
    print("\n" + "="*80)
    print("NOTES:")
    print("  - Cov(γ², φ) should be exactly zero by theory (symmetry argument)")
    print("  - Cov(φ, P_X) and Cov(φ, P_Y) should be exactly zero (φ depends only on C_r/C_i)")
    print("="*80 + "\n")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Correlated Time Series Simulation - Corner Plot Analysis")
    print("=" * 70)
    print(f"\nSimulation Parameters:")
    print(f"  Target coherence:     γ = {TARGET_GAMMA2:.3f}")
    print(f"  Target phase lag:     φ  = {TARGET_PHI:.4f} rad ({np.degrees(TARGET_PHI):.1f}°)")
    print(f"  Segments per realization: n = {N_SEGMENTS}")
    print(f"  Monte Carlo realizations: {N_REALIZATIONS}")
    print(f"  P_X = {P_X}, P_Y = {P_Y}")
    print()
    
    # Run Monte Carlo simulation
    print("Running Monte Carlo simulation...")
    results = run_monte_carlo(
        P_X, P_Y, TARGET_GAMMA2, TARGET_PHI,
        N_SEGMENTS, N_REALIZATIONS, seed=RANDOM_SEED
    )
    print("Done.\n")
    
    
    
    # Create corner plots
    print("Generating corner plots...")
    
    if SHOW_BIAS_LINE:
        gamma2_expected = expected_biased_coherence(TARGET_GAMMA2, N_SEGMENTS)
        print(f"  Expected biased coherence: E[gamma2] = {gamma2_expected:.4f}")
    
    if SHOW_THEORY_CONTOURS:
        print("  Theoretical contours enabled (1sigma and 2sigma)")
    
    # Plot 1: Coherence and Phase Lag
    fig1 = create_corner_plot_coherence_phase(
        results, TARGET_GAMMA2, TARGET_PHI, N_SEGMENTS,
        show_bias_line=SHOW_BIAS_LINE,
        show_theory_contours=SHOW_THEORY_CONTOURS
    )
    fig1.savefig(f'{SAVE_DIR}/corner_coherence_phase.png', 
                 dpi=DPI, bbox_inches='tight')
    print("  Saved: corner_coherence_phase.png")
    
    # Plot 2: Power Spectra and Cross-Spectrum
    fig2 = create_corner_plot_spectra(
        results, P_X, P_Y, TARGET_GAMMA2, TARGET_PHI, N_SEGMENTS,
        show_bias_line=SHOW_BIAS_LINE,
        show_theory_contours=SHOW_THEORY_CONTOURS
    )
    fig2.savefig(f'{SAVE_DIR}/corner_spectra.png', 
                 dpi=DPI, bbox_inches='tight')
    print("  Saved: corner_spectra.png")
    
    # Plot 3: All Quantities
    fig3 = create_corner_plot_all(
        results, P_X, P_Y, TARGET_GAMMA2, TARGET_PHI, N_SEGMENTS,
        show_bias_line=SHOW_BIAS_LINE,
        show_theory_contours=SHOW_THEORY_CONTOURS
    )
    fig3.savefig(f'{SAVE_DIR}/corner_all_quantities.png', 
                 dpi=DPI, bbox_inches='tight')
    print("  Saved: corner_all_quantities.png")
    

    print("\nAll plots generated successfully.")
    plt.close('all')
