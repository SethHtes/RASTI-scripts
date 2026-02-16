"""
Ensemble Validation Plotting Script

This script validates the simulation code by comparing ensemble-averaged simulations
against input models for PSDs, coherence, and phase lags. It demonstrates that the
CLSimulator correctly implements the Larner et al. (2026) coherent component framework.

The script:
1. Creates Q-based PSD models with 2 Lorentzian components
2. Creates coherence and phase lag models with specified phase differences
3. Simulates N=100 light curve pairs (ensemble)
4. Computes PSDs and cross-spectra for each realization
5. Plots mean ± standard error vs input models
6. Saves figure to examples/ directory
"""

import numpy as np
import matplotlib.pyplot as plt

from synthetic_timeseries import CLSimulator
from stingray import AveragedPowerspectrum, AveragedCrossspectrum
from validation_models import psd_n_lorentz, psd_n_lorentz_B
from validation_models import coherence_n_lorentz, phase_diff_n_lorentz 

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# =============================================================================
# CONFIGURATION SECTION - User-tunable parameters
# =============================================================================

# Component parameters (2 Lorentzian QPO-like features)
N_COMPONENTS = 2

# Component 0 (lower frequency QPO)
A_0 = 0.012      # PSD1 amplitude (fractional norm)
B_0 = 0.05      # PSD2 amplitude
M_0 = 1.0        # Center frequency (Hz)
Q_0 = .4     # Quality factor
LAG_0 = 0.15      # Phase lag (radians)

# Component 1 (higher frequency QPO)
A_1 = 0.01
B_1 = 0.005
M_1 = 50.0
Q_1 = 1
LAG_1 = -0.8     # Negative = band 2 leads

# Simulation settings
N_REALIZATIONS = 10  # Number of ensemble realizations
TIME_BINS = 2**18        # ~262 seconds
DT = 0.001              # Time resolution (seconds)
MEAN_COUNTRATE = 1000.0  # Mean count rate (counts/s)
TARGET_RMS = 0.2        # 20% fractional RMS
SEGMENT_SIZE = 16.0     # Averaging segments (seconds)
NORMALIZATION = 'frac'  # Normalization type
REB_FREQ = 0.02         # Log rebinning factor

# Output
OUTPUT_FILE = 'validation_ensemble.png'
RANDOM_SEED = 42
VERBOSE = True
SHOW_INLINE = False  # Display figure inline for debugging (set to True to enable)

# =============================================================================
# MODEL SETUP FUNCTIONS
# =============================================================================

def create_psd_models(config):
    """
    Create two Q-based PSD models with shared centers and Q factors but different amplitudes.

    Parameters
    ----------
    config : dict
        Configuration dictionary with component parameters

    Returns
    -------
    psd1_model : Model
        Q-based PSD model for band 1 (A amplitudes)
    psd2_model : Model
        Q-based PSD model for band 2 (B amplitudes)
    """
    n = config['N_COMPONENTS']

    # Create base models
    psd1_model = psd_n_lorentz(n)
    psd2_model = psd_n_lorentz_B(n)

    # Convert to Q-based parameterization
    psd1_model.as_QBased(inplace=True)
    psd2_model.as_QBased(inplace=True)

    # Set parameters for each component
    for i in range(n):
        # Get parameters from config
        A_i = config[f'A_{i}']
        B_i = config[f'B_{i}']
        m_i = config[f'M_{i}']
        Q_i = config[f'Q_{i}']

        # Set PSD1 parameters (A amplitudes)
        setattr(psd1_model, f'A_{i}', A_i)
        setattr(psd1_model, f'm_{i}', m_i)
        setattr(psd1_model, f'Q_{i}', Q_i)

        # Set PSD2 parameters (B amplitudes)
        setattr(psd2_model, f'B_{i}', B_i)
        setattr(psd2_model, f'm_{i}', m_i)
        setattr(psd2_model, f'Q_{i}', Q_i)

    # Set noise to 0.0 for noiseless validation
    psd1_model.noise = 0.0
    psd2_model.noise = 0.0

    return psd1_model, psd2_model


def create_crossspectrum_models(config):
    """
    Create coherence and phase lag models.

    Parameters
    ----------
    config : dict
        Configuration dictionary with component parameters

    Returns
    -------
    coh_model : Model
        Coherence model
    lag_model : Model
        Phase lag model
    """
    n = config['N_COMPONENTS']

    # Create models
    coh_model = coherence_n_lorentz(n)
    lag_model = phase_diff_n_lorentz(n)

    # Convert to Q-based parameterization to avoid astropy unit validation issues
    coh_model.as_QBased(inplace=True)
    lag_model.as_QBased(inplace=True)

    # Set parameters for each component using Q-based parameterization
    for i in range(n):
        # Get parameters from config
        A_i = config[f'A_{i}']
        B_i = config[f'B_{i}']
        m_i = config[f'M_{i}']
        Q_i = config[f'Q_{i}']
        lag_i = config[f'LAG_{i}']

        # Set coherence model parameters (Q-based: A, B, m, Q)
        setattr(coh_model, f'A_{i}', A_i)
        setattr(coh_model, f'B_{i}', B_i)
        setattr(coh_model, f'm_{i}', m_i)
        setattr(coh_model, f'Q_{i}', Q_i)
        setattr(coh_model, f'lag_{i}', lag_i)

        # Set lag model parameters (Q-based: A, B, m, Q, lag)
        setattr(lag_model, f'A_{i}', A_i)
        setattr(lag_model, f'B_{i}', B_i)
        setattr(lag_model, f'm_{i}', m_i)
        setattr(lag_model, f'Q_{i}', Q_i)
        setattr(lag_model, f'lag_{i}', lag_i)

    # Clip coherence to avoid γ²=1.0 numerical issues in simulation
    # CLSimulator cannot handle perfect coherence (division by zero)
    max_coh = 0.999
    freq_test = np.logspace(-2, 2, 1000)
    try:
        coh_test = coh_model(freq_test)
        if np.any(coh_test >= 1.0):
            # Scale amplitudes to ensure coherence < max_coh
            scale_factor = np.sqrt(max_coh / np.max(coh_test))
            for i in range(n):
                A_i = getattr(coh_model, f'A_{i}')
                B_i = getattr(coh_model, f'B_{i}')
                setattr(coh_model, f'A_{i}', A_i * scale_factor)
                setattr(coh_model, f'B_{i}', B_i * scale_factor)
                setattr(lag_model, f'A_{i}', A_i * scale_factor)
                setattr(lag_model, f'B_{i}', B_i * scale_factor)
    except (AttributeError, TypeError) as e:
        # Skip coherence clipping if model evaluation fails
        # This can happen with certain astropy versions
        pass

    return coh_model, lag_model


# =============================================================================
# ENSEMBLE SIMULATION FUNCTIONS
# =============================================================================

def simulate_single_realization(psd1_model, psd2_model, coh_model, lag_model, config, seed=None):
    """
    Simulate a single realization of light curve pair and compute spectral products.

    Parameters
    ----------
    psd1_model : Model
        PSD model for band 1
    psd2_model : Model
        PSD model for band 2
    coh_model : Model
        Coherence model
    lag_model : Model
        Phase lag model
    config : dict
        Configuration dictionary
    seed : int, optional
        Random seed for this realization

    Returns
    -------
    result : dict
        Dictionary containing freq, psd1, psd2, coherence, phase_lag arrays
    """
    # Initialize CLSimulator
    simulator = CLSimulator(
        N=config['TIME_BINS'],
        dt=config['DT'],
        mean=config['MEAN_COUNTRATE'],
        rms=config['TARGET_RMS'],
        red_noise=1,
        random_state=seed
    )

    # Create callable functions for PSD, coherence, and lag
    # CLSimulator expects functions (wrapping avoids astropy unit validation issues)
    def pds1_func(freq):
        return psd1_model(freq)

    def pds2_func(freq):
        return psd2_model(freq)

    def coh_func(freq):
        # Subtract small value to avoid coherence = 1.0 (known CL_simulate limitation)
        return coh_model(freq) - 0.001

    def lag_func(freq):
        return lag_model(freq)

    # Generate light curves
    lc1, lc2 = simulator.CL_simulate(
        pds1=pds1_func,
        pds2=pds2_func,
        coh=coh_func,
        lag=lag_func
    )

    # Compute averaged power spectra
    ps1 = AveragedPowerspectrum.from_lightcurve(
        lc1,
        segment_size=config['SEGMENT_SIZE'],
        norm=config['NORMALIZATION'],
        silent=True
    )

    ps2 = AveragedPowerspectrum.from_lightcurve(
        lc2,
        segment_size=config['SEGMENT_SIZE'],
        norm=config['NORMALIZATION'],
        silent=True
    )

    cs = AveragedCrossspectrum.from_lightcurve(
        lc1, lc2,
        segment_size=config['SEGMENT_SIZE'],
        norm=config['NORMALIZATION'],
        silent=True
    )

    # Apply logarithmic rebinning
    ps1_reb = ps1.rebin_log(config['REB_FREQ'])
    ps2_reb = ps2.rebin_log(config['REB_FREQ'])
    cs_reb = cs.rebin_log(config['REB_FREQ'])

    # Extract coherence and phase lag
    coherence, _ = cs_reb.coherence()
    phase_lag, _ = cs_reb.phase_lag()

    # Return results
    return {
        'freq': cs_reb.freq,
        'psd1': ps1_reb.power,
        'psd2': ps2_reb.power,
        'coherence': coherence,
        'phase_lag': phase_lag
    }


def run_ensemble_simulation(psd1_model, psd2_model, coh_model, lag_model, config):
    """
    Run ensemble simulation with N realizations.

    Parameters
    ----------
    psd1_model : Model
        PSD model for band 1
    psd2_model : Model
        PSD model for band 2
    coh_model : Model
        Coherence model
    lag_model : Model
        Phase lag model
    config : dict
        Configuration dictionary

    Returns
    -------
    ensemble_data : dict
        Dictionary with freq array and 2D arrays for each quantity (N_realizations × n_freq)
    """
    N = config['N_REALIZATIONS']
    seed = config['RANDOM_SEED']
    verbose = config['VERBOSE']

    # Run first realization to get frequency grid dimensions
    if verbose:
        print(f"  Realization 1/{N}...")

    first_seed = seed if seed is not None else None
    first_result = simulate_single_realization(
        psd1_model, psd2_model, coh_model, lag_model, config, seed=first_seed
    )

    # Initialize storage arrays
    n_freq = len(first_result['freq'])
    ensemble_data = {
        'freq': first_result['freq'],
        'psd1': np.zeros((N, n_freq)),
        'psd2': np.zeros((N, n_freq)),
        'coherence': np.zeros((N, n_freq)),
        'phase_lag': np.zeros((N, n_freq))
    }

    # Store first result
    ensemble_data['psd1'][0, :] = first_result['psd1']
    ensemble_data['psd2'][0, :] = first_result['psd2']
    ensemble_data['coherence'][0, :] = first_result['coherence']
    ensemble_data['phase_lag'][0, :] = first_result['phase_lag']

    # Loop over remaining realizations
    for i in range(1, N):
        if verbose and (i % 10 == 0 or i == N - 1):
            print(f"  Realization {i+1}/{N}...")

        # Generate new seed
        seed_i = (seed + i) if seed is not None else None

        # Simulate
        result = simulate_single_realization(
            psd1_model, psd2_model, coh_model, lag_model, config, seed=seed_i
        )

        # Handle potential frequency grid mismatch (rare but possible)
        if len(result['freq']) != n_freq:
            # Interpolate to reference grid
            for key in ['psd1', 'psd2', 'coherence', 'phase_lag']:
                result[key] = np.interp(
                    ensemble_data['freq'],
                    result['freq'],
                    result[key]
                )

        # Store results
        ensemble_data['psd1'][i, :] = result['psd1']
        ensemble_data['psd2'][i, :] = result['psd2']
        ensemble_data['coherence'][i, :] = result['coherence']
        ensemble_data['phase_lag'][i, :] = result['phase_lag']

    return ensemble_data


def compute_ensemble_statistics(ensemble_data):
    """
    Compute mean and standard error across ensemble.

    Parameters
    ----------
    ensemble_data : dict
        Dictionary with 2D arrays for each quantity

    Returns
    -------
    stats : dict
        Dictionary with {quantity}_mean and {quantity}_stderr for each quantity
    """
    N = ensemble_data['psd1'].shape[0]

    stats = {
        'freq': ensemble_data['freq'],
        'psd1_mean': np.mean(ensemble_data['psd1'], axis=0),
        'psd1_stderr': np.std(ensemble_data['psd1'], axis=0) / np.sqrt(N),
        'psd2_mean': np.mean(ensemble_data['psd2'], axis=0),
        'psd2_stderr': np.std(ensemble_data['psd2'], axis=0) / np.sqrt(N),
        'coherence_mean': np.mean(ensemble_data['coherence'], axis=0),
        'coherence_stderr': np.std(ensemble_data['coherence'], axis=0) / np.sqrt(N),
        'phase_lag_mean': np.mean(ensemble_data['phase_lag'], axis=0),
        'phase_lag_stderr': np.std(ensemble_data['phase_lag'], axis=0) / np.sqrt(N)
    }

    return stats


# =============================================================================
# VISUALIZATION FUNCTION
# =============================================================================

def create_validation_plot(stats, psd1_model, psd2_model, coh_model, lag_model, config):
    """
    Create 4-panel validation plot.

    Parameters
    ----------
    stats : dict
        Ensemble statistics dictionary
    psd1_model : Model
        PSD model for band 1
    psd2_model : Model
        PSD model for band 2
    coh_model : Model
        Coherence model
    lag_model : Model
        Phase lag model
    config : dict
        Configuration dictionary

    Returns
    -------
    fig : Figure
        Matplotlib figure object
    axes : array
        Array of axes objects
    """
    # Create 4-panel figure (4 rows × 1 column, shared x-axis)
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # Get frequency arrays
    freq = stats['freq']
    freq_fine = np.logspace(np.log10(freq.min()), np.log10(freq.max()), 1000)

    # Compute PSD normalization correction
    # The simulator scales the input PSD to achieve the target RMS
    # We need to apply the same scaling to the model PSDs for proper comparison
    freq_sim = np.fft.rfftfreq(config['TIME_BINS'], d=config['DT'])[1:]
    psd1_input = psd1_model(freq_sim)
    psd2_input = psd2_model(freq_sim)

    # For fractional normalization: RMS² = integral(PSD * df)
    rms1_input = np.sqrt(np.trapz(psd1_input, freq_sim))
    rms2_input = np.sqrt(np.trapz(psd2_input, freq_sim))
    target_rms = config['TARGET_RMS']

    # Scale factors: (target_rms / input_rms)²
    scale1 = (target_rms / rms1_input) ** 2
    scale2 = (target_rms / rms2_input) ** 2


    # Panel 0 - PSD1 (log-log)
    axes[0].errorbar(
        freq, stats['psd1_mean'] * freq, yerr=stats['psd1_stderr'],
        fmt='o', color='black', markersize=6, capsize=2,
        label='Ensemble mean', zorder=2, alpha=0.7
    )
    axes[0].plot(
        freq_fine, psd1_model(freq_fine) * scale1 * freq_fine,
        color='red', linewidth=3, label='Input model', zorder=1
    )

    # Plot individual Lorentzian components
    n_comp = config['N_COMPONENTS']
    for i in range(n_comp):
        # Create single-component model
        comp_model = psd_n_lorentz(1)
        comp_model.as_QBased(inplace=True)
        comp_model.A_0 = getattr(psd1_model, f'A_{i}')
        comp_model.m_0 = getattr(psd1_model, f'm_{i}')
        comp_model.Q_0 = getattr(psd1_model, f'Q_{i}')
        comp_model.noise = 0.0

        axes[0].plot(
            freq_fine, comp_model(freq_fine) * scale1 * freq_fine,
            color='red', linewidth=2, linestyle='--', alpha=0.6, zorder=0
        )

    axes[0].set(xscale='log', yscale='log', ylabel=r'$\nu \times \hat{P}_X$ (frac²)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=14)

    # Panel 1 - PSD2 (log-log)
    axes[1].errorbar(
        freq, stats['psd2_mean'] * freq, yerr=stats['psd2_stderr'],
        fmt='o', color='black', markersize=6, capsize=2,
        label='Ensemble mean', zorder=2, alpha=0.7
    )
    axes[1].plot(
        freq_fine, psd2_model(freq_fine) * scale2 * freq_fine,
        color='red', linewidth=3, label='Input model', zorder=1
    )

    # Plot individual Lorentzian components
    for i in range(n_comp):
        # Create single-component model
        comp_model = psd_n_lorentz_B(1)
        comp_model.as_QBased(inplace=True)
        comp_model.B_0 = getattr(psd2_model, f'B_{i}')
        comp_model.m_0 = getattr(psd2_model, f'm_{i}')
        comp_model.Q_0 = getattr(psd2_model, f'Q_{i}')
        comp_model.noise = 0.0

        axes[1].plot(
            freq_fine, comp_model(freq_fine) * scale2 * freq_fine,
            color='red', linewidth=2, linestyle='--', alpha=0.6, zorder=0
        )

    axes[1].set(xscale='log', yscale='log', ylabel=r'$\nu \times \hat{P}_Y$ (frac²)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=14)

    # Panel 2 - Coherence (log-x, linear-y)
    axes[2].errorbar(
        freq, stats['coherence_mean'], yerr=stats['coherence_stderr'],
        fmt='o', color='black', markersize=6, capsize=2,
        label='Ensemble mean', zorder=2, alpha=0.7
    )
    axes[2].plot(
        freq_fine, coh_model(freq_fine),
        color='red', linewidth=3, label='Input model', zorder=1
    )
    axes[2].axhline(1.0, color='gray', linestyle='--', alpha=0.5, zorder=0)
    axes[2].set(xscale='log', ylabel=r'$\hat{\gamma}^2$', ylim=[.5, 1.05])
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='best', fontsize=14)

    # Panel 3 - Phase Lag (log-x, linear-y with π ticks)
    axes[3].errorbar(
        freq, stats['phase_lag_mean'], yerr=stats['phase_lag_stderr'],
        fmt='o', color='black', markersize=6, capsize=2,
        label='Ensemble mean', zorder=2, alpha=0.7
    )
    axes[3].plot(
        freq_fine, lag_model(freq_fine),
        color='red', linewidth=3, label='Input model', zorder=1
    )
    axes[3].axhline(0.0, color='gray', linestyle='--', alpha=0.5, zorder=0)
    axes[3].set(
        xscale='log',
        xlabel='Frequency (Hz)',
        ylabel=r'$\hat{\phi}$',
        ylim=[-np.pi/2 - 0.2, np.pi/2 + 0.2]
    )
    axes[3].set_yticks([ -np.pi/2, 0, np.pi/2])
    axes[3].set_yticklabels([ r'$-\pi/2$', '0', r'$\pi/2$'], fontsize=14)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='best', fontsize=14)

    # Overall styling
    # fig.suptitle(
    #     f'Ensemble Validation (N={config["N_REALIZATIONS"]} realizations)',
    #     fontsize=14, y=0.995
    # )
    plt.tight_layout()

    return fig, axes


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    # Create config dict from top-level parameters
    config = {
        'N_COMPONENTS': N_COMPONENTS,
        'A_0': A_0, 'B_0': B_0, 'M_0': M_0, 'Q_0': Q_0, 'LAG_0': LAG_0,
        'A_1': A_1, 'B_1': B_1, 'M_1': M_1, 'Q_1': Q_1, 'LAG_1': LAG_1,
        'N_REALIZATIONS': N_REALIZATIONS,
        'TIME_BINS': TIME_BINS,
        'DT': DT,
        'MEAN_COUNTRATE': MEAN_COUNTRATE,
        'TARGET_RMS': TARGET_RMS,
        'SEGMENT_SIZE': SEGMENT_SIZE,
        'NORMALIZATION': NORMALIZATION,
        'REB_FREQ': REB_FREQ,
        'RANDOM_SEED': RANDOM_SEED,
        'VERBOSE': VERBOSE,
        'SHOW_INLINE': SHOW_INLINE
    }

    print("=" * 70)
    print("Ensemble Validation Script")
    print("=" * 70)
    print()

    print("Creating models...")
    psd1_model, psd2_model = create_psd_models(config)
    coh_model, lag_model = create_crossspectrum_models(config)
    print("  ✓ Models created")
    print()

    print(f"Running ensemble simulation (N={N_REALIZATIONS} realizations)...")
    ensemble_data = run_ensemble_simulation(
        psd1_model, psd2_model, coh_model, lag_model, config
    )
    print("  ✓ Ensemble simulation complete")
    print()

    print("Computing ensemble statistics...")
    stats = compute_ensemble_statistics(ensemble_data)
    print("  ✓ Statistics computed")
    print()

    print("Creating validation plot...")
    fig, axes = create_validation_plot(
        stats, psd1_model, psd2_model, coh_model, lag_model, config
    )
    print()

    print(f"Saving figure to {OUTPUT_FILE}...")
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print()
