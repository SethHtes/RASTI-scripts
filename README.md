# RASTI Scripts

Analysis scripts and notebooks for coherence simulations and estimator validation presented in **Larner, Nowak, & Wilms (2026)**.

## Overview

This repository implements the correlated time series simulation method based on Timmer & König (1995) with a transfer function approach for creating coherent and incoherent spectral components in X-ray timing analysis.

## Contents

### Python Scripts

- **`coherence_simulation.py`** - Main Monte Carlo simulation engine for correlated time series. Generates corner plots showing joint distributions of coherence, phase lag, power spectra, and cross-spectrum components with theoretical covariance predictions. (Figure 2)

- **`coherence_estimator_bias.py`** - Validates magnitude-squared coherence (MSC) estimator bias using Monte Carlo verification of the Nuttall & Carter (1976) formula. (Figure 3)

- **`validation_ensemble_plot.py`** - Ensemble validation using Q-based PSD models with Lorentzian components. (Figure 1)

### Mathematica Notebooks

- `coherence_sim_generalized_transfer_function.nb` - Transfer function formalism
- `coh_lag_var_montecarlo_simulations.nb` - Variance calculations
- `coherence_variance_different_P_cleaned.nb` - General power spectral cases

## Quick Start

```bash
# Run main simulation (generates corner plots)
python coherence_simulation.py

# Run bias validation
python coherence_estimator_bias.py
```

## Dependencies

Core requirements:
```bash
pip install numpy matplotlib scipy mpmath corner
```

For validation script:
```bash
pip install stingray
```

The validation script also requires:
- `synthetic-timeseries` package (see link in paper)



## Citation

If you use these scripts, please cite:

Larner, S., Nowak, M. A., & Wilms, J. (2026). [Paper title]. [Journal details].

## Key References

- Timmer, J., & König, M. (1995). On generating power law noise. A&A, 300, 707
- Nuttall, A. H., & Carter, G. C. (1976). Spectral estimation using combined time and lag weighting. IEEE Proc., 64(8), 1121-1125
