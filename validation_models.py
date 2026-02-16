"""
Simplified Q-Based Models for Validation Script

This module contains Q-based implementations of the simul_modeling models
needed for validation_ensemble_plot.py. These models are Q-based by default,
eliminating the need for parameter transformation.

Implements:
- psd_n_lorentz: N-component Lorentzian PSD model (A amplitudes)
- psd_n_lorentz_B: N-component Lorentzian PSD model (B amplitudes)
- coherence_n_lorentz: Frequency-dependent coherence model
- phase_diff_n_lorentz: Frequency-dependent phase lag model

All models use Q-based parameterization:
- m_i: center in frequency*power space (m = x_0 * sqrt(1 + 4*Q^2))
- Q_i: quality factor (Q = 2*x_0 / fwhm)
"""

from astropy.modeling import models, Fittable1DModel, Parameter
import numpy as np


# =============================================================================
# PSD MODELS
# =============================================================================

class psd_n_lorentz(Fittable1DModel):
    """
    N-component Lorentzian PSD model (Q-based parameterization).

    PSD(f) = noise + sum_i [A_i / (1 + ((f - x_0_i) / (fwhm_i/2))^2)]

    where x_0_i and fwhm_i are computed from m_i and Q_i:
        x_0 = m / sqrt(1 + 4*Q^2)
        fwhm = 2*x_0 / Q

    Parameters
    ----------
    n : int
        Number of Lorentzian components
    A_i : float
        Amplitude of component i
    m_i : float
        Center in frequency*power space for component i
    Q_i : float
        Quality factor of component i
    noise : float
        Poisson noise level
    """

    def __init__(self, n, *args, **kwargs):
        self.n = n

        # Create Q-based parameters for each component
        for i in range(n):
            A_param = Parameter(name=f"A_{i}", default=1.0)
            m_param = Parameter(name=f'm_{i}', default=1.0)
            Q_param = Parameter(name=f'Q_{i}', default=1.0)

            setattr(self, f"A_{i}", A_param)
            setattr(self, f"m_{i}", m_param)
            setattr(self, f"Q_{i}", Q_param)

        # Add noise parameter
        noise_param = Parameter(name="noise", default=0.0)
        setattr(self, 'noise', noise_param)

        super().__init__(*args, **kwargs)

    def as_QBased(self, inplace=False):
        """No-op - model is already Q-based."""
        pass

    def evaluate(self, x, *params, **kw_params):
        """Evaluate PSD at frequencies x."""
        if params:
            noise = params[-1]
        elif kw_params:
            noise_val = getattr(self, 'noise')
            noise_default = noise_val.value if hasattr(noise_val, 'value') else noise_val
            noise = kw_params.get('noise', noise_default)
        else:
            noise_val = getattr(self, 'noise')
            noise = noise_val.value if hasattr(noise_val, 'value') else noise_val

        out = np.full_like(x, noise, dtype=float)

        ngiven = len(params)
        for i in range(self.n):
            if ngiven > 3 * (i + 1):
                start_index = 3 * i
                A = params[start_index]
                m = params[start_index + 1]
                Q = params[start_index + 2]
            elif kw_params:
                A = kw_params[f'A_{i}']
                m = kw_params[f'm_{i}']
                Q = kw_params[f'Q_{i}']
            else:
                A_val = getattr(self, f'A_{i}')
                m_val = getattr(self, f'm_{i}')
                Q_val = getattr(self, f'Q_{i}')
                # Handle both Parameter objects and direct values
                A = A_val.value if hasattr(A_val, 'value') else A_val
                m = m_val.value if hasattr(m_val, 'value') else m_val
                Q = Q_val.value if hasattr(Q_val, 'value') else Q_val

            # Convert Q-based params to FWHM-based for Lorentzian evaluation
            x_0 = m / np.sqrt(1 + 4*Q**2)
            fwhm = 2*x_0 / Q

            out += models.Lorentz1D.evaluate(x, A, x_0, fwhm)

        return out


class psd_n_lorentz_B(Fittable1DModel):
    """
    N-component Lorentzian PSD model with B amplitudes (Q-based parameterization).

    Identical to psd_n_lorentz but uses B_i instead of A_i for amplitudes.

    Parameters
    ----------
    n : int
        Number of Lorentzian components
    B_i : float
        Amplitude of component i (second band)
    m_i : float
        Center in frequency*power space for component i
    Q_i : float
        Quality factor of component i
    noise : float
        Poisson noise level
    """

    def __init__(self, n, *args, **kwargs):
        self.n = n

        # Create Q-based parameters with B instead of A
        for i in range(n):
            B_param = Parameter(name=f"B_{i}", default=1.0)
            m_param = Parameter(name=f'm_{i}', default=1.0)
            Q_param = Parameter(name=f'Q_{i}', default=1.0)

            setattr(self, f"B_{i}", B_param)
            setattr(self, f"m_{i}", m_param)
            setattr(self, f"Q_{i}", Q_param)

        # Add noise parameter
        noise_param = Parameter(name="noise", default=0.0)
        setattr(self, 'noise', noise_param)

        super().__init__(*args, **kwargs)

    def as_QBased(self, inplace=False):
        """No-op - model is already Q-based."""
        pass

    def evaluate(self, x, *params, **kw_params):
        """Evaluate PSD at frequencies x."""
        if params:
            noise = params[-1]
        elif kw_params:
            noise_val = getattr(self, 'noise')
            noise_default = noise_val.value if hasattr(noise_val, 'value') else noise_val
            noise = kw_params.get('noise', noise_default)
        else:
            noise_val = getattr(self, 'noise')
            noise = noise_val.value if hasattr(noise_val, 'value') else noise_val

        out = np.full_like(x, noise, dtype=float)

        ngiven = len(params)
        for i in range(self.n):
            if ngiven > 3 * (i + 1):
                start_index = 3 * i
                B = params[start_index]
                m = params[start_index + 1]
                Q = params[start_index + 2]
            elif kw_params:
                B = kw_params[f'B_{i}']
                m = kw_params[f'm_{i}']
                Q = kw_params[f'Q_{i}']
            else:
                B_val = getattr(self, f'B_{i}')
                m_val = getattr(self, f'm_{i}')
                Q_val = getattr(self, f'Q_{i}')
                # Handle both Parameter objects and direct values
                B = B_val.value if hasattr(B_val, 'value') else B_val
                m = m_val.value if hasattr(m_val, 'value') else m_val
                Q = Q_val.value if hasattr(Q_val, 'value') else Q_val

            # Convert Q-based params to FWHM-based for Lorentzian evaluation
            x_0 = m / np.sqrt(1 + 4*Q**2)
            fwhm = 2*x_0 / Q

            out += models.Lorentz1D.evaluate(x, B, x_0, fwhm)

        return out


# =============================================================================
# CROSS-SPECTRUM MODELS
# =============================================================================

class phase_diff_n_lorentz(Fittable1DModel):
    """
    Frequency-dependent phase lag model (Q-based parameterization).

    Implements equation 9 of Méndez et al. 2024 (2024MNRAS.527.9405M):
    φ(f) = arctan(sum_i[C_i * L_i(f) * sin(lag_i)] / sum_i[C_i * L_i(f) * cos(lag_i)])

    where C_i = sqrt(A_i * B_i) and L_i(f) is the i-th Lorentzian.

    Parameters
    ----------
    n : int
        Number of Lorentzian components
    A_i : float
        Amplitude of component i in band 1
    B_i : float
        Amplitude of component i in band 2
    m_i : float
        Center in frequency*power space for component i
    Q_i : float
        Quality factor of component i
    lag_i : float
        Phase lag of component i (radians)
    """

    def __init__(self, n, *args, **kwargs):
        self.n = n

        # Create Q-based parameters for each component
        for i in range(n):
            A_param = Parameter(name=f"A_{i}", default=1.0)
            B_param = Parameter(name=f"B_{i}", default=1.0)
            m_param = Parameter(name=f'm_{i}', default=1.0)
            Q_param = Parameter(name=f'Q_{i}', default=1.0)
            lag_param = Parameter(name=f'lag_{i}', default=0.0)

            setattr(self, f"A_{i}", A_param)
            setattr(self, f"B_{i}", B_param)
            setattr(self, f"m_{i}", m_param)
            setattr(self, f"Q_{i}", Q_param)
            setattr(self, f'lag_{i}', lag_param)

        super().__init__(*args, **kwargs)

    def as_QBased(self, inplace=False):
        """No-op - model is already Q-based."""
        pass

    def evaluate(self, x, *params):
        """Evaluate phase lag at frequencies x."""
        num = 0
        den = 0

        ngiven = len(params)
        for i in range(self.n):
            if ngiven >= 5 * (i+1):
                start_index = 5*i
                A = params[start_index]
                B = params[start_index + 1]
                m = params[start_index + 2]
                Q = params[start_index + 3]
                lag = params[start_index + 4]
            else:
                A_val = getattr(self, f'A_{i}')
                B_val = getattr(self, f'B_{i}')
                m_val = getattr(self, f'm_{i}')
                Q_val = getattr(self, f'Q_{i}')
                lag_val = getattr(self, f'lag_{i}')
                # Handle both Parameter objects and direct values
                A = A_val.value if hasattr(A_val, 'value') else A_val
                B = B_val.value if hasattr(B_val, 'value') else B_val
                m = m_val.value if hasattr(m_val, 'value') else m_val
                Q = Q_val.value if hasattr(Q_val, 'value') else Q_val
                lag = lag_val.value if hasattr(lag_val, 'value') else lag_val

            # Convert Q-based params to FWHM-based for Lorentzian evaluation
            x_0 = m / np.sqrt(1 + 4*Q**2)
            fwhm = 2*x_0 / Q

            L_val = models.Lorentz1D.evaluate(x, amplitude=1.0, x_0=x_0, fwhm=fwhm)
            C = np.sqrt(A * B)

            num += C * L_val * np.sin(lag)
            den += C * L_val * np.cos(lag)

        return np.arctan(num / den)


class coherence_n_lorentz(Fittable1DModel):
    """
    Frequency-dependent coherence model (Q-based parameterization).

    Implements equation 10 of Méndez et al. 2024 (2024MNRAS.527.9405M):
    γ²(f) = [sum_i(C_i*L_i*cos(lag_i))]² + [sum_i(C_i*L_i*sin(lag_i))]²
            / [sum_i(A_i*L_i) * sum_i(B_i*L_i)]

    where C_i = sqrt(A_i * B_i) and L_i(f) is the i-th Lorentzian.

    Parameters
    ----------
    n : int
        Number of Lorentzian components
    A_i : float
        Amplitude of component i in band 1
    B_i : float
        Amplitude of component i in band 2
    m_i : float
        Center in frequency*power space for component i
    Q_i : float
        Quality factor of component i
    lag_i : float
        Phase lag of component i (radians)
    """

    def __init__(self, n, *args, **kwargs):
        self.n = n

        # Create Q-based parameters for each component
        for i in range(n):
            A_param = Parameter(name=f"A_{i}", default=1.0)
            B_param = Parameter(name=f"B_{i}", default=1.0)
            m_param = Parameter(name=f'm_{i}', default=1.0)
            Q_param = Parameter(name=f'Q_{i}', default=1.0)
            lag_param = Parameter(name=f'lag_{i}', default=0.0)

            setattr(self, f"A_{i}", A_param)
            setattr(self, f"B_{i}", B_param)
            setattr(self, f"m_{i}", m_param)
            setattr(self, f"Q_{i}", Q_param)
            setattr(self, f'lag_{i}', lag_param)

        super().__init__(*args, **kwargs)

    def as_QBased(self, inplace=False):
        """No-op - model is already Q-based."""
        pass

    def evaluate(self, x, *params):
        """Evaluate coherence at frequencies x."""
        num_1 = 0
        num_2 = 0
        den_1 = 0
        den_2 = 0

        ngiven = len(params)
        for i in range(self.n):
            if ngiven >= 5 * (i+1):
                start_index = 5*i
                A = params[start_index]
                B = params[start_index + 1]
                m = params[start_index + 2]
                Q = params[start_index + 3]
                lag = params[start_index + 4]
            else:
                A_val = getattr(self, f'A_{i}')
                B_val = getattr(self, f'B_{i}')
                m_val = getattr(self, f'm_{i}')
                Q_val = getattr(self, f'Q_{i}')
                lag_val = getattr(self, f'lag_{i}')
                # Handle both Parameter objects and direct values
                A = A_val.value if hasattr(A_val, 'value') else A_val
                B = B_val.value if hasattr(B_val, 'value') else B_val
                m = m_val.value if hasattr(m_val, 'value') else m_val
                Q = Q_val.value if hasattr(Q_val, 'value') else Q_val
                lag = lag_val.value if hasattr(lag_val, 'value') else lag_val

            # Convert Q-based params to FWHM-based for Lorentzian evaluation
            x_0 = m / np.sqrt(1 + 4*Q**2)
            fwhm = 2*x_0 / Q

            L_val = models.Lorentz1D.evaluate(x, amplitude=1.0, x_0=x_0, fwhm=fwhm)
            C = np.sqrt(A * B)

            num_1 += C * L_val * np.cos(lag)
            num_2 += C * L_val * np.sin(lag)
            den_1 += A * L_val
            den_2 += B * L_val

        num = num_1**2 + num_2**2
        den = den_1 * den_2

        return num / den
