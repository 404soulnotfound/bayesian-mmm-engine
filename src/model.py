"""
Hierarchical Bayesian Marketing Mix Model Implementation using PyMC.
Formulates:
- Trend + Fourier Seasonality + Baseline Intercept
- Per-channel Geometric Adstock (Beta priors on decay rates)
- Per-channel Hill Saturation (Gamma priors on half-saturation K and slope S)
- Half-Normal priors on channel effectiveness (Beta)
- Posterior Predictive Sampling
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from pathlib import Path

from src.transforms import geometric_adstock_tensor, hill_saturation_tensor


class BayesianMMM:
    def __init__(self, channel_names: list[str]):
        self.channel_names = channel_names
        self.n_channels = len(channel_names)
        self.model = None
        self.idata = None
        self.channel_max_spends = None
        self.target_scale = None

    def _prepare_design_matrices(self, df: pd.DataFrame):
        n_obs = len(df)
        t = np.arange(n_obs) / n_obs  # normalized time for trend [0, 1]
        
        # Fourier terms for annual seasonality (frequency = 52.14 weeks)
        weeks = np.arange(n_obs)
        sin_term = np.sin(2 * np.pi * weeks / 52.14)
        cos_term = np.cos(2 * np.pi * weeks / 52.14)
        sin_semi = np.sin(4 * np.pi * weeks / 52.14)
        cos_semi = np.cos(4 * np.pi * weeks / 52.14)
        
        fourier_matrix = np.column_stack([sin_term, cos_term, sin_semi, cos_semi])
        
        # Media spends normalized by max spend to stabilize priors
        spend_matrix = np.column_stack([df[f"spend_{ch}"].values for ch in self.channel_names])
        self.channel_max_spends = np.maximum(spend_matrix.max(axis=0), 1.0)
        norm_spend_matrix = spend_matrix / self.channel_max_spends
        
        # Target revenue scaled by mean
        revenue = df["revenue"].values
        self.target_scale = revenue.mean()
        norm_revenue = revenue / self.target_scale
        
        return t, fourier_matrix, norm_spend_matrix, norm_revenue

    def build_model(self, df: pd.DataFrame):
        t, fourier, norm_spends, norm_rev = self._prepare_design_matrices(df)
        n_obs = len(df)
        
        with pm.Model() as model:
            # 1. Baseline & Trend Priors
            intercept = pm.Normal("intercept", mu=1.0, sigma=0.3)
            trend_slope = pm.Normal("trend_slope", mu=0.1, sigma=0.2)
            baseline = intercept + trend_slope * t
            
            # 2. Seasonality Priors (Fourier weights)
            beta_fourier = pm.Normal("beta_fourier", mu=0.0, sigma=0.15, shape=4)
            seasonality = pt.dot(fourier, beta_fourier)
            
            # 3. Media Channel Parameters
            # Decay alpha in [0, 1]
            alpha = pm.Beta("alpha", alpha=2.0, beta=2.0, shape=self.n_channels)
            
            # Saturation parameters (normalized space)
            K = pm.Gamma("K", alpha=2.0, beta=2.0, shape=self.n_channels)
            S = pm.Gamma("S", alpha=3.0, beta=2.0, shape=self.n_channels)
            
            # Effectiveness coefficients (must be positive -> HalfNormal)
            beta_media = pm.HalfNormal("beta_media", sigma=0.8, shape=self.n_channels)
            
            # Calculate media effects channel by channel
            media_components = []
            for k in range(self.n_channels):
                # Apply adstock over time
                ch_spend = norm_spends[:, k]
                adstocked = geometric_adstock_tensor(ch_spend, alpha[k])
                saturated = hill_saturation_tensor(adstocked, K[k], S[k])
                media_components.append(beta_media[k] * saturated)
            
            media_contrib_matrix = pt.stack(media_components, axis=1)  # (T, K)
            total_media_effect = pt.sum(media_contrib_matrix, axis=1)
            
            # 4. Total Expected Revenue
            mu = pm.Deterministic("mu", baseline + seasonality + total_media_effect)
            
            # Observation noise
            sigma = pm.Exponential("sigma", lam=10.0)
            
            # Likelihood
            pm.Normal("likelihood", mu=mu, sigma=sigma, observed=norm_rev)
            
            self.model = model
            return model

    def fit(self, df: pd.DataFrame, draws: int = 1000, tune: int = 800, target_accept: float = 0.92, random_seed: int = 42):
        if self.model is None:
            self.build_model(df)
            
        with self.model:
            print("[*] Sampling posterior distributions via NUTS...")
            self.idata = pm.sample(
                draws=draws,
                tune=tune,
                target_accept=target_accept,
                random_seed=random_seed,
                return_inferencedata=True,
                progressbar=True
            )
            print("[*] Sampling posterior predictive checks...")
            pm.sample_posterior_predictive(self.idata, extend_inferencedata=True, random_seed=random_seed)
            
        return self.idata

    def save_trace(self, filepath: str | Path):
        if self.idata is None:
            raise ValueError("Model has not been fitted yet.")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # Store metadata attributes in inferencedata
        self.idata.attrs["channel_names"] = self.channel_names
        self.idata.attrs["channel_max_spends"] = list(self.channel_max_spends)
        self.idata.attrs["target_scale"] = float(self.target_scale)
        az.to_netcdf(self.idata, str(filepath))
        print(f"[+] Saved posterior trace and metadata to: {filepath}")

    @classmethod
    def load_trace(cls, filepath: str | Path):
        filepath = Path(filepath)
        idata = az.from_netcdf(str(filepath))
        channel_names = idata.attrs["channel_names"]
        instance = cls(channel_names=channel_names)
        instance.idata = idata
        instance.channel_max_spends = np.array(idata.attrs["channel_max_spends"])
        instance.target_scale = float(idata.attrs["target_scale"])
        return instance
