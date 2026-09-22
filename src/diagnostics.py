"""
Diagnostics & Posterior Evaluation Module.
Computes:
- Model convergence metrics (R-hat, effective sample size)
- Channel-level Return on Ad Spend (ROAS) posterior distributions & 94% HDI
- Waterfall contribution decomposition (Baseline vs Media vs Seasonality)
- Diminishing return thresholds
"""

import numpy as np
import pandas as pd
import arviz as az
from src.transforms import fast_geometric_adstock_np, fast_hill_saturation_np


def get_convergence_summary(idata) -> pd.DataFrame:
    """Returns R-hat and ESS summary for key parameters."""
    summary = az.summary(idata, var_names=["alpha", "K", "S", "beta_media", "intercept", "trend_slope", "sigma"])
    return summary


def extract_parameter_posteriors(idata, channel_names: list[str]) -> dict:
    """Extracts flattened posterior samples for all media channels."""
    posterior = idata.posterior
    n_samples = posterior.dims["chain"] * posterior.dims["draw"]
    
    alpha_samples = posterior["alpha"].values.reshape(n_samples, -1)
    k_samples = posterior["K"].values.reshape(n_samples, -1)
    s_samples = posterior["S"].values.reshape(n_samples, -1)
    beta_samples = posterior["beta_media"].values.reshape(n_samples, -1)
    
    return {
        "alpha": alpha_samples,
        "K": k_samples,
        "S": s_samples,
        "beta_media": beta_samples,
        "n_samples": n_samples,
        "channel_names": channel_names
    }


def compute_channel_roas(
    df: pd.DataFrame,
    posteriors: dict,
    channel_max_spends: np.ndarray,
    target_scale: float,
    n_eval_samples: int = 200
) -> pd.DataFrame:
    """
    Computes posterior distribution of ROAS (Incremental Revenue / Total Spend) for each channel.
    Returns mean, median, 94% HDI lower & upper bounds.
    """
    channel_names = posteriors["channel_names"]
    n_channels = len(channel_names)
    n_total_samples = posteriors["n_samples"]
    
    # Subsample for fast evaluation if large
    sample_indices = np.random.choice(n_total_samples, size=min(n_eval_samples, n_total_samples), replace=False)
    
    raw_spends = np.column_stack([df[f"spend_{ch}"].values for ch in channel_names])
    total_spends = raw_spends.sum(axis=0)
    norm_spends = raw_spends / channel_max_spends
    
    roas_records = []
    
    for k, ch_name in enumerate(channel_names):
        ch_spend_total = total_spends[k]
        ch_roas_draws = []
        
        for idx in sample_indices:
            alpha = posteriors["alpha"][idx, k]
            K = posteriors["K"][idx, k]
            S = posteriors["S"][idx, k]
            beta = posteriors["beta_media"][idx, k]
            
            # Adstock & Hill
            adstocked = fast_geometric_adstock_np(norm_spends[:, k:k+1], np.array([alpha]))
            saturated = fast_hill_saturation_np(adstocked, np.array([K]), np.array([S]))
            norm_contrib = beta * saturated.ravel()
            dollar_contrib = (norm_contrib * target_scale).sum()
            
            roas = dollar_contrib / max(ch_spend_total, 1.0)
            ch_roas_draws.append(roas)
            
        ch_roas_draws = np.array(ch_roas_draws)
        try:
            hdi = az.hdi(ch_roas_draws, prob=0.94)
        except TypeError:
            try:
                hdi = az.hdi(ch_roas_draws, hdi_prob=0.94)
            except Exception:
                hdi = [np.percentile(ch_roas_draws, 3), np.percentile(ch_roas_draws, 97)]
        except Exception:
            hdi = [np.percentile(ch_roas_draws, 3), np.percentile(ch_roas_draws, 97)]
        
        roas_records.append({
            "channel": ch_name,
            "mean_roas": np.mean(ch_roas_draws),
            "median_roas": np.median(ch_roas_draws),
            "hdi_3%": hdi[0],
            "hdi_97%": hdi[1],
            "total_spend": ch_spend_total,
            "estimated_revenue_generated": np.mean(ch_roas_draws) * ch_spend_total
        })
        
    return pd.DataFrame(roas_records)


def compute_waterfall_decomposition(
    df: pd.DataFrame,
    posteriors: dict,
    channel_max_spends: np.ndarray,
    target_scale: float
) -> pd.DataFrame:
    """Computes total business revenue split between Baseline, Seasonality, and Channels."""
    channel_names = posteriors["channel_names"]
    
    # Use posterior means for expected decomposition
    alpha_mean = posteriors["alpha"].mean(axis=0)
    k_mean = posteriors["K"].mean(axis=0)
    s_mean = posteriors["S"].mean(axis=0)
    beta_mean = posteriors["beta_media"].mean(axis=0)
    
    raw_spends = np.column_stack([df[f"spend_{ch}"].values for ch in channel_names])
    norm_spends = raw_spends / channel_max_spends
    
    channel_revenue = {}
    for k, ch in enumerate(channel_names):
        adstocked = fast_geometric_adstock_np(norm_spends[:, k:k+1], np.array([alpha_mean[k]]))
        saturated = fast_hill_saturation_np(adstocked, np.array([k_mean[k]]), np.array([s_mean[k]]))
        norm_contrib = beta_mean[k] * saturated.ravel()
        channel_revenue[ch] = float((norm_contrib * target_scale).sum())
        
    total_revenue = df["revenue"].sum()
    total_media = sum(channel_revenue.values())
    baseline_and_seasonality = max(0, total_revenue - total_media)
    
    decomp = [{"component": "Organic Baseline & Seasonality", "revenue": baseline_and_seasonality}]
    for ch, rev in channel_revenue.items():
        decomp.append({"component": ch.replace("_", " ").title(), "revenue": rev})
        
    return pd.DataFrame(decomp)
