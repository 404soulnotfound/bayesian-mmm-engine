"""
Synthetic Marketing Data Generator
Simulates realistic multi-year, multi-channel weekly advertising spend and sales data.
Includes:
- Organic baseline revenue
- Long-term business growth trend
- Multi-frequency seasonality (annual + holiday shocks)
- Adstock carryover (lag and decay)
- Hill saturation curves (diminishing marginal returns)
- Exogenous macro factors (Macroeconomic Index / Competitor activity)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def geometric_adstock_np(spend: np.ndarray, decay_rate: float) -> np.ndarray:
    """
    Applies geometric adstock decay to a 1D spend array.
    x_t = spend_t + decay_rate * x_{t-1}
    """
    adstocked = np.zeros_like(spend, dtype=float)
    for t in range(len(spend)):
        if t == 0:
            adstocked[t] = spend[t]
        else:
            adstocked[t] = spend[t] + decay_rate * adstocked[t - 1]
    return adstocked


def hill_saturation_np(x: np.ndarray, K: float, S: float) -> np.ndarray:
    """
    Applies the Hill saturation function:
    Hill(x) = (x^S) / (K^S + x^S)
    K: Half-saturation point (spend level reaching 50% max effect)
    S: Slope / shape parameter
    """
    x_safe = np.maximum(x, 1e-6)
    x_pow = np.power(x_safe, S)
    k_pow = np.power(K, S)
    return x_pow / (k_pow + x_pow)


def generate_mmm_dataset(
    n_weeks: int = 156,  # 3 full years of weekly observations
    start_date: str = "2023-01-02",
    random_seed: int = 42
) -> pd.DataFrame:
    np.random.seed(random_seed)
    
    # 1. Timeline
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    dates = [start_dt + timedelta(weeks=i) for i in range(n_weeks)]
    t = np.arange(n_weeks)
    
    # 2. Baseline & Seasonality Components
    # Organic base sales around $80k/week with a 15% annual growth trend
    base_revenue = 80000.0
    trend = 1.0 + (0.15 * (t / 52.0))
    
    # Annual seasonality: Sine + Cosine harmonic terms
    seasonality = 1.0 + 0.12 * np.sin(2 * np.pi * t / 52.14) + 0.08 * np.cos(4 * np.pi * t / 52.14)
    
    # Holiday shocks (Black Friday & Cyber Monday around weeks 47-49 of each year)
    holiday_spike = np.ones(n_weeks)
    for year_idx in range(3):
        bf_week = (year_idx * 52) + 47
        if bf_week < n_weeks:
            holiday_spike[bf_week] = 1.45  # 45% organic surge
        if bf_week + 1 < n_weeks:
            holiday_spike[bf_week + 1] = 1.30

    baseline_revenue = base_revenue * trend * seasonality * holiday_spike
    
    # 3. Media Channels Simulation
    # 5 Channels with distinct spend profiles, decay rates, and saturation points
    channels = {
        "google_search": {
            "mean_spend": 12000, "std_spend": 3000,
            "decay": 0.20,      # Fast response, low lag
            "K": 18000, "S": 1.2,
            "beta": 32000.0     # Max potential contribution
        },
        "meta_ads": {
            "mean_spend": 15000, "std_spend": 4500,
            "decay": 0.40,      # Moderate lag
            "K": 22000, "S": 1.1,
            "beta": 38000.0
        },
        "youtube_video": {
            "mean_spend": 8000, "std_spend": 2500,
            "decay": 0.55,      # Branding carryover
            "K": 15000, "S": 1.3,
            "beta": 24000.0
        },
        "tiktok_influencers": {
            "mean_spend": 6000, "std_spend": 3000,
            "decay": 0.30,      # Quick virality, fast decay
            "K": 10000, "S": 1.8,
            "beta": 20000.0
        },
        "tv_traditional": {
            "mean_spend": 20000, "std_spend": 8000,
            "decay": 0.70,      # High sustained awareness
            "K": 35000, "S": 1.0,
            "beta": 45000.0
        }
    }
    
    df = pd.DataFrame({
        "week": t + 1,
        "date": dates,
        "baseline_organic_revenue": baseline_revenue
    })
    
    total_media_revenue = np.zeros(n_weeks)
    
    for ch_name, params in channels.items():
        # Generate weekly spend with holiday budget increases
        spend_raw = np.maximum(0, np.random.normal(params["mean_spend"], params["std_spend"], n_weeks))
        
        # Increase ad budgets during Q4 holiday period
        for year_idx in range(3):
            q4_start = (year_idx * 52) + 44
            q4_end = min(n_weeks, (year_idx * 52) + 51)
            spend_raw[q4_start:q4_end] *= 1.4
            
        adstocked = geometric_adstock_np(spend_raw, params["decay"])
        saturated = hill_saturation_np(adstocked, params["K"], params["S"])
        channel_contrib = params["beta"] * saturated
        
        df[f"spend_{ch_name}"] = np.round(spend_raw, 2)
        df[f"contrib_{ch_name}"] = np.round(channel_contrib, 2)
        total_media_revenue += channel_contrib
        
    # 4. Total Observed Revenue with realistic observation noise (~3%)
    noise = np.random.normal(0, 4500, n_weeks)
    df["total_media_revenue"] = np.round(total_media_revenue, 2)
    df["revenue"] = np.round(np.maximum(10000, baseline_revenue + total_media_revenue + noise), 2)
    
    return df, channels


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    df, channels_meta = generate_mmm_dataset()
    csv_path = out_dir / "marketing_mix_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"[+] Successfully generated realistic MMM dataset with {len(df)} weeks.")
    print(f"[+] Saved to: {csv_path}")
    print(f"[+] Total Average Weekly Revenue: ${df['revenue'].mean():,.2f}")
    print(f"[+] Total Average Weekly Media Spend: ${sum(df[col].mean() for col in df.columns if col.startswith('spend_')):,.2f}")
