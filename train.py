"""
End-to-End Model Training and Trace Export Script.
Loads/generates data, builds the PyMC model, samples MCMC posterior chains,
and exports the pre-computed trace for instantaneous interactive web app loading.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from data.generate_mock_data import generate_mmm_dataset
from src.model import BayesianMMM
from src.diagnostics import get_convergence_summary, extract_parameter_posteriors, compute_channel_roas


def run_training_pipeline():
    data_dir = PROJECT_ROOT / "data"
    data_file = data_dir / "marketing_mix_dataset.csv"
    trace_file = PROJECT_ROOT / "data" / "mmm_posterior_trace.nc"
    
    # 1. Dataset Generation / Loading
    if not data_file.exists():
        print("[*] Generating synthetic multi-channel retail dataset...")
        df, _ = generate_mmm_dataset()
        df.to_csv(data_file, index=False)
    else:
        print(f"[*] Loading existing dataset from {data_file}...")
        df = pd.read_csv(data_file)
        
    channel_names = [col.replace("spend_", "") for col in df.columns if col.startswith("spend_")]
    print(f"[*] Identified {len(channel_names)} media channels: {channel_names}")
    
    # 2. Bayesian Model Formulation & NUTS Sampling
    mmm = BayesianMMM(channel_names=channel_names)
    print("[*] Fitting Bayesian MMM (PyMC NUTS sampler)...")
    # 800 draws & 600 tuning per chain provides great posterior resolution with quick convergence
    idata = mmm.fit(df, draws=800, tune=600, target_accept=0.92, random_seed=42)
    
    # 3. Save Pre-computed Posterior Trace
    mmm.save_trace(trace_file)
    
    # 4. Print Diagnostics Summary
    print("\n" + "="*50)
    print("BAYESIAN MMM POSTERIOR DIAGNOSTICS")
    print("="*50)
    conv = get_convergence_summary(idata)
    print(conv[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat", "ess_bulk"]].head(15))
    
    posteriors = extract_parameter_posteriors(idata, channel_names)
    roas_df = compute_channel_roas(df, posteriors, mmm.channel_max_spends, mmm.target_scale)
    print("\n" + "="*50)
    print("ESTIMATED CHANNEL ROAS (94% HDI)")
    print("="*50)
    print(roas_df[["channel", "mean_roas", "hdi_3%", "hdi_97%", "total_spend"]])
    print(f"\n[+] Pipeline completed successfully! Trace stored at: {trace_file}")


if __name__ == "__main__":
    run_training_pipeline()
