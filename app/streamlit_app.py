"""
Interactive Bayesian Marketing Mix Modeling & Budget Allocation Executive Dashboard.
Built with Streamlit & Plotly.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.model import BayesianMMM
from src.diagnostics import (
    extract_parameter_posteriors,
    compute_channel_roas,
    compute_waterfall_decomposition
)
from src.optimizer import BudgetOptimizer
from src.transforms import fast_geometric_adstock_np, fast_hill_saturation_np
from data.generate_mock_data import generate_mmm_dataset


# Page setup
st.set_page_config(
    page_title="Bayesian Marketing Mix Modeling Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e2530;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2d3748;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_or_generate_dataset():
    data_path = PROJECT_ROOT / "data" / "marketing_mix_dataset.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        df, _ = generate_mmm_dataset()
        df.to_csv(data_path, index=False)
    return df


@st.cache_resource
def load_fitted_model():
    trace_path = PROJECT_ROOT / "data" / "mmm_posterior_trace.nc"
    if trace_path.exists():
        try:
            mmm = BayesianMMM.load_trace(trace_path)
            return mmm, True
        except Exception as e:
            st.sidebar.warning(f"Could not load pre-computed trace: {e}")
            
    # Fallback to dynamic simulation mode if user has not yet run train.py
    df = load_or_generate_dataset()
    channel_names = [col.replace("spend_", "") for col in df.columns if col.startswith("spend_")]
    mmm = BayesianMMM(channel_names=channel_names)
    _, _, norm_spends, _ = mmm._prepare_design_matrices(df)
    return mmm, False


def main():
    # Sidebar
    st.sidebar.title("📊 Bayesian MMM Engine")
    st.sidebar.markdown("**Decision-Support & Budget Optimization**")
    st.sidebar.markdown("---")
    
    df = load_or_generate_dataset()
    channel_names = [col.replace("spend_", "") for col in df.columns if col.startswith("spend_")]
    mmm, is_fitted = load_fitted_model()
    
    if not is_fitted:
        st.sidebar.info("💡 **Demonstration Mode**: Using synthetic ground-truth parameters. Run `python train.py` to train full MCMC posterior chains.")
        # Synthetic mock posteriors for instant app usability
        n_samples = 400
        posteriors = {
            "alpha": np.column_stack([np.random.beta(4, 16, n_samples), np.random.beta(6, 9, n_samples), 
                                      np.random.beta(8, 7, n_samples), np.random.beta(5, 12, n_samples), 
                                      np.random.beta(12, 5, n_samples)]),
            "K": np.column_stack([np.random.gamma(2, 0.5, n_samples) for _ in range(5)]),
            "S": np.column_stack([np.random.gamma(3, 0.4, n_samples) for _ in range(5)]),
            "beta_media": np.column_stack([np.random.normal(0.45, 0.05, n_samples) for _ in range(5)]),
            "n_samples": n_samples,
            "channel_names": channel_names
        }
        channel_max_spends = np.array([df[f"spend_{ch}"].max() for ch in channel_names])
        target_scale = df["revenue"].mean()
    else:
        st.sidebar.success("✅ **Fitted MCMC Posterior Trace Loaded**")
        posteriors = extract_parameter_posteriors(mmm.idata, channel_names)
        channel_max_spends = mmm.channel_max_spends
        target_scale = mmm.target_scale

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏛️ Executive Overview & ROAS",
        "📈 Adstock & Saturation Physics",
        "🎯 Counterfactual Budget Optimizer",
        "📑 Model Diagnostics & Methodology"
    ])

    # ==========================================
    # TAB 1: EXECUTIVE OVERVIEW & ROAS
    # ==========================================
    with tab1:
        st.subheader("Marketing Impact & Incrementality Overview")
        
        # High level KPIs
        total_revenue = df["revenue"].sum()
        total_spend = sum(df[f"spend_{ch}"].sum() for ch in channel_names)
        overall_blended_roas = total_revenue / total_spend
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Observed Revenue", f"${total_revenue:,.0f}")
        col2.metric("Total Marketing Spend", f"${total_spend:,.0f}")
        col3.metric("Blended Program ROAS", f"{overall_blended_roas:.2f}x")
        col4.metric("Active Media Channels", f"{len(channel_names)}")
        
        st.markdown("---")
        
        # Row 1: Time Series Fit + Waterfall
        c_left, c_right = st.columns([3, 2])
        
        with c_left:
            st.markdown("#### Historical Weekly Revenue vs. Marketing Spend")
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=df["date"], y=df["revenue"], name="Actual Revenue", line=dict(color="#00bcd4", width=2)))
            
            # Stacked spend area
            total_weekly_spend = df[[f"spend_{ch}" for ch in channel_names]].sum(axis=1)
            fig_ts.add_trace(go.Scatter(x=df["date"], y=total_weekly_spend, name="Total Media Spend", line=dict(color="#ff9800", width=1.5, dash="dot")))
            fig_ts.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_ts, use_container_width=True)
            
        with c_right:
            st.markdown("#### Revenue Decomposition (Waterfall)")
            decomp_df = compute_waterfall_decomposition(df, posteriors, channel_max_spends, target_scale)
            fig_waterfall = px.pie(decomp_df, names="component", values="revenue", hole=0.45, color_discrete_sequence=px.colors.qualitative.Safe)
            fig_waterfall.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_waterfall, use_container_width=True)

        # Row 2: Channel ROAS with Credible Intervals
        st.markdown("#### Channel ROAS (Return On Ad Spend) with 94% Bayesian Credible Intervals (HDI)")
        roas_df = compute_channel_roas(df, posteriors, channel_max_spends, target_scale)
        
        fig_roas = go.Figure()
        for _, row in roas_df.iterrows():
            fig_roas.add_trace(go.Box(
                y=[row["mean_roas"]],
                q1=[row["hdi_3%"]],
                median=[row["median_roas"]],
                q3=[row["hdi_97%"]],
                lowerfence=[row["hdi_3%"]],
                upperfence=[row["hdi_97%"]],
                name=row["channel"].replace("_", " ").title(),
                boxpoints=False
            ))
            
        fig_roas.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Break-even ROAS (1.0x)")
        fig_roas.update_layout(height=380, yaxis_title="ROAS ($ Revenue / $ Spend)", showlegend=False)
        st.plotly_chart(fig_roas, use_container_width=True)

    # ==========================================
    # TAB 2: ADSTOCK & SATURATION PHYSICS
    # ==========================================
    with tab2:
        st.subheader("Adstock Decay (Carryover) & Hill Saturation (Diminishing Returns)")
        st.markdown(
            "Understanding the **physics of your media**: Channels have different lag times (memory) "
            "and maximum revenue ceilings (saturation points)."
        )
        
        selected_ch = st.selectbox(
            "Select Channel to Inspect:",
            options=channel_names,
            format_func=lambda x: x.replace("_", " ").title()
        )
        ch_idx = channel_names.index(selected_ch)
        
        col_phys_l, col_phys_r = st.columns(2)
        
        with col_phys_l:
            st.markdown("#### 1. Adstock Carryover Decay Profile")
            alpha_val = float(posteriors["alpha"][:, ch_idx].mean())
            half_life = np.log(0.5) / np.log(max(alpha_val, 1e-4))
            
            st.info(f"**Estimated Decay Rate (α):** `{alpha_val:.2f}` | **Adstock Half-Life:** `{max(0, half_life):.1f} weeks`")
            
            weeks_lag = np.arange(10)
            decay_curve = alpha_val ** weeks_lag
            fig_decay = px.bar(
                x=[f"Week +{w}" for w in weeks_lag],
                y=decay_curve,
                labels={"x": "Lag Horizon", "y": "Residual Impact Weight"},
                color_discrete_sequence=["#3f51b5"]
            )
            fig_decay.update_layout(height=340)
            st.plotly_chart(fig_decay, use_container_width=True)
            
        with col_phys_r:
            st.markdown("#### 2. Hill Saturation & Diminishing Returns")
            k_val = float(posteriors["K"][:, ch_idx].mean())
            s_val = float(posteriors["S"][:, ch_idx].mean())
            beta_val = float(posteriors["beta_media"][:, ch_idx].mean())
            max_ch_spend = float(channel_max_spends[ch_idx])
            
            # Generate curve over spend range
            spend_sweep = np.linspace(0, max_ch_spend * 1.6, 100)
            norm_sweep = spend_sweep / max_ch_spend
            sat_sweep = fast_hill_saturation_np(norm_sweep[:, None], np.array([k_val]), np.array([s_val]))
            rev_sweep = sat_sweep.ravel() * beta_val * target_scale
            
            current_avg_spend = float(df[f"spend_{selected_ch}"].mean())
            
            fig_sat = go.Figure()
            fig_sat.add_trace(go.Scatter(x=spend_sweep, y=rev_sweep, mode="lines", name="Revenue Curve", line=dict(color="#4caf50", width=3)))
            fig_sat.add_vline(x=current_avg_spend, line_dash="dash", line_color="#ff5722", annotation_text=f"Current Avg Spend (${current_avg_spend:,.0f})")
            fig_sat.update_layout(
                height=340,
                xaxis_title=f"Weekly Spend ($) on {selected_ch.replace('_', ' ').title()}",
                yaxis_title="Estimated Incremental Revenue ($)",
                showlegend=False
            )
            st.plotly_chart(fig_sat, use_container_width=True)

    # ==========================================
    # TAB 3: COUNTERFACTUAL BUDGET OPTIMIZER
    # ==========================================
    with tab3:
        st.subheader("🎯 Prescriptive Budget Allocation & What-If Simulator")
        st.markdown(
            "Use Bayesian non-linear constrained optimization (SciPy SLSQP) to determine the mathematically "
            "optimal weekly marketing budget split to maximize incremental revenue."
        )
        
        current_weekly_spend = sum(df[f"spend_{ch}"].mean() for ch in channel_names)
        
        col_opt_ctrl, col_opt_res = st.columns([1, 2])
        
        with col_opt_ctrl:
            st.markdown("#### Scenario Constraints")
            target_weekly_budget = st.slider(
                "Total Target Weekly Media Budget ($):",
                min_value=int(current_weekly_spend * 0.4),
                max_value=int(current_weekly_spend * 2.0),
                value=int(current_weekly_spend),
                step=5000,
                format="$%d"
            )
            
            st.markdown("##### Per-Channel Spend Limits")
            channel_bounds = {}
            for ch in channel_names:
                avg_sp = float(df[f"spend_{ch}"].mean())
                st.caption(f"**{ch.replace('_', ' ').title()}** (Historical Avg: ${avg_sp:,.0f})")
                min_val = st.slider(f"Min for {ch}", 0, int(avg_sp), 0, step=1000, key=f"min_{ch}", label_visibility="collapsed")
                max_val = st.slider(f"Max for {ch}", int(avg_sp), int(avg_sp * 3), int(avg_sp * 1.8), step=1000, key=f"max_{ch}", label_visibility="collapsed")
                channel_bounds[ch] = (float(min_val), float(max_val))
                
            horizon = st.selectbox("Planning Horizon (Weeks):", [4, 8, 12, 26], index=2)
            run_btn = st.button("🚀 Reallocate & Maximize Revenue", type="primary", use_container_width=True)
            
        with col_opt_res:
            st.markdown("#### Optimization Results & Revenue Projection")
            optimizer = BudgetOptimizer(posteriors, channel_max_spends, target_scale)
            
            # Run initial optimization
            opt_result = optimizer.optimize_budget(target_weekly_budget, channel_bounds, horizon_weeks=horizon)
            
            # Comparison metrics
            st.success(f"**Projected Incremental Revenue ({horizon} Weeks):** ${opt_result['projected_revenue_mean']:,.0f}")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Downside Risk (P10)", f"${opt_result['p10_downside_revenue']:,.0f}")
            m2.metric("Median Expected (P50)", f"${opt_result['p50_median_revenue']:,.0f}")
            m3.metric("Upside Potential (P90)", f"${opt_result['p90_upside_revenue']:,.0f}")
            
            # Allocation Comparison Bar Chart
            alloc_df = opt_result["allocations"].copy()
            alloc_df["current_spend"] = [df[f"spend_{ch}"].mean() for ch in alloc_df["channel"]]
            
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Bar(
                x=[ch.replace("_", " ").title() for ch in alloc_df["channel"]],
                y=alloc_df["current_spend"],
                name="Historical Average Spend",
                marker_color="#90a4ae"
            ))
            fig_compare.add_trace(go.Bar(
                x=[ch.replace("_", " ").title() for ch in alloc_df["channel"]],
                y=alloc_df["recommended_weekly_spend"],
                name="AI Recommended Spend",
                marker_color="#00e676"
            ))
            fig_compare.update_layout(
                barmode="group",
                height=360,
                yaxis_title="Weekly Spend ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # Allocation table
            st.markdown("##### Detailed Allocation Breakdown")
            st.dataframe(
                alloc_df.rename(columns={
                    "channel": "Channel",
                    "current_spend": "Current Weekly ($)",
                    "recommended_weekly_spend": "Recommended Weekly ($)",
                    "share_of_budget_pct": "Budget Share (%)"
                }),
                use_container_width=True,
                hide_index=True
            )

    # ==========================================
    # TAB 4: METHODOLOGY & MODEL DIAGNOSTICS
    # ==========================================
    with tab4:
        st.subheader("Methodology, Priors & MCMC Convergence")
        st.markdown(r"""
        ### 📐 The Mathematical Formulation
        Marketing Mix Modeling is formulated as a hierarchical non-linear Bayesian regression:
        
        $$\text{Revenue}_t = \text{Baseline}(t) + \text{Seasonality}(t) + \sum_{k=1}^K \beta_k \cdot \text{Hill}\big(\text{Adstock}(x_{k, t}, \alpha_k), K_k, S_k\big) + \epsilon_t$$
        
        #### Key Distinguishing Capabilities:
        1. **Geometric Adstock Carryover:** Parameterized by $\alpha_k \sim \text{Beta}(2, 2)$, capturing weekly memory retention.
        2. **Hill Saturation Function:** Parameterized by half-saturation $K_k \sim \text{Gamma}(2, 2)$ and shape $S_k \sim \text{Gamma}(3, 2)$, enforcing diminishing marginal returns.
        3. **Bayesian Priors on Channel Effectiveness:** Channel coefficients $\beta_k \sim \text{HalfNormal}(\sigma=0.8)$, mathematically guaranteeing non-negative marketing contributions.
        4. **Constrained Optimization:** Solves non-convex budget reallocation using SciPy's Sequential Least Squares Programming (SLSQP).
        """)
        
        if is_fitted:
            st.markdown("### Posterior Summary Table (R-hat & ESS)")
            conv_summary = az.summary(mmm.idata, var_names=["alpha", "K", "S", "beta_media", "intercept", "trend_slope"])
            st.dataframe(conv_summary[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat", "ess_bulk"]], use_container_width=True)


if __name__ == "__main__":
    main()
