"""
Interactive Bayesian Marketing Mix Modeling & Budget Allocation Executive Dashboard.
Redesigned with Option 4: Data Journalism & Storytelling UI/UX (Financial Times / The Pudding Editorial Style).
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
    page_title="Bayesian Econometrics // Financial Times Style MMM",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Option 4: Data Journalism & Storytelling Theme (FT Paper / Warm Cream Canvas)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,700;0,800;1,600&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    /* Warm Editorial Paper Canvas */
    .stApp {
        background-color: #FFFDF9;
        color: #1A1A1A;
    }
    
    /* Editorial Masthead */
    .editorial-masthead {
        border-bottom: 2px solid #1A1A1A;
        border-top: 1px solid #1A1A1A;
        padding: 16px 0;
        margin-bottom: 24px;
        text-align: center;
        background: #FFFDF9;
    }
    
    .editorial-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #0F172A;
        margin: 0;
        line-height: 1.2;
    }
    
    .editorial-subtitle {
        font-size: 14px;
        color: #64748B;
        font-style: italic;
        margin-top: 6px;
    }
    
    /* Editorial Narrative Callout Cards */
    .story-card {
        background: #FFFFFF;
        border: 1px solid #E5E0D8;
        border-left: 4px solid #C25E00;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    
    .story-card-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 18px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 8px;
    }
    
    /* Editorial Metric Box */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E5E0D8;
        padding: 16px;
        border-radius: 6px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Playfair Display', serif;
        color: #4A4A4A !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    
    [data-testid="stMetricValue"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #1A1A1A !important;
        font-weight: 800 !important;
    }
    
    /* Serif Headings */
    h1, h2, h3, h4 {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-weight: 700 !important;
        color: #1A1A1A !important;
    }
    
    /* Editorial Pill Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #F4EFEB;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #E5E0D8;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 15px;
        font-weight: 600;
        color: #5C5446;
        border-radius: 6px;
        padding: 8px 16px;
        background: transparent;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #C25E00 !important;
        font-weight: 700 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
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
            from src.model import BayesianMMM
            mmm = BayesianMMM.load_trace(trace_path)
            return mmm, True
        except Exception as e:
            st.sidebar.warning(f"Note: Running in demonstration mode ({e})")
            
    return None, False


def main():
    # Sidebar
    st.sidebar.markdown("### 🗞️ **The Econometrics Ledger**")
    st.sidebar.caption("Bayesian Decision Science & Econometric Modeling")
    st.sidebar.markdown("---")
    
    df = load_or_generate_dataset()
    channel_names = [col.replace("spend_", "") for col in df.columns if col.startswith("spend_")]
    mmm, is_fitted = load_fitted_model()
    
    if not is_fitted:
        st.sidebar.info("💡 **Demonstration Mode**: Calibrated synthetic posteriors loaded for rapid scenario exploration.")
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

    # Editorial Masthead
    st.markdown("""
    <div class="editorial-masthead">
        <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 2px; color: #8C7864; font-weight: 700;">
            THE FINANCIAL ECONOMETRICS BRIEFING &bull; VOLUME IV &bull; SPECIAL REPORT
        </div>
        <h1 class="editorial-title">Where Does Every Marketing Dollar Go?</h1>
        <div class="editorial-subtitle">
            An empirical investigation into adstock memory, diminishing returns, and optimal budget reallocation under Bayesian uncertainty.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 4 Key Financial Headline Metrics
    total_revenue = df["revenue"].sum()
    total_spend = sum(df[f"spend_{ch}"].sum() for ch in channel_names)
    overall_blended_roas = total_revenue / total_spend
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gross Attributable Revenue", f"${total_revenue:,.0f}")
    c2.metric("Cumulative Media Investment", f"${total_spend:,.0f}")
    c3.metric("Blended Program Efficiency", f"{overall_blended_roas:.2f}x ROAS")
    c4.metric("Active Media Channels Monitored", f"{len(channel_names)}")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Tabs (Narrative Chapters)
    tab1, tab2, tab3, tab4 = st.tabs([
        "Chapter I: The Historical Ledger",
        "Chapter II: The Physics of Advertising",
        "Chapter III: The Counterfactual Reallocator",
        "Appendix: Mathematical Appendix & Priors"
    ])

    # ==========================================
    # CHAPTER I: HISTORICAL LEDGER
    # ==========================================
    with tab1:
        st.markdown("""
        <div class="story-card">
            <div class="story-card-title">Disentangling Organic Demand from Paid Acquisition</div>
            Traditional attribution models (like Last-Click) credit advertising for sales that would have happened organically. 
            Using Bayesian time-series decomposition, we isolate organic demand, baseline trends, and holiday shocks 
            from the true incremental contribution of each paid marketing channel.
        </div>
        """, unsafe_allow_html=True)
        
        c_left, c_right = st.columns([3, 2])
        
        with c_left:
            st.markdown("#### Weekly Revenue Trajectory vs. Media Expenditure")
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=df["date"], y=df["revenue"], name="Actual Revenue", line=dict(color="#0F172A", width=2.5)))
            
            total_weekly_spend = df[[f"spend_{ch}" for ch in channel_names]].sum(axis=1)
            fig_ts.add_trace(go.Scatter(x=df["date"], y=total_weekly_spend, name="Media Investment", line=dict(color="#C25E00", width=1.5, dash="dot")))
            fig_ts.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=30, b=10),
                hovermode="x unified",
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_ts, use_container_width=True)
            
        with c_right:
            st.markdown("#### Program Revenue Share Breakdown")
            decomp_df = compute_waterfall_decomposition(df, posteriors, channel_max_spends, target_scale)
            fig_pie = px.pie(
                decomp_df, names="component", values="revenue", hole=0.5,
                color_discrete_sequence=["#D4A373", "#CCD5AE", "#E9EDC9", "#FAEDCD", "#DDA15E", "#BC6C25"]
            )
            fig_pie.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("#### Return on Ad Spend (ROAS) Credible Intervals (94% HDI)")
        st.caption("Channels to the right of the dashed break-even line (1.0x) generate net-positive incremental returns.")
        
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
                boxpoints=False,
                marker_color="#C25E00"
            ))
            
        fig_roas.add_hline(y=1.0, line_dash="dash", line_color="#DC2626", annotation_text="Break-even ROAS (1.0x)")
        fig_roas.update_layout(
            height=340,
            yaxis_title="ROAS ($ Revenue Generated per $ Spent)",
            showlegend=False,
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF"
        )
        st.plotly_chart(fig_roas, use_container_width=True)

    # ==========================================
    # CHAPTER II: THE PHYSICS OF ADVERTISING
    # ==========================================
    with tab2:
        st.markdown("""
        <div class="story-card">
            <div class="story-card-title">Memory & Saturation: Why More Spend Isn't Always Better</div>
            Advertising operates under two physical laws: <b>Carryover (Adstock)</b> dictates how long consumers remember your ad after exposure, 
            while <b>Saturation (Hill Function)</b> determines the ceiling where additional dollars produce diminishing returns.
        </div>
        """, unsafe_allow_html=True)
        
        selected_ch = st.selectbox(
            "Select Channel to Inspect:",
            options=channel_names,
            format_func=lambda x: x.replace("_", " ").title()
        )
        ch_idx = channel_names.index(selected_ch)
        
        c_phys1, c_phys2 = st.columns(2)
        
        with c_phys1:
            st.markdown("#### 1. Adstock Memory Decay Curve")
            alpha_val = float(posteriors["alpha"][:, ch_idx].mean())
            half_life = np.log(0.5) / np.log(max(alpha_val, 1e-4))
            
            st.info(f"**Retention Decay (α):** `{alpha_val:.2f}` &nbsp;|&nbsp; **Half-Life:** `{max(0, half_life):.1f} weeks`")
            
            weeks_lag = np.arange(10)
            decay_curve = alpha_val ** weeks_lag
            fig_decay = px.bar(
                x=[f"W+{w}" for w in weeks_lag],
                y=decay_curve,
                labels={"x": "Lag Horizon", "y": "Residual Impact"},
                color_discrete_sequence=["#DDA15E"]
            )
            fig_decay.update_layout(height=320, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig_decay, use_container_width=True)
            
        with c_phys2:
            st.markdown("#### 2. Diminishing Marginal Returns (Hill Saturation)")
            k_val = float(posteriors["K"][:, ch_idx].mean())
            s_val = float(posteriors["S"][:, ch_idx].mean())
            beta_val = float(posteriors["beta_media"][:, ch_idx].mean())
            max_ch_spend = float(channel_max_spends[ch_idx])
            
            spend_sweep = np.linspace(0, max_ch_spend * 1.6, 100)
            norm_sweep = spend_sweep / max_ch_spend
            sat_sweep = fast_hill_saturation_np(norm_sweep[:, None], np.array([k_val]), np.array([s_val]))
            rev_sweep = sat_sweep.ravel() * beta_val * target_scale
            current_avg_spend = float(df[f"spend_{selected_ch}"].mean())
            
            fig_sat = go.Figure()
            fig_sat.add_trace(go.Scatter(x=spend_sweep, y=rev_sweep, mode="lines", name="Expected Return", line=dict(color="#2D5A27", width=3)))
            fig_sat.add_vline(x=current_avg_spend, line_dash="dash", line_color="#C25E00", annotation_text=f"Current Avg (${current_avg_spend:,.0f})")
            fig_sat.update_layout(
                height=320,
                xaxis_title=f"Weekly Spend ($) on {selected_ch.replace('_', ' ').title()}",
                yaxis_title="Incremental Revenue ($)",
                showlegend=False,
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF"
            )
            st.plotly_chart(fig_sat, use_container_width=True)

    # ==========================================
    # CHAPTER III: THE COUNTERFACTUAL REALLOCATOR
    # ==========================================
    with tab3:
        st.markdown("""
        <div class="story-card">
            <div class="story-card-title">Prescriptive Decision Science: The Optimal Dollar Allocation</div>
            Using <b>Sequential Least Squares Programming (SLSQP)</b>, the model finds the mathematical global optimum 
            that maximizes revenue given channel budget constraints and downside risk tolerance (P10).
        </div>
        """, unsafe_allow_html=True)
        
        current_weekly_spend = sum(df[f"spend_{ch}"].mean() for ch in channel_names)
        
        c_sim_l, c_sim_r = st.columns([1, 2])
        
        with c_sim_l:
            st.markdown("#### Scenario Levers")
            target_weekly_budget = st.slider(
                "Total Target Weekly Media Budget ($):",
                min_value=int(current_weekly_spend * 0.4),
                max_value=int(current_weekly_spend * 2.0),
                value=int(current_weekly_spend),
                step=5000,
                format="$%d"
            )
            
            st.markdown("##### Individual Channel Guardrails")
            channel_bounds = {}
            for ch in channel_names:
                avg_sp = float(df[f"spend_{ch}"].mean())
                st.caption(f"**{ch.replace('_', ' ').title()}** (Historical: ${avg_sp:,.0f})")
                min_val = st.slider(f"Min for {ch}", 0, int(avg_sp), 0, step=1000, key=f"min_{ch}", label_visibility="collapsed")
                max_val = st.slider(f"Max for {ch}", int(avg_sp), int(avg_sp * 3), int(avg_sp * 1.8), step=1000, key=f"max_{ch}", label_visibility="collapsed")
                channel_bounds[ch] = (float(min_val), float(max_val))
                
            horizon = st.selectbox("Planning Horizon (Weeks):", [4, 8, 12, 26], index=2)
            run_btn = st.button("⚖️ Run Optimization Algorithm", type="primary", use_container_width=True)
            
        with c_sim_r:
            st.markdown("#### Optimization Dividend & Projected Outcome")
            optimizer = BudgetOptimizer(posteriors, channel_max_spends, target_scale)
            opt_result = optimizer.optimize_budget(target_weekly_budget, channel_bounds, horizon_weeks=horizon)
            
            st.success(f"**Projected Incremental Lift ({horizon} Weeks):** ${opt_result['projected_revenue_mean']:,.0f}")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Conservative Downside (P10)", f"${opt_result['p10_downside_revenue']:,.0f}")
            m2.metric("Median Expectation (P50)", f"${opt_result['p50_median_revenue']:,.0f}")
            m3.metric("Aggressive Upside (P90)", f"${opt_result['p90_upside_revenue']:,.0f}")
            
            alloc_df = opt_result["allocations"].copy()
            alloc_df["current_spend"] = [df[f"spend_{ch}"].mean() for ch in alloc_df["channel"]]
            
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Bar(
                x=[ch.replace("_", " ").title() for ch in alloc_df["channel"]],
                y=alloc_df["current_spend"],
                name="Historical Average Spend",
                marker_color="#CBD5E1"
            ))
            fig_compare.add_trace(go.Bar(
                x=[ch.replace("_", " ").title() for ch in alloc_df["channel"]],
                y=alloc_df["recommended_weekly_spend"],
                name="AI Optimized Allocation",
                marker_color="#C25E00"
            ))
            fig_compare.update_layout(
                barmode="group",
                height=340,
                yaxis_title="Weekly Spend ($)",
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_compare, use_container_width=True)
            
            st.dataframe(
                alloc_df.rename(columns={
                    "channel": "Channel",
                    "current_spend": "Current ($)",
                    "recommended_weekly_spend": "Recommended ($)",
                    "share_of_budget_pct": "Budget Share (%)"
                }),
                use_container_width=True,
                hide_index=True
            )

    # ==========================================
    # APPENDIX: MATHEMATICAL APPENDIX
    # ==========================================
    with tab4:
        st.subheader("Mathematical Methodology & Prior Formulations")
        st.markdown(r"""
        ### The Bayesian Data Generating Process
        The regression equation maps media expenditure to observed revenue via non-linear transforms:
        
        $$\text{Revenue}_t = \text{Intercept} + \text{Trend}_t + \text{FourierSeasonality}_t + \sum_{k=1}^K \beta_k \cdot \text{Hill}\big(\text{Adstock}(x_{k, t}, \alpha_k), K_k, S_k\big) + \epsilon_t$$
        
        1. **Geometric Adstock:** $\tilde{x}_{k, t} = x_{k, t} + \alpha_k \cdot \tilde{x}_{k, t-1}$, with prior $\alpha_k \sim \text{Beta}(2, 2)$.
        2. **Hill Saturation Function:** $\text{Hill}(\tilde{x}) = \frac{\tilde{x}^S}{K^S + \tilde{x}^S}$, parameterized by half-saturation $K_k \sim \text{Gamma}(2, 2)$ and slope $S_k \sim \text{Gamma}(3, 2)$.
        3. **Channel Effectiveness:** $\beta_k \sim \text{HalfNormal}(\sigma=0.8)$, ensuring media lift cannot be negative.
        4. **Constrained Optimization:** Solves non-convex budget reallocation using SciPy's Sequential Least Squares Programming (SLSQP).
        """)


if __name__ == "__main__":
    main()
