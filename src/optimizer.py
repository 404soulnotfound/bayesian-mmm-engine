"""
Constrained Counterfactual Marketing Budget Optimizer.
Uses SciPy SLSQP to find the optimal allocation across channels that maximizes
expected incremental revenue, subject to:
1. Total Budget constraint: sum(spend_k) <= Target_Budget
2. Individual channel bounds: Min_Spend_k <= spend_k <= Max_Spend_k
Also computes downside uncertainty bands (10th, 50th, 90th percentiles) from MCMC draws.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.transforms import fast_geometric_adstock_np, fast_hill_saturation_np


class BudgetOptimizer:
    def __init__(self, posteriors: dict, channel_max_spends: np.ndarray, target_scale: float):
        self.posteriors = posteriors
        self.channel_names = posteriors["channel_names"]
        self.n_channels = len(self.channel_names)
        self.channel_max_spends = channel_max_spends
        self.target_scale = target_scale
        
        # Precompute posterior means for deterministic optimization objective
        self.alpha_mean = posteriors["alpha"].mean(axis=0)
        self.k_mean = posteriors["K"].mean(axis=0)
        self.s_mean = posteriors["S"].mean(axis=0)
        self.beta_mean = posteriors["beta_media"].mean(axis=0)

    def _expected_revenue_for_spend(self, spend_vector: np.ndarray, horizon_weeks: int = 12) -> float:
        """
        Simulates steady-state revenue over a forward planning horizon given a weekly spend vector.
        """
        spend_matrix = np.tile(spend_vector, (horizon_weeks, 1))  # (horizon, K)
        norm_spends = spend_matrix / self.channel_max_spends
        
        adstocked = fast_geometric_adstock_np(norm_spends, self.alpha_mean)
        saturated = fast_hill_saturation_np(adstocked, self.k_mean, self.s_mean)
        
        # Calculate weekly incremental contribution
        norm_contrib = saturated * self.beta_mean
        total_rev = (norm_contrib * self.target_scale).sum()
        return total_rev

    def optimize_budget(
        self,
        total_weekly_budget: float,
        channel_bounds: dict[str, tuple[float, float]],
        horizon_weeks: int = 12
    ) -> dict:
        """
        Solves constrained non-linear optimization:
        maximize: expected_revenue(spend_vector)
        subject to: sum(spend_vector) <= total_weekly_budget
                    L_k <= spend_k <= U_k
        """
        # Formulate bounds
        bounds = []
        initial_guess = []
        for ch in self.channel_names:
            min_b, max_b = channel_bounds.get(ch, (0.0, total_weekly_budget))
            bounds.append((min_b, max_b))
            initial_guess.append(max(min_b, min(total_weekly_budget / self.n_channels, max_b)))
            
        initial_guess = np.array(initial_guess)
        # Normalize initial guess so it sums to budget
        if initial_guess.sum() > 0:
            initial_guess = initial_guess * (total_weekly_budget / initial_guess.sum())

        # Objective function (minimizing negative expected revenue)
        def objective(x):
            return -self._expected_revenue_for_spend(x, horizon_weeks=horizon_weeks)

        # Budget constraint: total_weekly_budget - sum(x) >= 0
        constraints = [
            {"type": "ineq", "fun": lambda x: total_weekly_budget - np.sum(x)}
        ]

        res = minimize(
            fun=objective,
            x0=initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-7}
        )

        optimized_spends = np.maximum(0.0, res.x)
        expected_optimal_rev = -res.fun

        # Uncertainty quantification across posterior draws
        n_draws = min(150, self.posteriors["n_samples"])
        draw_indices = np.random.choice(self.posteriors["n_samples"], size=n_draws, replace=False)
        simulated_draw_revenues = []
        
        spend_matrix = np.tile(optimized_spends, (horizon_weeks, 1))
        norm_spends = spend_matrix / self.channel_max_spends

        for idx in draw_indices:
            alpha = self.posteriors["alpha"][idx]
            K = self.posteriors["K"][idx]
            S = self.posteriors["S"][idx]
            beta = self.posteriors["beta_media"][idx]
            
            adstocked = fast_geometric_adstock_np(norm_spends, alpha)
            saturated = fast_hill_saturation_np(adstocked, K, S)
            norm_contrib = saturated * beta
            draw_rev = (norm_contrib * self.target_scale).sum()
            simulated_draw_revenues.append(draw_rev)

        simulated_draw_revenues = np.array(simulated_draw_revenues)
        p10 = np.percentile(simulated_draw_revenues, 10)
        p50 = np.percentile(simulated_draw_revenues, 50)
        p90 = np.percentile(simulated_draw_revenues, 90)

        allocation_table = []
        for k, ch in enumerate(self.channel_names):
            allocation_table.append({
                "channel": ch,
                "recommended_weekly_spend": round(optimized_spends[k], 2),
                "share_of_budget_pct": round((optimized_spends[k] / total_weekly_budget) * 100, 2)
            })

        return {
            "success": res.success,
            "total_allocated_spend": float(np.sum(optimized_spends)),
            "allocations": pd.DataFrame(allocation_table),
            "projected_revenue_mean": float(expected_optimal_rev),
            "p10_downside_revenue": float(p10),
            "p50_median_revenue": float(p50),
            "p90_upside_revenue": float(p90),
            "horizon_weeks": horizon_weeks
        }
