# Bayesian Marketing Mix Modeling (MMM) & Scenario Optimizer

A production-grade, end-to-end Bayesian Marketing Mix Modeling engine and interactive executive decision-support web application. Designed to model marketing carryover (**Adstock decay**), diminishing returns (**Hill saturation**), organic baseline demand, and perform **constrained budget reallocation optimization** with Bayesian credible risk intervals.

---

## 🌟 Key Features

1. **Differentiable Media Transformations:**
   * **Geometric Adstock Carryover:** Models audience memory retention ($\alpha_k \sim \text{Beta}(2, 2)$).
   * **Hill Saturation Functions:** Models diminishing marginal returns and identifies channel spend saturation ceilings ($K_k \sim \text{Gamma}(2, 2)$, $S_k \sim \text{Gamma}(3, 2)$).
2. **Bayesian Probabilistic Inference (PyMC & ArviZ):**
   * Uses NUTS (No-U-Turn Sampler) across 4 chains to infer parameter posteriors.
   * Produces **94% High Density Credible Intervals (HDI)** for channel-specific ROAS.
   * Isolates organic baseline, long-term trends, Fourier seasonality, and holiday shocks.
3. **Prescriptive Budget Reallocation (SciPy SLSQP):**
   * Formulates a non-linear constrained optimization problem to solve:
     $$\max_{\{s_1, \dots, s_K\}} \sum_{k=1}^K \mathbb{E}[\text{Incremental Revenue}(s_k)] \quad \text{s.t.} \quad \sum s_k \le B_{\text{total}}, \quad L_k \le s_k \le U_k$$
   * Quantifies downside risk (P10) and upside potential (P90) using MCMC draws.
4. **Interactive Executive Web Application (Streamlit + Plotly):**
   * **Executive Overview:** High-level KPIs, actual vs. fitted revenue, and waterfall decomposition.
   * **Media Physics Explorer:** Visualizes half-life decay curves and Hill saturation points.
   * **Scenario Simulator & Budget Reallocator:** Interactive sliders for total budget and channel limits with real-time AI reallocation recommendations.

---

## 🏗️ Architecture

```
bayesian_mmm_engine/
├── data/
│   ├── generate_mock_data.py       # Simulates 3 years of multi-channel retail marketing data
│   └── marketing_mix_dataset.csv   # Synthesized dataset
├── src/
│   ├── __init__.py
│   ├── transforms.py               # Differentiable PyTensor and fast NumPy Adstock/Hill functions
│   ├── model.py                    # PyMC Bayesian MMM model specification, sampling, and export
│   ├── diagnostics.py              # Convergence checks, ROAS distributions, waterfall calculations
│   └── optimizer.py                # SciPy SLSQP constrained budget reallocation solver
├── app/
│   └── streamlit_app.py            # Interactive 4-tab Streamlit dashboard
├── train.py                        # Full model training and trace export pipeline
├── requirements.txt                # Project dependencies
└── README.md                       # Documentation
```

---

## 🚀 Quickstart Guide (Local Setup & Run)

### 1. Prerequisites
Make sure you have **Python 3.10+** installed.

### 2. Create Virtual Environment & Install Dependencies
Open PowerShell or your terminal, navigate to the project directory:

```powershell
cd C:\Users\soumili\Desktop\projects\bayesian_mmm_engine
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Generate Data (Optional - Auto-generated if missing)
```powershell
python data/generate_mock_data.py
```

### 4. Train the Full Bayesian MCMC Model (Takes ~2-3 minutes)
```powershell
python train.py
```
*This runs NUTS sampling over 4 chains and saves the pre-computed trace to `data/mmm_posterior_trace.nc`.*

### 5. Launch the Interactive Dashboard
```powershell
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501` to use the dashboard!

> **Note:** Even if you haven't run `train.py` yet, the Streamlit app contains an instant demonstration mode with simulated posterior distributions so you can immediately explore the UI and optimization features!

---

## 🌐 Deployment Options

### Deploy to Streamlit Community Cloud (Free)
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Select your repository, set the entry point to `app/streamlit_app.py`, and click **Deploy**.

### Deploy with Docker
Create a simple `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
Build and run:
```bash
docker build -t bayesian-mmm .
docker run -p 8501:8501 bayesian-mmm
```
