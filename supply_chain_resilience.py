
"""
Supply Chain Resilience: ML + Monte Carlo Simulation
Based on: Vafadarnikjoo et al. (2025) - UK Electricity Supply Network

Framework:
  Layer 1: Predictive ML model (Logistic Regression proxy)
  Layer 2: Monte Carlo Simulation (5,000 iterations)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)

# =============================================================================
# STEP 1: Define the six risks and their real-world data
# =============================================================================

risks = {
    "CC": {
        "full_name": "Climate Change",
        "raw_value": 0.64,          # °C above 1991-2020 average (Met Office, 2025)
        "raw_max": 1.0,             # normalisation ceiling
        "impact_multiplier": 1.60,
    },
    "ND": {
        "full_name": "Natural Disasters",
        "raw_value": 6.3,           # million properties at flood risk (Environment Agency, 2025)
        "raw_max": 10.0,
        "impact_multiplier": 1.45,
    },
    "AF": {
        "full_name": "Affordability",
        "raw_value": 4.48,          # £bn domestic energy debt Q3 2025 (Ofgem, 2026)
        "raw_max": 5.0,
        "impact_multiplier": 0.95,
    },
    "ST": {
        "full_name": "Sabotage & Terrorism",
        "raw_value": 95.0,          # % CNI orgs with data breach (UK Gov, 2026)
        "raw_max": 100.0,
        "impact_multiplier": 0.85,
    },
    "IA": {
        "full_name": "Industrial Action",
        "raw_value": 155_000,       # working days lost Nov 2025 (ONS, 2026)
        "raw_max": 200_000,
        "impact_multiplier": 0.75,
    },
    "PI": {
        "full_name": "Political Instability",
        "raw_value": 43.5,          # % net import dependency 2025 (DESNZ, 2026)
        "raw_max": 50.0,
        "impact_multiplier": 0.65,
    },
}

# =============================================================================
# STEP 2: ML Layer — Logistic Regression to estimate exposure scores
# =============================================================================
# In a full application you would train on historical longitudinal data.
# Here we demonstrate the normalisation pipeline (MinMaxScaler equivalent to
# the exposure scores reported in Table 1) and show how Logistic Regression
# could classify a risk as "high-exposure" (>=0.5) vs "low-exposure" (<0.5).

print("=" * 60)
print("LAYER 1 — PREDICTIVE ML MODEL")
print("=" * 60)

# --- 2a. Normalise raw values to [0, 1] exposure scores ---
risk_ids = list(risks.keys())
raw_values = np.array([risks[r]["raw_value"] for r in risk_ids]).reshape(-1, 1)
raw_maxes  = np.array([risks[r]["raw_max"]   for r in risk_ids]).reshape(-1, 1)

exposure_scores = (raw_values / raw_maxes).flatten()

for i, rid in enumerate(risk_ids):
    risks[rid]["exposure_score"] = round(float(exposure_scores[i]), 3)

print("\nNormalised exposure scores:")
for rid in risk_ids:
    print(f"  {rid} ({risks[rid]['full_name']:25s}): {risks[rid]['exposure_score']:.3f}")

# --- 2b. Logistic Regression demo ---
# We augment the 6 real observations with synthetic historical samples so the
# classifier has examples of BOTH classes (required by sklearn).
# Real features: [exposure_score, impact_multiplier]
X_real = np.array([
    [risks[r]["exposure_score"], risks[r]["impact_multiplier"]]
    for r in risk_ids
])

# Synthetic historical data: low-exposure past years (label=0)
# and high-exposure recent years (label=1)
np.random.seed(0)
X_synthetic_low  = np.column_stack([
    np.random.uniform(0.10, 0.45, 20),   # low exposure
    np.random.uniform(0.50, 1.60, 20),
])
X_synthetic_high = np.column_stack([
    np.random.uniform(0.55, 1.00, 20),   # high exposure
    np.random.uniform(0.50, 1.60, 20),
])

X_train = np.vstack([X_synthetic_low, X_synthetic_high])
y_train = np.array([0] * 20 + [1] * 20)

# Scale features (fit on training data, transform both)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_real_scaled  = scaler.transform(X_real)

# Train logistic regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Predicted probabilities for the 6 actual risks
proba = lr_model.predict_proba(X_real_scaled)[:, 1]

print("\nLogistic Regression — P(high-exposure) for each risk:")
for i, rid in enumerate(risk_ids):
    label = "HIGH" if proba[i] >= 0.5 else "LOW "
    print(f"  {rid}: P={proba[i]:.3f}  → {label}")

print(f"\n  Model coefficients : {lr_model.coef_[0]}")
print(f"  Model intercept    : {lr_model.intercept_[0]:.4f}")

# =============================================================================
# STEP 3: Compute baseline risk scores
# =============================================================================

print("\n" + "=" * 60)
print("BASELINE RISK SCORES  (100 × Exposure × Impact_multiplier)")
print("=" * 60)

for rid in risk_ids:
    r = risks[rid]
    r["base_risk_score"] = round(
        100 * r["exposure_score"] * r["impact_multiplier"], 2
    )

df_base = pd.DataFrame(
    {
        "Risk": [r for r in risk_ids],
        "Full Name": [risks[r]["full_name"] for r in risk_ids],
        "Exposure Score": [risks[r]["exposure_score"] for r in risk_ids],
        "Impact Multiplier": [risks[r]["impact_multiplier"] for r in risk_ids],
        "Base Risk Score": [risks[r]["base_risk_score"] for r in risk_ids],
    }
).sort_values("Base Risk Score", ascending=False).reset_index(drop=True)

print(df_base.to_string(index=False))
print(f"\n  Total system risk (baseline): {df_base['Base Risk Score'].sum():.2f}")

# =============================================================================
# STEP 4: Monte Carlo Simulation
# =============================================================================

N_ITER     = 5_000
STD_RATIO  = 0.08      # 8% of exposure score
SCORE_CAP  = 1.20      # upper cap
THRESHOLD  = 500       # high-risk threshold

print("\n" + "=" * 60)
print(f"LAYER 2 — MONTE CARLO SIMULATION  ({N_ITER:,} iterations)")
print("=" * 60)

def run_monte_carlo(risk_dict, risk_ids, n_iter, std_ratio, score_cap, label):
    """Run Monte Carlo and return array of system risk values."""
    system_risks = np.zeros(n_iter)
    for k in range(n_iter):
        total = 0.0
        for rid in risk_ids:
            r       = risk_dict[rid]
            mu      = r["exposure_score"]
            sigma   = std_ratio * mu
            exp_k   = np.clip(np.random.normal(mu, sigma), 0, score_cap)
            score_k = 100 * exp_k * r["impact_multiplier"]
            total  += score_k
        system_risks[k] = total
    print(f"\n  [{label}]")
    print(f"    Mean system risk     : {system_risks.mean():.2f}")
    print(f"    Std deviation        : {system_risks.std():.2f}")
    print(f"    95th percentile      : {np.percentile(system_risks, 95):.2f}")
    prob_high = np.mean(system_risks > THRESHOLD) * 100
    print(f"    P(system risk > {THRESHOLD}) : {prob_high:.2f}%")
    return system_risks

# --- 4a. Baseline ---
baseline_risks = run_monte_carlo(
    risks, risk_ids, N_ITER, STD_RATIO, SCORE_CAP, "BASELINE"
)

# --- 4b. Mitigation scenario ---
# CC −15%, ND −10%, AF −5%
mitigation_risks_dict = {rid: dict(r) for rid, r in risks.items()}
mitigation_risks_dict["CC"]["exposure_score"] *= (1 - 0.15)
mitigation_risks_dict["ND"]["exposure_score"] *= (1 - 0.10)
mitigation_risks_dict["AF"]["exposure_score"] *= (1 - 0.05)

mitigation_risks = run_monte_carlo(
    mitigation_risks_dict, risk_ids, N_ITER, STD_RATIO, SCORE_CAP, "MITIGATION SCENARIO"
)

mean_reduction = (
    (baseline_risks.mean() - mitigation_risks.mean()) / baseline_risks.mean() * 100
)
print(f"\n  Mean reduction from mitigation: {mean_reduction:.2f}%")

# =============================================================================
# STEP 5: Visualisations
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Supply Chain Resilience — UK Electricity Network\nML + Monte Carlo Framework",
    fontsize=14, fontweight="bold"
)

# --- Figure 1: Baseline risk scores (bar chart) ---
ax1 = axes[0]
colors_bar = ["#C0392B", "#E67E22", "#E74C3C", "#8E44AD", "#2980B9", "#27AE60"]
sorted_df = df_base.sort_values("Base Risk Score", ascending=True)
bars = ax1.barh(
    sorted_df["Risk"],
    sorted_df["Base Risk Score"],
    color=colors_bar,
    edgecolor="white",
    height=0.6,
)
ax1.set_xlabel("Base Risk Score", fontsize=11)
ax1.set_title("Figure 1 — Baseline Risk Scores", fontsize=12, fontweight="bold")
for bar, val in zip(bars, sorted_df["Base Risk Score"]):
    ax1.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
             f"{val:.1f}", va="center", fontsize=9)
ax1.set_xlim(0, 120)
ax1.grid(axis="x", alpha=0.3, linestyle="--")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# --- Figure 2: Monte Carlo distributions ---
ax2 = axes[1]
ax2.hist(baseline_risks, bins=60, alpha=0.65, color="#2980B9",
         label=f"Baseline  (μ={baseline_risks.mean():.1f})", density=True)
ax2.hist(mitigation_risks, bins=60, alpha=0.65, color="#27AE60",
         label=f"Mitigation  (μ={mitigation_risks.mean():.1f})", density=True)
ax2.axvline(THRESHOLD, color="#C0392B", linestyle="--", linewidth=1.5,
            label=f"Threshold = {THRESHOLD}")
ax2.axvline(np.percentile(baseline_risks, 95), color="#2980B9",
            linestyle=":", linewidth=1.2, label="95th pct (baseline)")
ax2.set_xlabel("System Risk", fontsize=11)
ax2.set_ylabel("Density", fontsize=11)
ax2.set_title("Figure 2 — Monte Carlo Distributions", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/supply_chain_resilience.png", dpi=150, bbox_inches="tight")
print("\n  Figures saved → supply_chain_resilience.png")
plt.show()

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
