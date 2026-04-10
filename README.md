# Supply Chain Resilience — ML + Monte Carlo Framework

> Companion code for the book chapter:  
> **"A Machine Learning and Simulation Approach for Improving Supply Chain Resilience"**

---

## Overview

This repository implements the two-layer computational framework described in the chapter:

- **Layer 1 — Predictive ML Model:** A logistic regression pipeline that normalises heterogeneous real-world risk indicators into comparable exposure scores on a [0, 1] scale.
- **Layer 2 — Monte Carlo Simulation:** 5,000-iteration stochastic simulation that constructs an empirical distribution of system-level risk and evaluates mitigation scenarios.

The illustrative case applies the framework to **six risks** in the UK electricity supply network, using publicly available 2024–2026 proxy indicators.

---

---

## The Six Risks

| Code | Risk | Proxy Indicator | Source |
|------|------|----------------|--------|
| CC | Climate Change | UK mean temperature anomaly (2024) | Met Office, 2025 |
| ND | Natural Disasters | Properties at flood risk in England | Environment Agency, 2025 |
| AF | Affordability | Domestic energy debt & arrears (Q3 2025) | Ofgem, 2026 |
| ST | Sabotage & Terrorism | CNI organisations with a data breach (2024) | UK Government, 2026 |
| IA | Industrial Action | Working days lost to disputes (Nov 2025) | ONS, 2026 |
| PI | Political Instability | Net import dependency (2025) | DESNZ, 2026 |

---

## How to Run

### Option A — Run locally

```bash
# 1. Clone the repository
git clone https://github.com/rozaghaedi/supply-chain-resilience-ml.git
cd supply-chain-resilience-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the script
python supply_chain_resilience.py
```

### Option B — Run on Google Colab (no installation needed)

Click the badge below to open directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rozaghaedi/supply-chain-resilience-ml/blob/main/supply_chain_resilience.py)

---

## Outputs

Running the script produces:

1. **Console output** — normalised exposure scores, logistic regression probabilities, baseline risk scores, and Monte Carlo statistics
2. **`supply_chain_resilience.png`** — two-panel figure:
   - Figure 1: Baseline risk scores (horizontal bar chart)
   - Figure 2: Monte Carlo distributions for baseline and mitigation scenarios

---

## Key Results

| Metric | Value |
|--------|-------|
| Mean system risk (baseline) | 474.11 |
| Standard deviation | 15.72 |
| 95th percentile | 500.14 |
| P(system risk > 500) | 5.10% |
| Mean system risk (mitigation) | 445.55 |
| Mean reduction from mitigation | 6.02% |

---

## Dependencies

- Python 3.9+
- numpy, pandas, matplotlib, scikit-learn


