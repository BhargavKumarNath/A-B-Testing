# Criteo Uplift: Causal AI for Algorithmic Profit Optimization
![alt text](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)
![alt text](https://img.shields.io/badge/python-3.10-blue.svg)
![alt text](https://img.shields.io/badge/Polars-Fast_Data-orange)
![alt text](https://img.shields.io/badge/License-MIT-yellow.svg)
📉

# 📉 Executive Summary
**The Problem:** Traditional A/B testing revealed that while board advertising targeting increased conversion rates by $+56\%$, the high cost of media resulted in a **net loss of %0.05 per user**. The campaign was technically effective but economically disastrous.

**The Solution:** We architected an end-to-end Casual Inference pipeline to transition from descriptive analytics ("What happened?") to prescriptive intervention ("Who should we target?"). Using **X-Learner** architectures and **Contextual Bandits**, we isolated the sub-population of "Persuadable" users.

The Impact:
- **Unit Economics:** Turned negative margins into a positive outcome, reaching **0.09 profit per user**, representing a $0.14$ improvement from the previous baseline.

- **Efficiency:** Reduced ad spend volume by $80\%$ while retaining $104\%$ of the net profit.

- **Production:** Distilled heavy meta-learners into a sub-millisecond Decision Tree for real-time bidding (RTB).

[👉 View the Live Command Center](https://ablytics.streamlit.app/)

---
# 🏗 System Architecture
The pipeline is designed for scale(14M+ rows), utilising Polars for memory-efficient ingestion and **LightGBM/XGBoost** for gradient-boosted causal estimation.

![System Design](system_design_img.png)

---
# 🔬 Methodology & Technical Deep Dive
## 1. Experiment Integrity (The Foundation)
Before modeling, we rigorously validated the RCT (Randomized Controlled Trial) assumptions to ensure downstream causality.

* **Sample Ratio Mismatch (SRM):** Validated traffic split (85/15) with Chi-Square tests (p=0.9989), confirming no assignment bias.
* **Covariate Balance:** Calculated Standardized Mean Differences (SMD). All 12 features exhibited $SMD < 0.05$, confirming groups were statistically identical pre-treatment.

## 2. Casual Inference (X-Learner)
We selected the X-Learner over the T-Learner due to the dataset's severe class imbalance (conversions ~$0.3\%$).

- **Propensity Scoring:** modeing $P(T = 1| X)$ to weight the estimators.
- **Imputation:** Estimating counterfactuals for Control (What is they were treated?) and Treatment (What if they weren't?)
- **CATE Estimation:** The final model predicts $\tau(x) = \mathbb{E}[Y \mid X, T = 1] − \mathbb{E}[Y \mid X, T = 0]$.

## 3. Uncertainty Quantification
We rejected point estimates in favor of **Bootstrapped Qini Curves**. By resampling the test set 100 times, we generated $95\%$ Confidence Intervals for our uplift curves.

- **Result:** The lower bound of the CI consistently outperformed random targeting, providing statistical guarantees of value to stakeholders.

## 4. Economic Simulation (Contextual Bandits)
We simulated a LinUCB Bandit using the Replay Method (Off-Policy Evaluation) on logged data.

- **Logic:** The agent only bids when `Predicted_Uplift * LTV > Cost`.
- **Sensitivity Analysis:** We stress-tested the policy against rising CPMs. The model demonstrated "Anti-Fragility" automatically retreating to safer, higher-probability segments as costs rose, maintaining positive profitability even at 5x cost.

## 5. Production Engineering (Distillation)
The X-Learner ensemble (**4+ gradient boosted trees**) is too slow for Real-Time Bidding (RTB) SLAs (**<10ms**).

-  **Technique:** Knowledge Distillation.
- **Teacher:** X-Learner.
- **Student:** Depth-Constrained Decision Tree.
- **Outcome:** The Student model achieved  **$0.093 profit/user** (vs Teacher's **$0.089**), proving that simpler models acted as effective regularizers against overfitting.

<br>

# 📊 Key Results
| **Metric**               | **Fixed A/B Strategy**     | **Causal Bandit Strategy** | **Impact**               |
|----------------------|----------------------|----------------------|--------------------|
| **Conversion Rate**       | 0.30%                | 0.65%                | **+116%**              |
| **Lift**                  | +59.4%               | +350% (Top Decile)   | **6x Precision**       |
| **Net Profit / User**     | **-$0.05** (Loss)        | **+$0.09** (Profit)      | **Turnaround**         |
| **Inference Latency**     | N/A                  | < 1ms                | **RTB Ready**          |

<br>

### **Insight: The "Persuadables"**
Surrogate modeling revealed that **Feature** `f4` is the primary driver of persuadability.
- **Strategy:** Users with `f4 < 11.7` and `f3 < 3.0` are "Persuadables".
- Action: Aggressive bidding on this micro-segment (approx. $15\%$ of population) yields $70\%$

---
# 💻 Installation & Usage
## Prerequisites
- Python 3.9+
- 16GB RAM recommended for full dataset processing.

## 1. Clone and Install
<pre>
git clone https://github.com/BhargavKumarNath/A-B-Testing.git

cd A-B-Testing

pip install -r requirements.txt
</pre>

## 2. Run the End-to-End Pipeline

This script handles ingestion, validation, training, simulation, and distillation.

<pre>python main.py</pre>
Artifacts will be saved to `results/` (Models, Plots,CSVs).

## 3. Launch the Dashboard
Access the interactive command center locally

<pre> streamlit run dashboard/app.py</pre>

<pre>
├── data/                   # Raw Parquet files (gitignored)
├── results/                # Generated artifacts (Plots, Models)
├── src/
│   ├── components/
│   │   ├── data_loader.py    # Polars Optimization
│   │   ├── validation.py     # SRM & SMD Checks
│   │   ├── statistics.py     # Frequentist ATE / CUPED
│   │   ├── models.py         # X-Learner Implementation
│   │   ├── evaluation.py     # Bootstrapped Qini
│   │   ├── segmentation.py   # Surrogate Trees
│   │   ├── bandit.py         # LinUCB Simulation
│   │   ├── distillation.py   # Student/Teacher Engine
│   │   └── main.py           # Pipeline Entry Point
│   ├── analysis/
│   │   └── baseline.py       # Baseline A/B Analysis
│   └── utils/
│       └── plotting.py       # Plotly Visualization Suite
├── app.py                  # Streamlit Dashboard Entry Point
├── tests/                  # Unit and Integration Tests for statistical checks
└── README.md               # Documentation
</pre>

## 🛡 License & Acknowledgements
Dataset: Criteo Uplift Modeling Dataset.
License: MIT License.
Authorship: Designed and implemented by Bhargav Kumar Nath, Pipeline Architect.
