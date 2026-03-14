# Criteo Uplift: Causal AI for Algorithmic Profit Optimization

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![Polars](https://img.shields.io/badge/Polars-Fast_Data-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boosting-green)
![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📉 Executive Summary

**The Problem:**  
Traditional A/B testing revealed that while broad advertising targeting increased conversion rates by **+59.4%**, the high cost of media resulted in a **net loss of $0.05 per user**. The campaign was statistically effective but economically disastrous.

**The Solution:**  
We architected an end-to-end Causal Inference pipeline to transition from descriptive analytics ("What happened?") to prescriptive intervention ("Who should we target?"). Using **X-Learner architectures** and **Contextual Bandits**, we successfully isolated the sub-population of "Persuadable" users.

**The Impact:**
- **Unit Economics Turnaround:** Turned negative margins into a positive outcome, achieving **$0.09 profit per user**—a **$0.14 improvement** from the baseline rollout strategy.
- **Efficiency:** Drastically reduced ad spend volume while capturing the true incremental value of the campaign.
- **Production Readiness:** Distilled heavy meta-learners into a sub-millisecond Decision Tree for Real-Time Bidding (RTB).

[👉 View the Live Command Center](https://ablytics.streamlit.app/)

---

## 🏗 System Architecture

The project implements a **memory-optimized causal inference pipeline** designed for scale (14M+ rows) and sub-millisecond production inference. The architecture transitions from rigorous offline experimental validation to profit-aware policy simulation, culminating in a distilled, low-latency artifact for Real-Time Bidding (RTB).

The end-to-end workflow is composed of five specialized subsystems:

### 1. Ingestion & Optimization (`DataLoader`)
- **Responsibility:** Efficiently loads massive Parquet datasets while preventing out-of-memory (OOM) errors.
- **Workflow:** Utilizes **Polars** for lazy-evaluation and columnar mapping. Executes explicit type downcasting (e.g., `Float64` → `Float32`, `Int64` → `Int8`), reducing the memory footprint by ~50% and executing in $O(1)$ complexity relative to Spark clusters.

### 2. Integrity Gatekeeper (`ExperimentValidator`)
- **Responsibility:** Programmatically guarantees the statistical soundness of the randomized control trial before any modeling occurs, mitigating *Garbage In, Garbage Out*.
- **Workflow:** 
  - Computes a **Sample Ratio Mismatch (SRM)** via Chi-Square test to catch broken traffic splits.
  - Calculates **Standardized Mean Differences (SMD)** across all covariates to ensure Treatment and Control groups are identical pre-exposure.
  - *Fails fast* by initiating an abort sequence if bias is detected.

### 3. Offline Causal Learning (`X-Learner`)
- **Responsibility:** Estimates the Conditional Average Treatment Effect (CATE) for every individual to isolate the persuadable population.
- **Workflow:** Employs an **X-Learner meta-architecture** using **LightGBM** as the base learner. 
  - Trains a propensity model to correct observational bias.
  - Trains separate response models for Control and Treatment to output imputed counterfactuals.
  - Fits final effect models on the imputed data to predict individual uplift.

### 4. Economic Policy & Uncertainty (`BanditSimulator` & `UpliftEvaluator`)
- **Responsibility:** Translates raw uplift predictions into actionable, risk-aware economic policies.
- **Workflow:** 
  - The `UpliftEvaluator` performs bootstrap resampling to generate **95% Confidence Intervals** for Qini curves, quantifying deployment risk.
  - The `BanditSimulator` employs a **LinUCB Contextual Bandit** logic in an offline replay simulation to optimize for *Net Profit* rather than vanity metrics (Lift/CTR), strictly bidding when `Predicted_Uplift * LTV > Cost`.

### 5. Production Distillation (`DistillationEngine`)
- **Responsibility:** Compresses the heavy meta-learner ensemble into an ultra-fast artifact suitable for production environments.
- **Workflow:** Uses **Knowledge Distillation**. The X-Learner (Teacher) generates soft labels (smoothed CATE predictions). A depth-constrained Decision Tree (Student) is trained on these labels, yielding an interpretable ruleset with **sub-millisecond latency (~45 µs)** while retaining >95% of the Teacher's profit performance.

---

### System Data Flow Diagram

![Alt text](system_design.svg)
---

## 📊 Key Results

| Metric                  | Fixed A/B Strategy | Causal Bandit Strategy | Impact            |
|-------------------------|-----------------|----------------------|-----------------|
| Conversion Rate          | 0.30%           | 0.65%                | **+116%**       |
| Lift                     | +59.4%          | +350% (Top Decile)   | **6x Precision**|
| Net Profit / User        | -$0.05 (Loss)   | +$0.09 (Profit)      | **Turnaround**  |
| Inference Latency        | N/A             | < 1ms                  | **RTB Ready**   |

**Insight: The "Persuadables"**  
Users with high `f4 (> 11.7)` and low `f3` are micro-segmented for aggressive bidding. This small population yields the vast majority of incremental lift.

---

## 💻 Installation & Usage

### Prerequisites
- Python 3.9+  
- 16GB RAM recommended for full dataset execution (Polars optimizes this significantly).

### Clone and Install
```bash
git clone https://github.com/BhargavKumarNath/A-B-Testing.git
cd A-B-Testing
pip install -r requirements.txt
```

### Run the End-to-End Pipeline
This entry point handles ingestion, statistical validation (SRM, SMD), X-Learner training, Bandit simulation, and Distillation.
```bash
python main.py
```
*Note: Artifacts such as trained models, Qini plots, and segmentation rules will be saved to the `results/` directory.*

### Launch the Interactive Command Center
Access the principal-level analysis dashboard locally.
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
A-B-Testing/
├── data/                       # Raw Parquet files (e.g., criteo_uplift.parquet)
├── results/                    # Generated artifacts (Plots, Models, Rules)
├── src/
│   ├── components/
│   │   ├── data_loader.py      # Polars Memory Optimization
│   │   ├── validation.py       # SRM & Covariate Balance (SMD) Checks 
│   │   ├── statistics.py       # Frequentist ATE / CUPED Variance Reduction
│   │   ├── models.py           # T-Learner and X-Learner Implementation (LightGBM)
│   │   ├── evaluation.py       # Bootstrapped Qini & Decile Analysis
│   │   ├── segmentation.py     # Surrogate Trees & Feature Importance
│   │   ├── bandit.py           # LinUCB Contextual Bandit Simulation
│   │   └── distillation.py     # Knowledge Distillation (Teacher/Student Engine)
│   └── utils/
│       └── plotting.py         # Plotly Visualization Suite
├── tests/                      # Unit & Integration Tests (Pytest)
├── app.py                      # Streamlit Dashboard Entry Point
├── main.py                     # Main Execution Pipeline
├── requirements.txt            # Python Dependencies
└── README.md                   # Documentation
```

---

## 🛡 License & Acknowledgements
- **Dataset:** Based on the Criteo Uplift Modeling Dataset.
- **License:** MIT License.
- **Authorship:** Designed and implemented by Bhargav Kumar Nath.
