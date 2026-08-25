# Criteo Uplift: AI for Advertising Profit Optimization

[![Live Demo](https://img.shields.io/badge/Demo-Live_Dashboard-8B5CF6?style=for-the-badge&logo=vercel&logoColor=white)](https://frontend-blond-zeta-3p7cbxlssm.vercel.app/)
[![API Status](https://img.shields.io/badge/API-Render_Live-10B981?style=for-the-badge&logo=render&logoColor=white)](https://a-b-testing-569o.onrender.com/health)

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![Polars](https://img.shields.io/badge/Polars-Fast_Data-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boosting-green)
![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)

> **Live Dashboard URL:** [https://frontend-blond-zeta-3p7cbxlssm.vercel.app/](https://frontend-blond-zeta-3p7cbxlssm.vercel.app/)  
> **Production API (Render):** [https://a-b-testing-569o.onrender.com](https://a-b-testing-569o.onrender.com)

## Executive Summary

**The Business Problem:**  
Traditional A/B testing on advertising campaigns showed that broad targeting increased conversion rates by +59.4%. However, the high cost of media resulted in a net loss of $0.05 per user. The campaign was statistically effective but economically unprofitable.

**The Solution:**  
We built an end-to-end machine learning pipeline to transition from descriptive analytics ("What happened?") to prescriptive intervention ("Who should we target?"). Using advanced modeling techniques and Contextual Bandits, we successfully isolated the sub-population of "Persuadable" users.

**The Impact:**
* **Profitability:** Turned negative margins into a positive outcome, achieving $0.09 profit per user (a $0.14 improvement over the baseline strategy).
* **Efficiency:** Drastically reduced ad spend volume while capturing the true value of the campaign.
* **Production Readiness:** Compressed complex models into a high-speed Decision Tree for Real-Time Bidding (RTB).

## System Overview

This project implements a highly optimized data pipeline designed for scale and sub-millisecond production performance. The architecture handles everything from offline experimental validation to profit-aware simulation, ending in a low-latency model for real-time bidding.

The workflow is divided into three main areas: the offline data pipeline, the backend API, and the frontend dashboard.

### 1. Data Ingestion
* **Role:** Efficiently loads massive datasets (over 14 million rows) while preventing memory errors.
* **Technology:** Utilizes Polars for fast data processing and memory optimization, reducing the memory footprint by approximately 50%.

### 2. Data Integrity
* **Role:** Programmatically guarantees the statistical soundness of the randomized control trial before any modeling occurs.
* **Technology:** Automatically checks for broken traffic splits and ensures that Treatment and Control groups are identical before the experiment.

### 3. Machine Learning Model
* **Role:** Estimates the treatment effect for every individual to isolate the persuadable population.
* **Technology:** Employs an X-Learner architecture using LightGBM as the base algorithm.

### 4. Economic Simulation
* **Role:** Translates raw predictions into actionable, risk-aware business policies.
* **Technology:** Uses Contextual Bandit logic in a simulation to optimize for Net Profit rather than vanity metrics like Click-Through Rate. It strictly bids when the predicted value is greater than the cost.

### 5. Production Deployment
* **Role:** Compresses the heavy machine learning model into a very fast version suitable for live production environments.
* **Technology:** Uses Knowledge Distillation to create an interpretable ruleset with sub-millisecond latency (around 45 microseconds) while retaining over 95% of the original model's performance.

## Key Results

| Metric | Standard A/B Strategy | Machine Learning Strategy | Impact |
| :--- | :--- | :--- | :--- |
| Conversion Rate | 0.31% | 1.90% | **+512%** |
| Lift | +59.4% | +350% | **6x Precision** |
| Net Profit per User | -$0.05 (Loss) | +$0.09 (Profit) | **Turnaround** |
| Inference Latency | N/A | < 1ms | **Ready for Live Bidding** |

**Key Insight:**  
Users with high `f4` scores and low `f3` scores are micro-segmented for aggressive bidding. This small population yields the vast majority of the profit.

## Setup and Installation

### Prerequisites
* Python 3.9+  
* Node.js 18+ (for the React Dashboard)

### 1. Clone and Install
```bash
git clone https://github.com/BhargavKumarNath/A-B-Testing.git
cd A-B-Testing
pip install -r requirements.txt
```

### 2. Run the Offline Pipeline
This script handles data loading, statistical validation, model training, and simulation. It will generate the final models and insights.
```bash
python pipeline.py
```

### 3. Start the Backend API
The FastAPI server hosts the optimized model for real-time decisions.
```bash
uvicorn api.main:app --port 8080
```
The API will be available at `http://127.0.0.1:8080/predict`.

### 4. Launch the Frontend Dashboard
Open a new terminal, navigate to the frontend folder, install dependencies, and start the development server.
```bash
cd frontend
npm install
npm run dev
```
Access the dashboard at `http://localhost:5173`.

## Power BI Dashboard
In addition to the React application, this project includes a Power BI Dashboard for executive reporting and interactive analysis. 

**Key Features:**
* **Executive Performance:** High-level tracking comparing Bandit vs. A/B testing strategies.
* **Audience Intelligence:** Deep dive into user segments with dynamic tooltips.
* **Budget Optimisation:** Interactive tools to dynamically recalculate ROI based on custom budgets.

**Documentation:**
* **Dashboard File:** `results/powerbi/criteo_uplift_analytics.pbix`

## License and Acknowledgements
* **Dataset:** Based on the Criteo Uplift Modeling Dataset.
* **License:** MIT License.
* **Authorship:** Designed and implemented by Bhargav Kumar Nath.
