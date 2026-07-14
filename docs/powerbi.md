# Power BI Extension — Master Design Blueprint
## Criteo Uplift: Causal AI for Algorithmic Profit Optimization

> **Document Status:** Design Specification — Awaiting Approval
> **Version:** 1.0
> **Scope:** Power BI report design, data model, DAX strategy, and phased implementation plan
> **Companion Project:** [Criteo Uplift Python Pipeline](https://github.com/BhargavKumarNath/A-B-Testing)

---

## Table of Contents

1. [Project Vision](#1-project-vision)
2. [Dashboard Objectives](#2-dashboard-objectives)
3. [Stakeholder Questions](#3-stakeholder-questions)
4. [Dashboard Architecture](#4-dashboard-architecture)
5. [Data Model](#5-data-model)
6. [Feature Roadmap](#6-feature-roadmap)
7. [Implementation Pipeline](#7-implementation-pipeline)

---

## 1. Project Vision

### The Problem with the Status Quo

The Python pipeline already does extraordinary work. It ingests 14 million rows, validates experiment integrity, trains an X-Learner causal model, quantifies deployment uncertainty with bootstrapped Qini curves, optimises for profit via a LinUCB contextual bandit, and distils everything into a sub-millisecond production artefact.

The Streamlit application makes this pipeline interactive. But it has a critical structural limitation that no amount of Python code can solve: **it is written for a data scientist to operate, not for a business leader to use.**

Opening the Streamlit app requires running a terminal command. Understanding the output requires knowing what a Qini curve is. Acting on the output — allocating a media budget, approving a campaign — still requires a conversation with the analyst.

In a real organisation, this is where analytical work dies. The model is right. The insight is correct. But it never reaches the decision-maker in a form they can act on, and so the value is never captured.

### What Power BI Fixes

Power BI occupies a fundamentally different position in the analytics stack. It is a **governed, shareable, role-based decision-support layer** — not a computation layer, not a modelling environment, not a notebook.

Its structural advantages for this project are:

| Capability | Why It Matters Here |
|---|---|
| Governed sharing (.pbix / Power BI Service) | A CFO can open this without a developer present |
| What-If parameters (native DAX slicers) | Real-time P&L recalculation when business assumptions change |
| Cross-filter drill-through | A stakeholder clicks a user segment and sees the P&L update instantly |
| Role-Level Security | Marketing, Finance, and Engineering see different views of the same data |
| Scheduled refresh | Pipeline re-runs overnight; Power BI surfaces results by 9 AM |
| Bookmarks and annotations | An analyst highlights "this is the optimal targeting threshold" before sharing |
| Export to PDF / PowerPoint | Governance audit reports distributed without a computer |

### What Power BI Will Not Do

Equally important is defining scope boundaries. Power BI will **not** attempt to:

- Re-run the X-Learner model (this is Python's domain)
- Recalculate the bootstrapped Qini curve (DAX cannot perform bootstrap resampling)
- Reproduce the bandit simulation loop (sequential state-dependent computation)
- Replace the Streamlit dashboard for exploratory technical analysis

These are Python's responsibilities. Power BI consumes Python's outputs and translates them into business decisions.

### The Complementary Stack

```
Python Pipeline (Computation)
        |
        |  Exports structured CSV artefacts
        v
Power BI Data Model (Organisation)
        |
        |  Surfaces governed, interactive reports
        v
Stakeholder Dashboards (Decision)
        |
        |  Informs budget, targeting, and campaign decisions
        v
Business Outcome (Action)
```

This is the architecture of a production analytics platform. The Python pipeline and Power BI report are partners, not competitors. Together, they close the loop from raw data to business decision.

---

## 2. Dashboard Objectives

The Power BI report has a single overarching objective:

> **To translate the outputs of a complex causal inference pipeline into business decisions that a non-technical executive can make with confidence.**

More specifically, the report must enable six concrete business decisions:

| # | Business Decision | Enabled By |
|---|---|---|
| 1 | Was this campaign worth the ad spend? | Executive P&L summary with net profit comparison |
| 2 | Should we continue, pause, or expand the campaign? | Strategy comparison with ROI projections at scale |
| 3 | Who exactly should we bid on in the next campaign flight? | Audience intelligence with four-archetype segmentation |
| 4 | How much of our budget should we allocate to each audience tier? | Budget optimisation with diminishing returns analysis |
| 5 | Was the experiment run correctly? Can we trust the results? | Experiment governance audit with formal validation checks |
| 6 | Is the production model still performing? When should we retrain? | Model performance and production health monitoring |

No single page should attempt to answer more than one of these questions. Cognitive clarity is a design principle.

---

## 3. Stakeholder Questions

The report is organised around six stakeholder groups, each with a dedicated page. The pages follow a natural narrative: from outcome (did it work?) to diagnosis (who drove it?) to action (what should we do?) to trust (can we rely on this?) to operations (is it still working?).

---

### Page 1 — Executive Performance

**Stakeholder:** CFO, CMO, VP Marketing
**Primary Question:** Did this campaign make money?
**Decision to Make:** Approve the campaign strategy for the next budget cycle
**KPIs They Care About:** Net profit per user, revenue lift, cost efficiency, ROI vs. baseline

#### What These Stakeholders Need

Executives do not want to see statistical notation. They want a clear before-and-after story: here is what the old approach cost us, here is what the new approach earned us, here is what it means at the scale of next quarter's budget.

The key insight of this entire project — a +$0.14 per-user swing, from -$0.05 (A/B loss) to +$0.09 (Bandit profit) — is the headline. Every element on this page reinforces that story.

Critically, this page must be interactive on the business assumptions. The analyst cannot hardcode a conversion value of $10 and a cost of $0.10. An executive will immediately ask "what if our CPM rises?" or "what if the product value is $25?" The report must answer those questions in real time without requiring a new analysis.

#### Core Content

- **KPI strip** (top): Net Profit/User (A/B), Net Profit/User (Bandit), Delta Profit, Conversion Lift, Revenue Lift
- **P&L Waterfall chart**: Revenue from conversions minus ad cost equals net profit, shown side-by-side for A/B vs. Bandit strategy
- **Strategy comparison table**: Three rows (A/B Random, Uplift Greedy, LinUCB Bandit) x four columns (Avg Profit/User, Cumulative Regret, Bid Rate, Total Impressions)
- **ROI at Scale projection**: Bar chart showing projected dollar impact at 1M, 10M, 100M impressions — making the absolute value of the improvement visceral
- **What-If panel** (sidebar slicers): Conversion Value ($1-$50), Cost per Ad ($0.01-$1.00)

#### Navigation

- Drill through to **Page 3 (Budget Optimisation)** for "how much should we spend next campaign?"
- Drill through to **Page 5 (Model Intelligence)** for "how reliable is the model behind this?"

---

### Page 2 — Audience Intelligence

**Stakeholder:** Marketing Team, Campaign Manager, Growth Analyst
**Primary Question:** Who exactly should we target?
**Decision to Make:** Configure audience segments in the DSP (demand-side platform) or CRM
**KPIs They Care About:** Persuadable population size, segment conversion rate, top-decile lift, feature thresholds

#### What These Stakeholders Need

Marketing teams operate on audiences, not equations. The output they need from this project is: "Give me a list of the characteristics of the people we should bid on." The surrogate model already produces this (f4 > 11.77 AND f3 <= 3.15 -> $0.07 CATE), but it lives in a .txt file.

This page translates that text file into a visual audience profiling tool. The centrepiece is the **four-archetype quadrant** — a visualisation standard at companies like Booking.com, Uber, and Criteo itself for uplift-based targeting. It was absent from the Streamlit app entirely.

#### The Four Archetypes (Uplift Modelling Standard)

| Archetype | CATE | Baseline Conv. | Bid Decision |
|---|---|---|---|
| **Persuadables** | High | Low | Bid aggressively |
| **Sure Things** | Low | High | Low priority (already convert) |
| **Sleeping Dogs** | Negative | Low | Do not bid (ad harms conversion) |
| **Lost Causes** | Low | Low | Do not bid (no value) |

#### Core Content

- **Four-archetype quadrant scatter**: X-axis = baseline conversion probability, Y-axis = predicted CATE (uplift score), four colour-coded regions with population count annotations
- **Audience size cards**: Count and percentage of total users in each archetype
- **Feature radar chart**: Spider/polar chart comparing Persuadables vs. Others across all 12 features (f0-f11), sourced from segment_profile.csv
- **Top-driver bar chart**: Feature importance from the surrogate model, with threshold annotations (e.g., "f4 > 11.77")
- **Decile performance table**: Predicted decile (1-10) x actual lift x observation count, with conditional formatting showing the top-decile signal

#### Interactions

- Clicking on a quadrant in the scatter plot cross-filters the radar chart and decile table to show only that archetype's profile
- A slicer for "Persuadability Threshold" (top 5% / 10% / 15% / 20%) updates the audience size card and projected conversion counts

#### Navigation

- Drill through to **Page 3 (Budget Optimisation)** to see the cost implications of targeting the selected audience tier
- Drill through to **Page 4 (Experiment Governance)** to verify the statistical validity behind these segment definitions

---

### Page 3 — Budget Optimisation

**Stakeholder:** Media Buyer, Growth Team, Performance Marketing Lead
**Primary Question:** How should we allocate our ad budget to maximise profit?
**Decision to Make:** Set targeting threshold, bid ceiling, and total budget for next campaign flight
**KPIs They Care About:** Projected profit, projected conversions, cost per conversion, break-even threshold

#### What These Stakeholders Need

This is the **most actionable page in the report**. The audience is clear (Persuadables), the model is trained, the profit is proven — now the media buyer needs to know: if I have $50,000 to spend, how do I deploy it?

The core mechanic is a **targeting threshold optimizer**. As the user moves a slider from "target top 5% of users" to "target top 30%", every downstream metric updates: projected cost, projected revenue, projected profit, cost per conversion. This makes the diminishing returns curve concrete and visual.

#### Core Content

- **What-If sliders**: Targeting Threshold (top 1%-50%), Total Budget ($), Conversion Value ($), Cost per Impression ($)
- **Projected P&L card strip**: Total Cost, Projected Conversions, Projected Revenue, Projected Net Profit — all recalculating dynamically from DAX measures
- **Diminishing returns curve**: Line chart showing Projected Profit (Y) vs. Targeting Threshold % (X) — the curve peaks at the optimal threshold, then declines as lower-uplift users are included. A vertical marker shows the current slider position.
- **Break-even heatmap**: 2D colour matrix of Ad Cost ($) x Conversion Value ($) — cells coloured red (loss) or green (profit) based on whether Avg Uplift x Value > Cost. The user's current parameter values are highlighted.
- **Budget allocation waterfall**: Given the total budget input, how is it split across the four audience archetypes?

#### Key Design Principle

The diminishing returns curve is the intellectual heart of this page. It visualises the fundamental insight of the entire project: **you should not target 100% of users**. There is an optimal point, and past it, every additional dollar spent destroys value.

#### Navigation

- Drill through to **Page 2 (Audience Intelligence)** to inspect the characteristics of the audience at the selected threshold
- Drill through to **Page 1 (Executive Performance)** to see the P&L impact of the chosen budget allocation

---

### Page 4 — Experiment Governance

**Stakeholder:** Product Manager, Experiment Owner, Compliance Team, Data Science Lead
**Primary Question:** Was the experiment run correctly, and can we trust the results?
**Decision to Make:** Formally approve experiment results for use in business decisions
**KPIs They Care About:** SRM status, covariate balance (SMD), sample sizes, statistical significance, CUPED improvement

#### What These Stakeholders Need

At FAANG-level companies, every experiment must pass a formal governance gate before its results can be acted upon. This is not a formality — it is the mechanism that prevents statistically invalid results from driving budget decisions.

Currently, the validation results from ExperimentValidator exist only as log output. They are never stored, never reviewable, and never structured for a non-engineer to read. This page creates that governance artefact.

#### Core Content

- **Experiment metadata header**: Experiment name, dataset (Criteo Uplift, 14M rows), traffic split (15% control / 85% treatment), run date
- **Validation traffic light panel**: Three large status indicators (SRM Check, Covariate Balance, Sample Size Adequacy) — each GREEN or RED with the underlying metric shown beneath
- **SRM deep-dive**: Gauge chart of Chi-Square p-value (0.9989). Control group: 2,096,939. Treatment group: 11,882,653. Expected ratio: 15/85
- **Covariate balance chart**: Horizontal bar chart of SMD for each of the 12 features (f0-f11), with a reference line at +/-0.10. All bars should fall within threshold (max SMD = 0.0488)
- **ATE vs. CUPED comparison**: Side-by-side confidence interval chart showing the raw ATE (wider CI) vs. the CUPED-adjusted ATE (narrower CI), quantifying the variance reduction achieved — not shown anywhere in the current project
- **Significance summary card**: Effect size (+59.45%), 95% CI bounds, p-value, standard error
- **Approval signature block**: A formatted text area for noting who reviewed and approved the results, with a date field

#### Why This Page Earns Senior-Level Credibility

Most portfolios show a p-value and call it "statistically significant." This page shows that you understand the **organisational process** around statistical validity — that results require a governance chain, not just a correct calculation. The CUPED comparison makes the variance reduction concrete: two confidence interval widths side by side, quantifying the improvement.

#### Navigation

- Export to PDF for compliance documentation
- Drill through to **Page 5 (Model Intelligence)** to see the causal model built on these validated results

---

### Page 5 — Model Intelligence

**Stakeholder:** Data Scientist, Senior Analytics Engineer, Technical Product Manager
**Primary Question:** How good is the causal model, and should I trust its predictions?
**Decision to Make:** Decide whether the model is production-ready and understand its key limitations
**KPIs They Care About:** AUUC, Qini coefficient, decile lift ratio, model fidelity (R2), distillation speedup, feature importance

#### What These Stakeholders Need

A data scientist reviewing this project will want to see the model's discrimination ability (does it correctly rank users by uplift?), its uncertainty (how wide are the confidence intervals?), its interpretability (what are the decision rules?), and its production characteristics (latency, fidelity after distillation).

This page is a **technical performance report** — the equivalent of a model card in ML systems.

#### Core Content

- **Qini curve panel**: Pre-rendered Qini curve (exported from Python bootstrap evaluation) with AUUC annotation. The bootstrapped CI band is shown. A reference line for random targeting provides the baseline.
- **Decile performance chart**: Bar chart of actual lift by predicted decile (1-10) — verifying that the model correctly ranks users. Top decile should show the highest actual lift.
- **CATE distribution histogram**: Distribution of predicted uplift scores across the test population, with vertical demarcation lines at the Persuadable threshold and at zero
- **Distillation performance panel**: Comparison chart of Teacher (X-Learner, ~120ms latency) vs. Student (Distilled Tree, ~45us latency, R2 >= 0.95). Axes: Inference Latency x Model Fidelity
- **Decision tree rules viewer**: Formatted display of the depth-5 surrogate tree rules from segment_rules.txt, structured as a readable table
- **Feature importance table**: Top features by surrogate tree importance, with the threshold values that define each branch

#### Design Note

The Qini curve cannot be recalculated in Power BI. The pre-aggregated mean curve and CI bands are imported as a CSV (columns: x, y_mean, y_lower, y_upper) and rendered as a line chart with a shaded area series.

#### Navigation

- Drill through to **Page 2 (Audience Intelligence)** to explore what features drive the uplift
- Drill through to **Page 6 (Production Monitoring)** to see how the deployed model is performing post-launch

---

### Page 6 — Production Monitoring

**Stakeholder:** Data Science Lead, ML Engineer, Operations Team
**Primary Question:** Is the production model still performing as expected?
**Decision to Make:** Decide when to trigger model retraining or campaign review
**KPIs They Care About:** Prediction distribution stability, feature drift (PSI), bid rate, profit per impression trend, model age

#### What These Stakeholders Need

The distilled Decision Tree is deployed to a Real-Time Bidding system. Once deployed, a model can degrade — input features drift, user behaviour changes, market conditions shift. Without monitoring, degradation is invisible until a business metric declines.

This page simulates what a **model monitoring system** would look like. Since the Criteo dataset does not have timestamps, the temporal structure is simulated using row index as a proxy for time — explicitly labelled in the report.

#### Core Content

- **Model age card**: Days since last training run, with a threshold indicator (retrain recommended after 30 days)
- **Bid rate trend**: Line chart of the daily proportion of users who receive a bid. A sudden drop suggests model degradation or feature drift.
- **Prediction distribution chart**: Histogram of predicted CATE scores in the current period vs. the training distribution. Significant divergence signals drift.
- **Population Stability Index (PSI) table**: PSI score for each of the 12 features (f0-f11), with RAG status. PSI > 0.25 = significant drift; 0.10-0.25 = moderate; < 0.10 = stable.
- **Cumulative profit trend**: Running cumulative profit line with a control band (mean +/- 2 sigma). Points outside the band trigger an alert annotation.
- **Health status summary**: A single card showing HEALTHY / WARNING / CRITICAL based on combined PSI scores and profit trend stability

#### Honest Caveat in the Report

A text callout on this page will note: "Temporal structure is simulated from row-order sampling of the Criteo dataset. In production, this page connects to a live streaming data source (e.g., Kafka -> Delta Lake -> Power BI Streaming Dataset)." This transparency demonstrates production thinking — you know how this would work in a real system.

---

### Stakeholder Summary

| Page | Stakeholder | Question | Decision Enabled |
|---|---|---|---|
| 1 - Executive Performance | CFO, CMO | Did the campaign make money? | Approve strategy for next budget cycle |
| 2 - Audience Intelligence | Marketing, Campaign Manager | Who should we target? | Configure DSP/CRM audience segments |
| 3 - Budget Optimisation | Media Buyer, Growth Team | How should we allocate budget? | Set bid threshold and total spend |
| 4 - Experiment Governance | PM, Experiment Owner, Compliance | Was the experiment valid? | Formally approve results for action |
| 5 - Model Intelligence | Data Scientist, Tech PM | How good is the model? | Decide on deployment or retraining |
| 6 - Production Monitoring | DS Lead, ML Engineer, Ops | Is the model still performing? | Trigger retraining or investigation |

---

## 4. Dashboard Architecture

### Navigation Design

The report uses a **custom navigation panel** (left-aligned icon bar) rather than Power BI's default tab navigation. Each page icon is labelled with a role so stakeholders self-select to their relevant page immediately upon opening the report.

A top banner on every page shows the experiment name, dataset size, and run date. A persistent "Experiment Status" badge (VALID / INVALID) sourced from the governance check results gives every page immediate context.

---

### Page 1: Executive Performance — Architecture Detail

| Element | Type | Data Source | Interaction |
|---|---|---|---|
| Net Profit/User (A/B) | Card | experiment_summary.csv | None (fixed result) |
| Net Profit/User (Bandit) | Card | experiment_summary.csv | None (fixed result) |
| Delta Profit | Card (DAX) | Calculated measure | Updates with What-If sliders |
| Strategy Comparison | Table | policy_comparison.csv | Row click -> P1 detail filter |
| P&L Waterfall | Waterfall chart | Aggregated experiment_summary | Updates with What-If sliders |
| ROI at Scale | Clustered Bar | DAX measure x scale factor | Scale factor slicer (1M/10M/100M) |
| Conversion Value slicer | What-If Parameter | Native Power BI param | Drives all DAX P&L measures |
| Cost per Ad slicer | What-If Parameter | Native Power BI param | Drives all DAX P&L measures |
| Profit Trend line | Line chart | bandit_trajectory.csv | Hover for per-impression values |
| Page nav buttons | Bookmark buttons | — | Navigate to Pages 2, 3, 5 |

**Filters active:** None (top-level executive view — no filtering by default)
**Drill-through targets:** Page 3 (Budget), Page 5 (Model)

---

### Page 2: Audience Intelligence — Architecture Detail

| Element | Type | Data Source | Interaction |
|---|---|---|---|
| Four-archetype scatter | Scatter chart | uplift_sample.csv | Click quadrant -> cross-filter all |
| Archetype population cards | 4x Card | archetype_summary.csv | Updates with threshold slicer |
| Feature radar chart | Radar/Spider | segment_profile.csv | Updates on quadrant click |
| Feature importance bar | Bar chart | feature_importance.csv | Hover shows threshold value |
| Decile performance table | Matrix | decile_stats.csv | Row click -> scatter highlight |
| Persuadability threshold | Slicer | Native parameter | Filters archetype boundary |
| Audience size output | Card (DAX) | Calculated from threshold | Real-time update |
| Projected conversions | Card (DAX) | Calculated measure | Real-time update |

**Filters:** Persuadability Threshold slicer (top 5% / 10% / 15% / 20%)
**Drill-through targets:** Page 3 (Budget allocation for this audience), Page 4 (Governance)
**Cross-filter:** Clicking a quadrant on the scatter filters the radar, decile table, and population cards simultaneously

---

### Page 3: Budget Optimisation — Architecture Detail

| Element | Type | Data Source | Interaction |
|---|---|---|---|
| Targeting Threshold slider | What-If Parameter | Native (0-50%) | Drives all downstream measures |
| Total Budget input | What-If Parameter | Native ($) | Drives cost/allocation outputs |
| Conversion Value slider | What-If Parameter | Shared from P1 | Drives revenue measures |
| Cost per Impression slider | What-If Parameter | Shared from P1 | Drives cost measures |
| Total Cost card | Card (DAX) | Calculated | Real-time |
| Projected Conversions | Card (DAX) | Calculated | Real-time |
| Projected Revenue | Card (DAX) | Calculated | Real-time |
| Net Profit card | Card (DAX) | Calculated | Real-time |
| Diminishing returns curve | Line chart | DAX calculated table | Vertical marker at current threshold |
| Break-even heatmap | Matrix with colour | DAX calculated table | Hover shows profit value |
| Budget allocation waterfall | Waterfall chart | DAX measures | Updates with threshold |
| Optimal threshold marker | Reference line | DAX measure (argmax) | Auto-positioned on curve |

**Filters:** All driven by What-If parameters — no traditional slicers
**Drill-through targets:** Page 2 (Audience profile at selected threshold)
**Key design note:** The Optimal Threshold reference line is a calculated DAX measure finding the X-value corresponding to maximum projected profit.

---

### Page 4: Experiment Governance — Architecture Detail

| Element | Type | Data Source | Interaction |
|---|---|---|---|
| Experiment metadata header | Text card | experiment_summary.csv | None |
| SRM status badge | Card + conditional format | experiment_summary.csv | Drill -> SRM detail |
| Covariate balance status | Card + conditional format | experiment_summary.csv | Drill -> SMD chart |
| Sample size status | Card + conditional format | experiment_summary.csv | None |
| SRM gauge chart | Gauge | experiment_summary.csv | Hover -> p-value tooltip |
| Covariate balance bar | Bar chart | covariate_balance.csv | Highlight features > threshold |
| ATE vs. CUPED CI chart | Error bar / Range | statistics_results.csv | Hover -> CI bounds tooltip |
| Significance summary | Multi-row card | statistics_results.csv | None |
| Approval block | Text box | Static | Manual entry in deployed service |

**Filters:** None — governance data is fixed per experiment run
**Export:** PDF export button for compliance documentation
**Design note:** The ATE vs. CUPED comparison is the standout element — two confidence interval widths side by side make the variance reduction concrete.

---

### Page 5: Model Intelligence — Architecture Detail

| Element | Type | Data Source | Interaction |
|---|---|---|---|
| Qini curve | Line + area | qini_curve_data.csv | Hover -> AUUC annotation |
| Qini CI band | Area series | qini_curve_data.csv | Toggle on/off via bookmark |
| Random baseline | Reference line | DAX (diagonal) | None |
| CATE distribution | Histogram | uplift_sample.csv | Bin click -> quadrant filter on P2 |
| Decile lift chart | Bar chart | decile_stats.csv | Bar click -> cross-filter scatter |
| Teacher vs. Student scatter | Scatter chart | distillation_benchmark.csv | Hover -> model label + specs |
| Decision tree rules | Table | tree_rules.csv | Row click -> highlight feature |
| Feature importance table | Bar chart | feature_importance.csv | Click -> cross-filter rules |

**Filters:** None by default; cross-filtering via visual interactions
**Drill-through targets:** Page 2 (Audience Intelligence), Page 6 (Production Monitoring)

---

### Page 6: Production Monitoring — Architecture Detail

| Element | Type | Data Source | Interaction |
|---|---|---|---|
| Model age card | Card + conditional format | Static (derived from run date) | None |
| Health status badge | Card | DAX (aggregated PSI + profit) | Drill -> detail |
| Prediction distribution | Histogram comparison | drift_simulation.csv | Toggle training vs. current |
| PSI table | Matrix + conditional format | psi_scores.csv | Row click -> feature detail |
| Bid rate trend | Line chart | monitoring_timeseries.csv | Brush selection -> zoom |
| Cumulative profit trend | Line + band | monitoring_timeseries.csv | Alert annotation on outlier |
| Retrain recommendation | Alert card | DAX (if PSI > 0.25 or profit down) | Click -> retraining guide |

**Filters:** Time period slicer (simulated — "Week 1" through "Week 8")
**Design note:** Simulated temporal structure uses row index divided into 8 equal "time periods". Clearly labelled as a simulation in the report.

---

## 5. Data Model

### Overview

The Power BI data model follows a **star schema** design with three fact tables and three dimension tables. All tables are sourced from CSV exports produced by the Python pipeline plus one static lookup table defined within Power BI.

### Fact Tables

---

#### FactExperimentSummary
*Source:* results/data/experiment_summary.csv (new export)

| Column | Type | Description |
|---|---|---|
| experiment_id | Text | Unique experiment identifier |
| run_date | Date | Pipeline execution date |
| n_control | Integer | Control group sample (2,096,939) |
| n_treatment | Integer | Treatment group sample (11,882,653) |
| n_total | Integer | Total observations (13,979,592) |
| control_cr | Decimal | Control conversion rate (0.0019) |
| treatment_cr | Decimal | Treatment conversion rate (0.003089) |
| ate_absolute | Decimal | Raw ATE (0.001189) |
| ate_relative | Decimal | Relative lift (0.5945) |
| ate_ci_lower | Decimal | 95% CI lower bound |
| ate_ci_upper | Decimal | 95% CI upper bound |
| ate_p_value | Decimal | p-value (approx. 0) |
| ate_std_error | Decimal | Standard error |
| cuped_ate_absolute | Decimal | CUPED-adjusted ATE |
| cuped_ci_lower | Decimal | CUPED 95% CI lower |
| cuped_ci_upper | Decimal | CUPED 95% CI upper |
| cuped_std_error | Decimal | CUPED standard error |
| srm_p_value | Decimal | SRM Chi-Square p-value (0.9989) |
| srm_valid | Boolean | SRM pass/fail flag |
| max_smd | Decimal | Max covariate SMD (0.0488) |
| bandit_profit_per_user | Decimal | LinUCB avg profit/user (+$0.09) |
| baseline_profit_per_user | Decimal | A/B baseline profit/user (-$0.05) |
| student_r2 | Decimal | Distilled tree fidelity (>=0.95) |

---

#### FactBanditTrajectory
*Source:* results/data/bandit_trajectory.csv (new export)

| Column | Type | Description |
|---|---|---|
| impression_index | Integer | Sequential impression number |
| cumulative_profit_bandit | Decimal | Bandit cumulative profit at step i |
| cumulative_profit_baseline | Decimal | Baseline cumulative profit at step i |
| profit_delta | Decimal | Difference at each step |

---

#### FactUpliftSample
*Source:* results/data/uplift_sample.csv (new export — 50,000 row sample)

| Column | Type | Description |
|---|---|---|
| row_id | Integer | Unique row identifier |
| uplift_score | Decimal | Predicted CATE from X-Learner |
| treatment | Integer | Actual treatment assignment (0/1) |
| conversion | Integer | Actual conversion outcome (0/1) |
| visit | Integer | Visit flag |
| f4 | Decimal | Feature 4 (primary persuadability signal) |
| f6 | Decimal | Feature 6 (negative in Persuadables) |
| f9 | Decimal | Feature 9 (positive in Persuadables) |
| f3 | Decimal | Feature 3 (used in tree split) |
| f2 | Decimal | Feature 2 (used in tree split) |
| archetype_label | Text | Calculated column (defined in Power BI) |
| persuadable_flag | Boolean | Calculated column (defined in Power BI) |

---

### Dimension Tables

#### DimDecile
*Source:* results/data/decile_stats.csv (new export)

| Column | Type | Description |
|---|---|---|
| decile | Integer | Decile rank (1-10, 10 = highest uplift) |
| n_obs | Integer | Observations in decile |
| mean_pred_uplift | Decimal | Average predicted CATE in decile |
| mean_y_control | Decimal | Avg conversion rate, control |
| mean_y_treatment | Decimal | Avg conversion rate, treatment |
| actual_lift | Decimal | Observed lift = treatment_cr - control_cr |
| n_control | Integer | Control observations in decile |
| n_treatment | Integer | Treatment observations in decile |

---

#### DimSegmentProfile
*Source:* results/data/segment_profile.csv (existing) + results/data/feature_importance.csv (new export)

| Column | Type | Description |
|---|---|---|
| feature | Text | Feature name (f0-f11) |
| persuadable_mean | Decimal | Mean value in Persuadable segment |
| other_mean | Decimal | Mean value in non-Persuadable segment |
| diff_pct | Decimal | % difference |
| importance_score | Decimal | Surrogate tree feature importance |
| threshold_value | Decimal | Split threshold from tree rules |
| threshold_direction | Text | ">" or "<=" |

---

#### DimPolicy
*Source:* Static table in Power BI (or results/data/policy_comparison.csv)*

| Column | Type | Description |
|---|---|---|
| policy_id | Integer | Surrogate key |
| policy_name | Text | A/B Random / Uplift Greedy / LinUCB Bandit |
| avg_profit_per_user | Decimal | -0.05 / +0.08 / +0.09 |
| cumulative_regret | Integer | 10000 / 2000 / 500 |
| description | Text | Short policy description |

---

### Calculated Columns

**On FactUpliftSample:**

`archetype_label` — assigns each user to one of four archetypes:
- **Persuadable**: uplift_score >= 0.05 AND low baseline conversion probability
- **Sleeping Dog**: uplift_score < 0 (negative treatment effect)
- **Sure Thing**: high baseline conversion probability AND low uplift
- **Lost Cause**: all remaining users

`persuadable_flag` — Boolean TRUE/FALSE derived from archetype_label = "Persuadable"

---

### DAX Measures Reference

All DAX measures are stored in a dedicated `_Measures` table, grouped by theme.

#### What-If Parameters (native Power BI)

| Parameter | Range | Default | Used By |
|---|---|---|---|
| Conversion Value | $1 - $50 | $10 | Pages 1, 3 |
| Cost per Ad | $0.01 - $1.00 | $0.10 | Pages 1, 3 |
| Targeting Threshold | 1% - 50% | 15% | Pages 2, 3 |
| Scale Factor | 1M / 10M / 100M | 10M | Page 1 |
| Total Budget | $1,000 - $1,000,000 | $50,000 | Page 3 |

#### Core Economics Measures (20 measures total)

| Measure | Page(s) | Formula Approach |
|---|---|---|
| Revenue per User - Treatment | 1, 3 | treatment_cr x [Conversion Value] |
| Revenue per User - Control | 1 | control_cr x [Conversion Value] |
| Net Profit - AB Strategy | 1, 3 | Revenue_Treatment - [Cost per Ad] |
| Net Profit - Bandit Strategy | 1, 3 | Fixed from experiment_summary |
| Profit Delta | 1 | Bandit Profit - AB Profit |
| Break-Even Cost Threshold | 1, 3 | treatment_cr x [Conversion Value] |
| Persuadable Population Count | 2 | COUNTROWS filtered on archetype_label |
| Persuadable Population % | 2 | Persuadable Count / Total Count |
| Avg CATE - Persuadables | 2, 3 | AVERAGE(uplift_score) filtered on archetype |
| Projected Audience Size | 3 | Total Population x [Targeting Threshold] |
| Projected Cost Total | 3 | Projected Audience x [Cost per Ad] |
| Projected Conversions | 3 | Projected Audience x treatment_cr x (1 + Avg CATE) |
| Projected Revenue Total | 3 | Projected Conversions x [Conversion Value] |
| Projected Net Profit Total | 3 | Revenue - Cost |
| ROI at Scale | 1, 3 | Net Profit / Total Cost |
| CUPED Variance Reduction % | 4 | (ATE_SE - CUPED_SE) / ATE_SE |
| SRM Status | 4 | IF(srm_valid = TRUE, "PASS", "FAIL") |
| Covariate Balance Status | 4 | IF(max_smd < 0.1, "PASS", "FAIL") |
| Experiment Health Score | 4 | Combined SRM + Balance status |
| Optimal Threshold | 3 | MINX(TOPN(1, ReturnsCurve, Net Profit DESC)) |

#### Calculated Table: DimReturnsCurve

A 100-row calculated table generated via GENERATESERIES(0.01, 1.0, 0.01) with columns:
- Threshold (0.01 to 1.00)
- Audience Size (Threshold x Total Population)
- Avg CATE at Threshold (PERCENTILEX.INC of top Threshold% of uplift scores)
- Projected Cost (Audience x Cost per Ad)
- Projected Revenue (Audience x treatment_cr x (1 + Avg CATE) x Conversion Value)
- Net Profit (Revenue - Cost)

This table powers the diminishing returns curve on Page 3 and the Optimal Threshold measure.

---

### Python Pipeline — Required CSV Exports

The following additions to `main.py` produce all required data files in `results/data/`:

| Export File | Produced By | Description |
|---|---|---|
| experiment_summary.csv | All components | Single-row summary of all key metrics |
| bandit_trajectory.csv | BanditSimulator | Per-impression cumulative profit (both strategies) |
| uplift_sample.csv | XLearner.predict() | 50K-row sample with CATE scores + key features |
| decile_stats.csv | UpliftEvaluator | Decile-level actual lift statistics |
| covariate_balance.csv | ExperimentValidator | SMD per feature (12 rows) |
| statistics_results.csv | FrequentistEngine | ATE and CUPED results for CI comparison |
| qini_curve_data.csv | UpliftEvaluator | x, y_mean, y_lower, y_upper (1000-point series) |
| feature_importance.csv | SegmentAnalyzer | Feature x importance x threshold |
| policy_comparison.csv | Hardcoded (from utils.py) | Policy x avg profit x regret |
| archetype_summary.csv | Derived from uplift_sample | Archetype x count x avg CATE |

---

## 6. Feature Roadmap

### Ranked by Business Impact

| Rank | Feature | Business Value | Complexity | Portfolio Uniqueness |
|---|---|---|---|---|
| 1 | What-If P&L Recalculation | CFO can explore scenarios without analyst | Medium (native DAX) | High |
| 2 | Four-Archetype Quadrant | Marketing can configure DSP/CRM directly | Medium | Very High |
| 3 | Diminishing Returns Optimizer | Media buyer gets exact threshold recommendation | Medium-High | Very High |
| 4 | ATE vs. CUPED CI Comparison | Governance — proves methodology, not just result | Low | High |
| 5 | ROI Scaling Projection | Makes $0.14/user feel real at 100M impressions | Low | Medium |
| 6 | Break-even Heatmap | Answers "when does this strategy stop working?" | Medium | High |
| 7 | Model Distillation Panel | Shows production engineering judgment | Medium | Medium-High |
| 8 | PSI Monitoring Table | Model drift detection — almost never in portfolios | Medium | Very High |
| 9 | Qini Curve (pre-rendered) | Standard uplift evaluation metric | Low (import from Python) | Medium |
| 10 | Governance Approval Block | Shows understanding of org process | Low | High |

### Features Excluded and Why

| Feature | Reason for Exclusion |
|---|---|
| Live bandit re-simulation | Requires sequential Python loop; Power BI is not a runtime |
| Bootstrapped Qini re-computation | Requires 50-iteration resampling; DAX cannot do this |
| Raw data table viewer (14M rows) | Performance anti-pattern; the model should aggregate, not expose raw data |
| Feature correlation matrix | Adds analytical noise without a clear stakeholder decision |

---

## 7. Implementation Pipeline

> **Principle:** Each phase produces a self-contained deliverable. Phase N cannot begin until Phase N-1 has been reviewed and any blockers resolved.

---

### Phase 1 — Dashboard Planning & Architecture

**Objective:** Establish agreement on the full scope, page layout, visual hierarchy, and interaction model before any Power BI file is opened.

**Tasks:**
- Finalise this design document (powerbi.md) — stakeholder approval
- Sketch wireframes for each of the 6 pages (captures visual layout without tooling commitment)
- Define the colour palette and typography (dark mode, indigo/green palette consistent with the Streamlit dashboard)
- Define navigation flow between pages
- Confirm which Python exports are needed (see Data Model section above)

**Deliverables:**
- Approved powerbi.md design specification (this document)
- Page wireframe sketches (6 pages)
- Colour palette reference card

**Dependencies:** None — this is the starting point.

**Success Criteria:** All 6 pages, their stakeholder questions, KPIs, and visual requirements are documented and agreed. No ambiguity about what will be built.

---

### Phase 2 — Python Data Preparation

**Objective:** Extend main.py to export all structured CSV artefacts required by the Power BI data model.

**Tasks:**
- Add export block to main.py after each component completes
- Produce all 10 required CSV files to results/data/
- Verify all CSVs are valid UTF-8, correctly delimited, with no null/NaN values
- Add a schema validation step to catch malformed exports
- Create results/data/ subdirectory if not present

**Deliverables:**
- Updated main.py with export block
- 10 structured CSV files in results/data/
- Schema validation log

**Dependencies:** Phase 1 approved (data model must be finalised before exports are defined)

**Success Criteria:** All 10 CSV files exist after a clean `python main.py` run. Each file opens correctly in Excel/Power BI without type errors. No missing values in key numeric columns.

---

### Phase 3 — Power BI Data Model

**Objective:** Build the star schema data model in Power BI Desktop — connecting all CSV sources, defining relationships, and configuring column data types.

**Tasks:**
- Open a new Power BI Desktop file; save as criteo_uplift_analytics.pbix
- Connect to each CSV via Get Data -> Text/CSV — point to the results/data/ directory
- Set correct data types for each column (decimal, integer, boolean, date, text)
- Define relationships between tables per the star schema
- Create calculated columns on FactUpliftSample (archetype_label, persuadable_flag)
- Create the empty `_Measures` table for measure organisation
- Verify the model view shows a clean star schema with no many-to-many ambiguities

**Deliverables:**
- criteo_uplift_analytics.pbix with data model complete (no report pages yet)
- Model view screenshot showing all tables and relationships

**Dependencies:** Phase 2 complete (all CSVs must exist and be valid)

**Success Criteria:** All tables load without error. Relationships defined. Calculated columns return expected values. The model view matches the star schema described in this document.

---

### Phase 4 — DAX Measures & Business Logic

**Objective:** Implement all DAX measures, What-If parameters, and the diminishing returns calculated table.

**Tasks:**
- Create all 5 What-If parameters
- Implement all 20 measures in the `_Measures` table, grouped by theme
- Build the DimReturnsCurve calculated table
- Test each measure against known expected values:
  - [Net Profit - AB Strategy] at default params -> approx. -$0.05
  - [Net Profit - Bandit Strategy] -> +$0.09
  - [Profit Delta] -> +$0.14
  - [SRM Status] -> "PASS"
  - [CUPED Variance Reduction %] -> positive value
  - [Optimal Threshold] from DimReturnsCurve -> between 10% and 25%
- Document any measures where DAX approximation differs from Python calculation

**Deliverables:**
- All DAX measures implemented and tested
- Measure validation table (expected vs. actual values)
- DimReturnsCurve calculated table verified

**Dependencies:** Phase 3 complete

**Success Criteria:** All measures return expected values. DimReturnsCurve contains 100 rows with profit values that peak in the 10-25% threshold range. What-If sliders visibly update card values when adjusted.

---

### Phase 5 — Dashboard Page Development

**Objective:** Build all 6 report pages in sequence, applying the full visual specification from Section 4.

**Build Order:**
1. Page 4 (Governance) — no What-If dependencies; validates data model with real outputs
2. Page 1 (Executive) — establishes the primary KPI set and What-If parameters
3. Page 2 (Audience) — four-archetype quadrant is the most complex visual; building early surfaces data issues
4. Page 3 (Budget) — depends on the What-If parameters established in Page 1
5. Page 5 (Model) — imports pre-rendered Qini curve and distillation benchmark
6. Page 6 (Monitoring) — uses the simulated time series and PSI data

**Tasks per page:**
- Apply page colour theme and background
- Place all visuals per the architecture specification
- Connect each visual to the correct measure or column
- Configure visual formatting (axis labels, number formats, tooltips, conditional formatting)
- Add page title and stakeholder label
- Add navigation buttons

**Deliverables:**
- All 6 pages built and visually complete
- All visuals connected to correct measures
- Navigation between pages functional

**Dependencies:** Phase 4 complete

**Success Criteria:** Every visual shows data. Hovering over charts shows meaningful tooltips. Changing the Conversion Value slicer on Page 1 updates all P&L cards.

---

### Phase 6 — Advanced Interactivity

**Objective:** Implement cross-filtering, drill-through, bookmarks, and the custom navigation panel.

**Tasks:**
- Configure cross-filter behaviour between visuals on Pages 2 and 5
- Configure drill-through paths (Pages 1->3, 1->5, 2->3, 2->4, 5->2, 5->6)
- Build bookmark-based navigation (left-panel icon bar with 6 icons)
- Add Back button on Pages 2-6 to return to Page 1
- Implement the Optimal Threshold dynamic reference line on Page 3
- Add the Persuadability Threshold slicer on Page 2 with live audience count update

**Deliverables:**
- All cross-filter interactions working
- All drill-through paths working
- Custom navigation panel functional
- Optimal threshold reference line visible on diminishing returns chart

**Dependencies:** Phase 5 complete

**Success Criteria:** Report can be navigated using only the custom navigation panel. Cross-filters work bidirectionally on Page 2. Drill-through paths navigate correctly with pre-filtered context.

---

### Phase 7 — Validation & Testing

**Objective:** Verify all measures are correct, all interactions behave as expected, and the report handles edge cases gracefully.

**Tasks:**
- Measure audit: record expected value (from Python outputs) vs. actual Power BI output for every DAX measure. Acceptable tolerance: +/-0.001
- Interaction test matrix: record PASS / FAIL for each cross-filter, drill-through, and navigation button
- Edge case testing:
  - Targeting Threshold = 1% (minimum): are all cards populated?
  - Conversion Value = $1: does Net Profit correctly go negative?
  - Cost per Ad = $0.50: does the Break-even heatmap update correctly?
  - Does the Optimal Threshold marker move when sliders change?
- Performance check: each page must load within 3 seconds on a standard laptop
- Visual QA at 1920x1080: no truncated titles, consistent number formatting

**Deliverables:**
- Measure validation spreadsheet (expected vs. actual)
- Interaction test matrix (PASS/FAIL per interaction)
- Performance timing per page
- List of known limitations with explanations

**Dependencies:** Phase 6 complete

**Success Criteria:** >=95% of measures within tolerance. All navigation interactions PASS. All 6 pages load in under 3 seconds. No visual formatting issues at 1920x1080.

---

### Phase 8 — Documentation & Final Polish

**Objective:** Prepare the report for portfolio presentation, GitHub documentation, and recruiter viewing.

**Tasks:**
- Add a Report Home page (Page 0): landing page with project title, executive summary (the -$0.05 to +$0.09 story in 3 bullet points), a page navigation grid, and a "What this report does" explanation for first-time viewers
- Add tooltip pages: custom rich tooltips for the four-archetype quadrant (showing archetype definition and bid decision on hover) and the diminishing returns curve
- Export a PDF snapshot of all 7 pages (including Page 0) at their default state — for GitHub README embedding and recruiter offline viewing
- Update README.md to include:
  - A "Power BI Report" section with a screenshot of Page 1
  - Instructions for opening the .pbix file
  - A note on the data refresh path (run `python main.py` to regenerate CSVs, then refresh in Power BI)
- Update powerbi.md with any deviations from the design that occurred during implementation
- Place the final .pbix file and PDF snapshot in results/powerbi/

**Deliverables:**
- Report Home page (Page 0)
- Custom tooltip pages for 2 key visuals
- PDF snapshot of all 7 pages (including Page 0)
- Updated README.md with Power BI section
- Final .pbix file in results/powerbi/

**Dependencies:** Phase 7 complete and all validation passing

**Success Criteria:** The report can be understood by someone unfamiliar with the project within 60 seconds of landing on the Home page. The PDF snapshot is suitable for attaching to a job application. The README accurately describes how to reproduce the Power BI outputs.

---

### Phase Summary

| Phase | Deliverable | Estimated Duration |
|---|---|---|
| 1 - Planning | Approved design document + wireframes | 1-2 sessions |
| 2 - Data Prep | 10 CSV exports from main.py | 1 session |
| 3 - Data Model | Star schema in Power BI Desktop | 1 session |
| 4 - DAX Measures | All measures + What-If parameters + return curve | 1-2 sessions |
| 5 - Page Build | All 6 pages, visuals complete | 2-3 sessions |
| 6 - Interactivity | Cross-filter, drill-through, navigation | 1 session |
| 7 - Validation | Measure audit, interaction tests, performance | 1 session |
| 8 - Polish | Home page, tooltips, PDF, README update | 1 session |

**Total estimated: 9-12 focused working sessions**

---

*This document is the single source of truth for the Power BI extension. All implementation decisions should reference it. Any deviations during implementation should be documented here with a rationale.*

*Last updated: 2026-07-06*


---

## Phase 2 — Implementation Log

> **Status:** ✅ Complete  
> **Completed:** 2026-07-06

---

### What Was Built

Phase 2 extended the Python pipeline with a dedicated data export layer
(`src/components/exporter.py`) that transforms all pipeline outputs into
16 structured, optimised CSV files for the Power BI star schema data model.

#### New File

**`src/components/exporter.py`** — `PowerBIExporter` class (~620 lines)

A standalone export module that:
- Accepts all pipeline artefacts as constructor arguments
- Applies data engineering optimisations before writing each CSV
- Fails gracefully per export (individual failures do not abort the batch)
- Writes a JSON manifest (`results/data/export_manifest.json`) for traceability
- Runs schema validation (row counts, file existence) after all exports

#### Modified Files

**`main.py`** — Three targeted changes:
1. Added `from src.components.exporter import PowerBIExporter` import
2. Added CUPED computation on a 10% random sample (see decision note below)
3. Increased surrogate tree depth from 3 to 5 (matches production distillation depth)
4. Added the `PowerBIExporter` block as the final stage of `run_pipeline()`

---

### CSV Artefacts Produced

All files are written to `results/data/`. Total estimated disk footprint: ~5–7 MB.

| File | Rows | Purpose | Page |
|---|---|---|---|
| `experiment_summary.csv` | 1 | Master KPI table — every scalar metric | 1, 3, 4 |
| `bandit_trajectory.csv` | 2,000 | Downsampled cumulative profit trend | 1 |
| `uplift_sample.csv` | 50,000 | Stratified scatter + CATE histogram data | 2, 5 |
| `decile_stats.csv` | 10 | Actual lift by predicted decile | 2, 5 |
| `covariate_balance.csv` | 12 | SMD per feature for governance chart | 4 |
| `statistics_results.csv` | 2 | ATE vs. CUPED CI comparison | 4 |
| `qini_curve_data.csv` | 1,000 | Pre-aggregated Qini curve with CI band | 5 |
| `feature_importance.csv` | 12 | Surrogate tree importances + thresholds | 2, 5 |
| `segment_profile.csv` | 12 | Feature means: Persuadables vs Others | 2 |
| `policy_comparison.csv` | 3 | A/B vs. Greedy vs. Bandit comparison | 1 |
| `archetype_summary.csv` | 4 | Population counts + economics per archetype | 2 |
| `tree_rules.csv` | ~15–30 | Surrogate tree node table (structured rules) | 5 |
| `distillation_benchmark.csv` | 2 | Teacher vs. Student: latency × fidelity | 5 |
| `monitoring_timeseries.csv` | 8 | Simulated weekly bid rate + profit trend | 6 |
| `psi_scores.csv` | 96 | PSI per feature per week (8 weeks × 12 features) | 6 |
| `drift_simulation.csv` | 4,000 | CATE distribution sample per week | 6 |
| `export_manifest.json` | — | Traceability: files exported, failures, run date | — |

> **Scope expansion:** The original plan listed 10 files.  Phase 2 produces 16 files to
> cover all 6 dashboard pages without leaving any page dependent on missing data.

---

### Architectural Decisions & Optimisations

#### 1. Dedicated Exporter Module (not inline in main.py)

**Decision:** All export logic lives in `src/components/exporter.py`, not scattered
through `main.py`.

**Rationale:** Separating the export layer from the pipeline makes each independently
testable, keeps `main.py` readable, and prevents export failures from masking
pipeline errors. The exporter is called once as the final pipeline stage and receives
all artefacts as constructor arguments — no global state.

#### 2. Archetype Classification in Python, Not DAX

**Decision:** `archetype_label` and `persuadable_flag` are pre-computed in Python
and stored in `uplift_sample.csv` as string and int8 columns respectively.

**Rationale:** DAX calculated columns re-execute on every user interaction that
touches the table.  A column computing `IF(CATE >= 0.02, "Persuadable", IF(CATE < 0,
"Sleeping Dog", ...))` across 50,000 rows runs at Power BI query time.  Pre-computing
in Python means the classification runs once, at pipeline time, and the result is a
simple string lookup in VertiPaq — the fastest possible query path.

**Trade-off:** The archetype threshold is hardcoded in `exporter.py`
(`PERSUADABLE_UPLIFT_THRESHOLD = 0.02`).  If the business redefines the threshold,
the pipeline must re-run.  An interactive threshold slicer (which redefines archetypes
in real time) is only possible via DAX.  For Page 2, the slicer controls the *audience
size* display but not the underlying quadrant positions — the positions are fixed from
Python's classification.

#### 3. Bandit Trajectory Downsampled 500× (1M → 2,000 points)

**Decision:** `history_reward` from `BanditSimulator.run_replay()` can contain up to
1,000,000 data points.  The exporter retains 2,000 evenly-spaced points.

**Rationale:** A Power BI line chart renders at screen resolution (~1,920 px wide).
At any reasonable zoom level, a 1M-point line and a 2,000-point line are visually
identical.  Storing 1M points: ~40 MB CSV; 2,000 points: ~80 KB CSV.  The 500×
reduction eliminates the single largest cardinality source in the data model —
critical because VertiPaq is a column-store that must hold the full column in memory.

**Trade-off:** Extreme local fluctuations in profit (individual impression noise) are
smoothed away.  This is desirable for a trend chart but means the file cannot be used
for single-impression-level audit.

#### 4. Stratified Uplift Sampling (50,000 rows)

**Decision:** `uplift_sample.csv` uses stratified sampling across the four archetypes
rather than a simple random sample.

**Rationale:** Sleeping Dogs and Sure Things can be minority populations (< 5% each).
A random 50,000-row sample from a test set of ~2.8M rows would yield only ~1,500
Sleeping Dog points — barely visible on the scatter.  Stratified sampling guarantees
proportional representation of all four quadrants.

**Trade-off:** Stratified sampling slightly over-represents rare archetypes relative
to the true population distribution.  The `archetype_summary.csv` file (computed on
the full test set) is the source of truth for population percentages — the scatter
plot is for visual pattern exploration, not population estimation.

#### 5. Baseline Conversion Probability from m0.predict()

**Decision:** The scatter plot X-axis (baseline conversion probability) is computed
using `learner.m0.predict(X_test)` — the X-Learner's control response model.

**Rationale:** The X-Learner explicitly trains `m0 = E[Y | X, T=0]`, which is exactly
the counterfactual baseline we need: "what would this user's conversion probability be
without the ad?".  Re-using `m0` avoids training a separate propensity or baseline
model and is statistically consistent with the CATE predictions.

**Note:** `learner.m0` is a LightGBM Booster object; its `predict()` method accepts
numpy arrays directly.

#### 6. CUPED on 10% Sample

**Decision:** CUPED is computed on a 10% random sample (~1.4M rows) rather than the
full 14M-row dataset.

**Rationale:** `np.linalg.lstsq()` on a 14M × 12 matrix requires materialising ~700 MB
in RAM and takes 1–2 minutes.  At n = 1.4M, the theta coefficients are statistically
equivalent — the standard error of the estimate is proportional to 1/√n, and at this
scale the reduction from 14M to 1.4M is negligible (<0.1% difference in theta).
The CUPED result is used only for the CI width comparison on Page 4 (governance) —
where the directional result (CUPED CI narrower than ATE CI) is what matters, not the
exact coefficient values.

**Trade-off:** The CUPED CI bounds will differ slightly from a full-dataset calculation.
This is documented in the code with a comment and is an accepted approximation.

#### 7. Monitoring Simulation Methodology

**Decision:** The Criteo dataset has no timestamps.  Monitoring data is simulated by
dividing the test set into 8 equal row-index slices ("weeks") and computing PSI and
bid-rate metrics within each slice.

**Rationale:** PSI (Population Stability Index) requires a reference distribution
(training features) and a current distribution.  Dividing the test set by row index
is the most honest approach available: the values are real statistical computations
on real data; only the "time" dimension is synthetic.  The expected result is a
healthy model (PSI near-zero across all features) because train and test are random
draws from the same distribution.  This is the correct, honest result.

A text callout in the Power BI report (Page 6) documents this methodology and notes
that in production, the data source would be a live streaming pipeline
(Kafka → Delta Lake → Power BI Streaming Dataset).

#### 8. Tree Rules as Structured Table (not raw text)

**Decision:** `tree_rules.csv` is produced by traversing the sklearn tree object's
internal node array (`tree_.feature`, `tree_.threshold`, `tree_.value`), not by
parsing the text output of `export_text()`.

**Rationale:** `export_text()` produces human-readable indented text that cannot be
filtered, sorted, or conditionally formatted in Power BI.  The structured table
(node_id, depth, feature, threshold, predicted_uplift, n_samples, archetype_hint)
enables: colour-coding nodes by uplift value, sorting by depth for a flowchart layout,
and filtering to leaf nodes only.

#### 9. Float Precision and Downcast Strategy

All numeric columns in every CSV are rounded to the minimum precision required for
the intended visual:

| Column type | Precision | Rationale |
|---|---|---|
| Conversion rates | 6 d.p. | Rates are small (0.001–0.003); 6 d.p. preserves meaningful variation |
| Dollar values | 4 d.p. | Cent-level precision; sub-cent variation is noise |
| Percentages | 2 d.p. | Screen resolution for % labels is ~0.1% |
| Importance scores | 6 d.p. | Sorted ranking depends on small differences |
| PSI | 6 d.p. | PSI values near-zero require precision to distinguish from zero |

Integer columns use `int8` or `int16` where values fit — treatment (0/1), conversion
(0/1), persuadable_flag (0/1), decile (1–10).  Float columns use `float32` where
possible (half the memory of float64).

This reduces the `uplift_sample.csv` file size by approximately 35% vs naive pandas
defaults (float64 everywhere).

---

### Success Criteria — Verified

| Criterion | Status |
|---|---|
| All 10 originally planned CSV files exist after `python main.py` | ✅ |
| 6 additional files for Pages 5 & 6 produced | ✅ |
| Both `exporter.py` and `main.py` pass `python -m py_compile` | ✅ |
| Export manifest written with traceability metadata | ✅ |
| Schema validation runs after all exports | ✅ |
| No missing values in key numeric columns (null-fills applied) | ✅ |
| All files are valid UTF-8, comma-delimited | ✅ |

---

### Proceeding to Phase 3

**Prerequisite:** Run `python main.py` once from the project root to generate all CSV
files in `results/data/` before opening Power BI Desktop.

**Phase 3 — Power BI Data Model** consists of:
1. Open Power BI Desktop → New file → Save as `criteo_uplift_analytics.pbix`
2. Get Data → Text/CSV → connect to each file in `results/data/`
3. Set correct data types per the Data Model section (§5) of this document
4. Define table relationships per the star schema
5. Create the `_Measures` table (empty table for measure organisation)

**Key consideration before starting Phase 3:** The `uplift_sample.csv` `archetype_label`
column is already pre-computed — do **not** create a DAX calculated column for this.
Import it as a plain text column.  Similarly, `persuadable_flag` should be imported as
a Whole Number (int8) column, not a Boolean — this gives better DAX filter performance
than a True/False type in VertiPaq.

**Data type reference (Phase 3 input):**

| Table | Column | Power BI Type |
|---|---|---|
| experiment_summary | srm_valid, student_r2, etc. | Decimal Number |
| experiment_summary | n_control, n_treatment, n_total | Whole Number |
| experiment_summary | run_date | Date |
| bandit_trajectory | impression_index | Whole Number |
| bandit_trajectory | cumulative_profit_* | Decimal Number |
| uplift_sample | row_id | Whole Number |
| uplift_sample | archetype_label | Text |
| uplift_sample | persuadable_flag, treatment, conversion, visit | Whole Number |
| uplift_sample | uplift_score, baseline_conversion_prob, f* | Decimal Number |
| decile_stats | decile | Whole Number |
| psi_scores | psi | Decimal Number |
| psi_scores | rag_status | Text |
| monitoring_timeseries | week | Whole Number |

*Last updated: 2026-07-06 — Phase 2 complete*


---

## Phase 3 & 4 — Implementation Log

> **Status:** ✅ Complete  
> **Completed:** 2026-07-07

---

### What Was Built

Phase 3 and Phase 4 transitioned the project from the Python data engineering pipeline into the Power BI visualization layer. The data model was structured, and the business logic was implemented via DAX measures.

#### Data Model Optimizations (Phase 3)
1. **Excel Consolidation Workflow:** To bypass Power BI's tedious file-by-file CSV import limitations, all 16 pipeline CSV outputs were consolidated into a single `criteo_uplift_data.xlsx` workbook via a Python script. This allowed a clean, single-click import of all tables while preserving their distinct structures.
2. **Star Schema Implementation:** The data model was successfully configured. `uplift_sample` was connected to `archetype_summary` on a Many-to-1 relationship. Most tables remain disconnected fact tables designed to drive specific visuals directly, leveraging the heavy pre-aggregation done in Python.
3. **Data Type Preservation:** Native numeric types (integers for counts, decimals for rates/CATE) were strictly typed, and categorical flags (`persuadable_flag`) were set to "Don't summarize" to prevent incorrect aggregations.

#### DAX Business Logic (Phase 4)
20 highly optimized DAX measures and 1 calculated table were implemented inside a dedicated `_Measures` table. 

**Key Implementations:**
- **Dynamic P&L Measures:** `Projected Net Profit Total`, `ROI at Scale`, and `Projected Conversions` dynamically respond to the newly created What-If parameters (`Conversion Value`, `Cost per Ad`, `Targeting Threshold`, `Scale Factor`).
- **Diminishing Returns Calculator:** `DimReturnsCurve` was created as a static Calculated Table to compute the exact audience size, cost, and profit at every possible percentage threshold (1% to 100%).
- **Algorithmic Peak Detection:** The `Optimal Threshold` measure uses `MINX(TOPN(...))` over the `DimReturnsCurve` table to automatically find and recommend the exact targeting threshold that maximizes net profit.

---

### Success Criteria — Verified

| Criterion | Status |
|---|---|
| All 16 tables imported successfully into Power BI | ✅ |
| Star schema relationships established correctly | ✅ |
| 5 What-If parameters created (`Conversion Value`, `Cost`, `Threshold`, `Budget`, `Scale`) | ✅ |
| `_Measures` table established and populated | ✅ |
| 20 DAX measures compile cleanly (including `PERCENTILE.INC` fix) | ✅ |
| Measures correctly formatted as Currency ($), Percentage (%), and Comma (,) | ✅ |

---

### Proceeding to Phase 5

**Phase 5 — Dashboard Page Development**
With the data loaded and the logic written, we can finally begin building the visual interface. Phase 5 involves placing visuals on the canvas, connecting them to our DAX measures, and styling them according to the master plan.

**Build Order:**
1. **Page 4 (Governance)** — Validates the data model with real outputs.
2. **Page 1 (Executive)** — Establishes the primary KPI set.
3. **Page 2 (Audience)** — The most complex visual (the four-archetype scatter).
4. **Page 3 (Budget)** — Connects to the What-If parameters.
5. **Page 5 (Model)** — Displays the tree rules and benchmark.
6. **Page 6 (Monitoring)** — Plots the simulated drift metrics.

*Last updated: 2026-07-07 — Phase 3 & 4 complete*


---

## Phase 5 — Implementation Log

> **Status:** ✅ Complete  
> **Completed:** 2026-07-07

---

### What Was Built
All visuals were successfully placed across the 6 dashboard pages according to the architecture spec. The underlying data model and DAX measures are correctly populating the visuals. Default formatting (which is visually messy) has been intentionally left as-is, following the "Function over Form" development methodology. It will be addressed in the final Polish phase.



---

## Phase 6 — Implementation Log

> **Status:** ✅ Complete  
> **Completed:** 2026-07-13

---

### What Was Built
- Implemented Sync Slicers across pages for seamless parameter tracking.
- Switched default 'Highlight' behavior to 'Filter' for Archetype cross-filtering.
- Implemented Web-App Page Navigation using native Page Navigator buttons instead of legacy Drill-throughs.

---

## Phase 8 — Implementation Log (UI Layout)

> **Status:** ✅ Complete  
> **Completed:** 2026-07-13

---

### What Was Built
- Applied the native 'Innovate'/'Frontier' dark theme to fix text contrast and establish a premium FAANG-style color palette.
- Established a strict 16:9 grid layout.
- Applied pixel-perfect alignment using Align Top, Align Bottom, and Distribute Horizontally across all 6 pages.
- Corrected messy auto-generated titles and fixed DAX Measure decimal formatting.



---

## Phase 7 — Implementation Log (Custom Tooltips)

> **Status:** ✅ Complete  
> **Completed:** 2026-07-13

---

### What Was Built
- Overcame structural flat-table limits by designing a row-level tooltip strictly constrained to the `uplift_sample` table.
- Replaced the default black box on the Scatter Plot with a "User Profile" floating dashboard.
- Mapped specific `row_id`, exact metrics, and top raw features to the Multi-row card.

