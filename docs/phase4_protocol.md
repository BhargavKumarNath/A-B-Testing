# Phase 4 evaluation protocol: uplift model comparison and threshold sensitivity

Written 2026-11-10, before any comparison model was scored. Deviations are appended at the bottom, never edited into the sections above.

## Questions

1. Does the X-Learner rank users better than a T-Learner, a plain response model (no causal structure) and random targeting?
2. Does it earn more profit per user than those alternatives at the same share of users treated?
3. The segment rules use a CATE cut of 0.02. How much does profit per user change if the cut is moved, and is 0.02 near the best value?

## Data and split

Same file and same split as `pipeline.py`: `rng = default_rng(42)`, one uniform draw per row, `u >= 0.8` is test (the identical 2,796,971 test rows as the headline run). The old training portion is divided again so that nothing tuned on the test rows:
- fit: `u < 0.6`
- validation: `0.6 <= u < 0.8` (used only to choose a threshold in question 3)
- test: `u >= 0.8` (scored once)

All models are fitted on the fit rows only. Because the comparison models are fitted on 75% of the old training rows, the X-Learner numbers here will not match the pipeline's headline figures exactly; the pipeline is left unchanged.

## Methods compared (all LightGBM, learning rate 0.05, 100 trees, same features f0..f11)

- X-Learner: the repo's `XLearner`.
- T-Learner: the repo's `TLearner` with 100 trees (score = mu1(x) - mu0(x)).
- Response model: one binary classifier of conversion on the pooled fit rows, ignoring treatment. Score = P(conversion | x). This is the usual non-causal targeting baseline.
- Random targeting: an analytic policy that treats each user with probability k. Its expected value at budget k is computed exactly by the estimator below, with no random draw.

## Metrics (test rows)

Policy value is the inverse-propensity-weighted profit per user. The assignment is a randomised trial with p = share of treated rows in the test split. For a 0/1 policy pi(x), with value V = 10 per conversion and cost c = 0.10 per ad (the pipeline's values):

    value = mean( pi * T/p * (V*Y - c) + (1 - pi) * (1 - T)/(1 - p) * V*Y )

This is unbiased for the profit per user the policy would earn. Reported:
- profit per user when treating the top k of {10, 20, 30, 50}% by score, per method, with 95% CIs;
- paired differences against the X-Learner, with 95% CIs;
- AUUC (the repo's `UpliftEvaluator` curve) per method, with the area against the random diagonal.

Intervals: bootstrap over test rows. Profit values use 1,000 resamples; AUUC uses 200 (each resample needs a sort). Seed 0 for the bootstrap. The same row indices are used for every method in a resample, so differences are paired.

## Threshold sensitivity

Using the X-Learner only. Policy: treat if predicted CATE > tau. Grid, fixed now: tau in {0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06}. For each tau: share treated and profit per user on test, with CI. Theory says treating pays when CATE x V > c, which is CATE > 0.01 at these prices; the grid is chosen to straddle that. The data-driven choice: the tau on the grid with the highest IPW profit on the validation rows, then evaluated on test. Whether 0.02 is within the CI of that choice is reported as measured. 0.02 is not changed in the code or on the page unless the result calls for it, and any change is disclosed.

## Honesty rules

- One run per configuration; the numbers from that run are reported, favourable or not.
- No hyper-parameter, grid or budget is changed after seeing a result.
- Every result file is written under `results/phase4/` by the script, never typed in.
- The pipeline's headline numbers (bandit +$0.072) are unchanged; the page keeps saying which split they come from.

## Deviations

1. **Added after the main comparison table was seen (2026-11-10).** The best IPW policy value in the main table is about +$0.024 per user, while the pipeline's headline bandit figure is +$0.072. The replay method averages profit over matched events only; with an 85/15 logging policy, the match probability differs by arm, so contexts where the bandit picks "treat" are over-weighted. Added question 4: *what does the same bandit run earn under the IPW estimator used above?* `experiments/phase4/p4_bandit_ipw.py` replays the identical stream (test rows u >= 0.8, 1,000,000 events sampled with seed 42, LinUCB alpha 1.0, exactly the pipeline code), records the arm chosen on every event, and scores it with the IPW formula on all 1,000,000 events (bootstrap 1,000, seed 0). It also reports the replay number from the same run, and the IPW value of treat-all and treat-none on that sample. Run once. This question was not in the original plan; it was added because of a discrepancy in the results already seen.
