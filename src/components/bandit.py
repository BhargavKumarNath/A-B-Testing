import numpy as np
import polars as pl
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class LinUCBAgent:
    """
    LinUCB Disjoint Agent. 
    Maintains separate linear models for each arm (Treatment vs Control).
    """
    def __init__(self, n_features: int, alpha: float = 1.0):
        self.alpha = alpha
        self.n_features = n_features
        # 0 = Control, 1 = Treatment
        self.A = [np.identity(n_features), np.identity(n_features)] 
        self.b = [np.zeros(n_features), np.zeros(n_features)]

    def select_arm(self, x: np.ndarray) -> int:
        """
        Calculates UCB for both arms and picks the highest.
        """
        p = []
        for arm in [0, 1]:
            try:
                theta = np.linalg.solve(self.A[arm], self.b[arm])
            except np.linalg.LinAlgError:
                theta = np.dot(np.linalg.pinv(self.A[arm]), self.b[arm])
            
            mean = np.dot(theta, x)
            
            # Variance
            A_inv = np.linalg.inv(self.A[arm])
            var = np.sqrt(np.dot(x.T, np.dot(A_inv, x)))
            
            ucb = mean + self.alpha * var
            p.append(ucb)
            
        return np.argmax(p)

    def update(self, arm: int, x: np.ndarray, reward: float):
        """Updates the linear model for the chosen arm."""
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x

class BanditSimulator:
    """
    Performs Off-Policy Evaluation (Replay Method).
    Now supports PROFIT optimization.
    """
    
    def __init__(self, df: pl.DataFrame, feature_cols: List[str], reward_col: str, treatment_col: str):
        self.df = df
        self.feature_cols = feature_cols
        self.reward_col = reward_col
        self.treatment_col = treatment_col

    def run_replay(self, sample_size: int = 100000, conversion_value: float = 10.0, cost_per_ad: float = 0.10) -> Dict:
        """
        Simulates the bandit on a stream of data optimizing for Net Profit.
        
        Reward Logic:
        - Treatment (Arm 1): (Conversion * Value) - Cost
        - Control (Arm 0):   (Conversion * Value)
        """
        logger.info(f"Starting Bandit Replay (Profit Mode) on {sample_size} events...")
        logger.info(f"Params: Value=${conversion_value}, Cost=${cost_per_ad}")
        
        stream = self.df.sample(n=sample_size, shuffle=True, seed=42)
        
        X = stream.select(self.feature_cols).to_numpy()
        conversions = stream[self.reward_col].to_numpy().reshape(-1)
        treatments = stream[self.treatment_col].to_numpy().reshape(-1)
        
        # Calculate calculated 'Profit' for every row in advance to verify logic
        # But the agent only sees the reward for the arm it picks (if matched)
        
        n_features = len(self.feature_cols)
        agent = LinUCBAgent(n_features=n_features, alpha=1.0) 
        
        aligned_counts = 0
        cumulative_profit = 0.0
        
        history_profit = []
        chosen_arms = np.empty(len(X), dtype=np.float64)  # (n,) arm picked on every event, matched or not
        matched_profits = []  # realized profit of each matched event, for the bootstrap CI
        
        for i in range(len(X)):
            x_context = X[i]
            actual_treatment = treatments[i]
            actual_conversion = conversions[i]
            
            # Calculate the ACTUAL PROFIT that occurred
            # If Treatment: Val*Conv - Cost
            # If Control:   Val*Conv
            if actual_treatment == 1:
                realized_profit = (actual_conversion * conversion_value) - cost_per_ad
            else:
                realized_profit = (actual_conversion * conversion_value)
            
            # 1. Agent chooses arm
            chosen_arm = agent.select_arm(x_context)
            chosen_arms[i] = chosen_arm
            
            # 2. Replay Logic (Match?)
            if chosen_arm == actual_treatment:
                # Scale reward to [0, 1] for stable LinUCB convergence
                min_profit = -cost_per_ad
                max_profit = conversion_value
                scaled_reward = (realized_profit - min_profit) / (max_profit - min_profit)
                
                # Update Agent with the scaled PROFIT
                agent.update(chosen_arm, x_context, scaled_reward)
                
                cumulative_profit += realized_profit
                matched_profits.append(realized_profit)
                aligned_counts += 1
                history_profit.append(cumulative_profit)
                
        avg_profit = cumulative_profit / aligned_counts if aligned_counts > 0 else 0
        
        logger.info(f"Replay Complete. Matches: {aligned_counts}/{sample_size}")
        logger.info(f"Avg Profit per User: ${avg_profit:.4f}")
        
        # Inverse-propensity score of the same decisions on EVERY event. The replay average above uses only
        # matched events; with a non-uniform logging policy (85% treated) matching is likelier when the agent
        # picks "treat", which over-weights the contexts it treats. IPW removes that bias.
        ipw = self.ipw_profit(chosen_arms, treatments.astype(np.float64), conversions.astype(np.float64),
                              conversion_value, cost_per_ad)
        logger.info(f"IPW profit per user (all events): ${ipw['bandit']:.4f}; treat-none ${ipw['treat_none']:.4f}")

        return {
            "ipw_profit": ipw,
            "chosen_arms": chosen_arms,
            "aligned_events": aligned_counts,
            "cumulative_reward": cumulative_profit,
            "avg_profit": avg_profit,
            "history_reward": history_profit,
            "matched_profits": np.asarray(matched_profits, dtype=np.float64),
        }
    @staticmethod
    def ipw_profit(arms: np.ndarray, treatment: np.ndarray, conversion: np.ndarray,
                   conversion_value: float = 10.0, cost_per_ad: float = 0.10) -> Dict[str, float]:
        """Unbiased profit per user of a 0/1 policy from a randomised log with P(treated) = mean(treatment).

        value = mean( pi*T/p*(V*Y - c) + (1-pi)*(1-T)/(1-p)*V*Y ).  All inputs have shape (n,).
        """
        assert arms.shape == treatment.shape == conversion.shape and arms.ndim == 1
        p = float(treatment.mean())
        assert 0.0 < p < 1.0, "needs both treated and control rows"

        def value(pi: np.ndarray) -> float:
            return float(np.mean(pi * treatment / p * (conversion_value * conversion - cost_per_ad)
                                 + (1 - pi) * (1 - treatment) / (1 - p) * conversion_value * conversion))

        n = len(arms)
        return {"bandit": value(arms), "treat_all": value(np.ones(n)), "treat_none": value(np.zeros(n)),
                "share_treated": float(arms.mean())}
