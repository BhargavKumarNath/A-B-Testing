import polars as pl
import numpy as np
from src.components.bandit import BanditSimulator, LinUCBAgent

df = pl.read_parquet('data/criteo_uplift.parquet')
sim = BanditSimulator(df, [f'f{i}' for i in range(12)], 'conversion', 'treatment')
stream = sim.df.sample(n=100000, shuffle=True, seed=42)
X = stream.select(sim.feature_cols).to_numpy()
convs = stream[sim.reward_col].to_numpy().reshape(-1)
treats = stream[sim.treatment_col].to_numpy().reshape(-1)
agent = LinUCBAgent(n_features=12)

aligns = 0
conv_aligns = 0
for i in range(len(X)):
    a = agent.select_arm(X[i])
    if a == treats[i]:
        realized_profit = (convs[i] * 10) - 0.1 if a == 1 else (convs[i] * 10)
        agent.update(a, X[i], realized_profit)
        if a == 1:
            aligns += 1
            conv_aligns += convs[i]

if aligns > 0:
    print(f'Bandit Conversion Rate (Arm 1): {conv_aligns/aligns*100:.4f}%')
else:
    print('No arm 1 matches')
