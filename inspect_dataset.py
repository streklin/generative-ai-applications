import minari
import numpy as np

dataset = minari.load_dataset('D4RL/relocate/human-v2', download=True)
episodes = list(dataset.iterate_episodes())
returns = [np.sum(ep.rewards) for ep in episodes]
print(f"Max return: {np.max(returns)}")
print(f"Mean return: {np.mean(returns)}")
print(f"Min return: {np.min(returns)}")
print(f"Episode lengths: {[len(ep.rewards) for ep in episodes]}")
