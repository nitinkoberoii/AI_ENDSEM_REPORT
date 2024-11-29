# binary_bandit.py

import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import random

@dataclass
class BanditStats:
    """Statistics for bandit performance"""
    total_reward: float
    average_reward: float
    action_counts: List[int]
    optimal_action_percentage: float

class BinaryBandit:
    
    def __init__(self, epsilon: float = 0.1, learning_rate: float = 0.1):
        
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.values = [0.0, 0.0]  
        self.counts = [0, 0]     
        self.optimal_action = None
        self.total_reward = 0
        self.action_history: List[int] = []
        self.reward_history: List[float] = []
        
    def choose_action(self) -> int:
        
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        return np.argmax(self.values)
    
    def update(self, action: int, reward: float) -> None:
        
        self.counts[action] += 1
        self.values[action] += self.learning_rate * (reward - self.values[action])
        self.total_reward += reward
        self.action_history.append(action)
        self.reward_history.append(reward)
        
        # Update optimal action
        if self.optimal_action is None or self.values[action] > self.values[self.optimal_action]:
            self.optimal_action = action
    
    def get_statistics(self) -> BanditStats:
        
        total_actions = sum(self.counts)
        optimal_actions = sum(1 for a in self.action_history 
                            if a == self.optimal_action)
        
        return BanditStats(
            total_reward=self.total_reward,
            average_reward=self.total_reward / total_actions if total_actions > 0 else 0,
            action_counts=self.counts.copy(),
            optimal_action_percentage=optimal_actions / total_actions if total_actions > 0 else 0
        )
    
    def plot_performance(self) -> None:
        """Plot performance metrics"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot running average reward
        window = 100
        running_avg = np.convolve(self.reward_history, 
                                np.ones(window)/window, 
                                mode='valid')
        ax1.plot(running_avg)
        ax1.set_title('Running Average Reward')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Average Reward')
        
        # Plot action selection frequencies
        ax2.bar(['Action 0', 'Action 1'], self.counts)
        ax2.set_title('Action Selection Frequencies')
        ax2.set_ylabel('Number of Selections')
        
        plt.tight_layout()
        plt.show()

def test_binary_bandit(episodes: int = 1000) -> BanditStats:
    
    # Initialize bandit with true probabilities
    true_probs = [0.3, 0.7]  # Action 1 is optimal
    bandit = BinaryBandit(epsilon=0.1, learning_rate=0.1)
    
    for _ in range(episodes):
        action = bandit.choose_action()
        # Generate reward based on true probabilities
        reward = np.random.binomial(1, true_probs[action])
        bandit.update(action, reward)
    
    # Plot results
    bandit.plot_performance()
    return bandit.get_statistics()

if __name__ == "__main__":

    stats = test_binary_bandit(1000)
    print("\nFinal Statistics:")
    print(f"Total Reward: {stats.total_reward:.2f}")
    print(f"Average Reward: {stats.average_reward:.2f}")
    print(f"Action Counts: {stats.action_counts}")
    print(f"Optimal Action Percentage: {stats.optimal_action_percentage:.2%}")

class BinaryBanditExperiment:
    """
    Class to run multiple experiments with binary bandits
    """
    
    def __init__(self, n_experiments: int, episodes_per_experiment: int):
        self.n_experiments = n_experiments
        self.episodes_per_experiment = episodes_per_experiment
        self.results: List[BanditStats] = []
    
    def run_experiments(self) -> None:
        """Run multiple experiments and collect statistics"""
        for _ in range(self.n_experiments):
            stats = test_binary_bandit(self.episodes_per_experiment)
            self.results.append(stats)
    
    def get_aggregate_statistics(self) -> dict:
        """Calculate aggregate statistics across all experiments"""
        return {
            'avg_total_reward': np.mean([r.total_reward for r in self.results]),
            'avg_reward_per_episode': np.mean([r.average_reward for r in self.results]),
            'avg_optimal_action_pct': np.mean([r.optimal_action_percentage for r in self.results]),
            'std_total_reward': np.std([r.total_reward for r in self.results]),
            'std_optimal_action_pct': np.std([r.optimal_action_percentage for r in self.results])
        }

# Additional test code for multiple experiments
if __name__ == "__main__":
    experiment = BinaryBanditExperiment(n_experiments=10, episodes_per_experiment=1000)
    experiment.run_experiments()
    agg_stats = experiment.get_aggregate_statistics()
    
    print("\nAggregate Statistics across Experiments:")
    for key, value in agg_stats.items():
        print(f"{key}: {value:.3f}")