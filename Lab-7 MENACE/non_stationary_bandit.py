# non_stationary_bandit.py

import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass
import random

@dataclass
class NonStationaryBanditStats:
    """Statistics for non-stationary bandit performance"""
    average_reward: float
    cumulative_reward: float
    action_counts: List[int]
    reward_history: List[float]
    true_optimal_actions: List[int]
    selected_optimal_percentage: float

class NonStationaryBandit:
    
    def __init__(self, n_arms: int = 10, step_size: float = 0.1, epsilon: float = 0.1):
       
        self.n_arms = n_arms
        self.step_size = step_size
        self.epsilon = epsilon
        
        # Initialize action values
        self.q_true = np.zeros(n_arms)    
        self.q_estimated = np.zeros(n_arms)  
        self.counts = np.zeros(n_arms)
        
        # History tracking
        self.reward_history: List[float] = []
        self.action_history: List[int] = []
        self.optimal_action_history: List[int] = []
    
    def get_optimal_action(self) -> int:
        """Get current optimal action based on true values"""
        return np.argmax(self.q_true)
    
    def update_true_values(self) -> None:
        
        self.q_true += np.random.normal(0, 0.01, self.n_arms)
    
    def choose_action(self) -> int:
        
        if random.random() < self.epsilon:
            return random.randint(0, self.n_arms - 1)
        return np.argmax(self.q_estimated)
    
    def get_reward(self, action: int) -> float:
        
        # Update true values (random walk)
        self.update_true_values()
        
        # Generate reward with noise
        reward = np.random.normal(self.q_true[action], 1)
        return reward
    
    def update(self, action: int, reward: float) -> None:
        
        self.counts[action] += 1
        # Use constant step size for non-stationary environment
        self.q_estimated[action] += self.step_size * (reward - self.q_estimated[action])
        
        # Update history
        self.reward_history.append(reward)
        self.action_history.append(action)
        self.optimal_action_history.append(self.get_optimal_action())
    
    def get_statistics(self) -> NonStationaryBanditStats:
        
        total_actions = len(self.action_history)
        optimal_actions = sum(1 for a, opt in zip(self.action_history, 
                                                self.optimal_action_history) 
                            if a == opt)
        
        return NonStationaryBanditStats(
            average_reward=np.mean(self.reward_history),
            cumulative_reward=sum(self.reward_history),
            action_counts=self.counts.tolist(),
            reward_history=self.reward_history,
            true_optimal_actions=self.optimal_action_history,
            selected_optimal_percentage=optimal_actions / total_actions if total_actions > 0 else 0
        )
    
    def plot_performance(self) -> None:
        """Plot performance metrics"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot running average reward
        window = 100
        running_avg = np.convolve(self.reward_history, 
                                np.ones(window)/window, 
                                mode='valid')
        ax1.plot(running_avg)
        ax1.set_title('Running Average Reward')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Average Reward')
        
        # Plot true action values over time
        episodes = len(self.reward_history)
        time_steps = np.arange(episodes)
        for arm in range(self.n_arms):
            ax2.plot(time_steps[::100], 
                    [self.q_true[arm] for _ in range(0, episodes, 100)],
                    label=f'Arm {arm}')
        ax2.set_title('True Action Values Over Time')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('True Value')
        ax2.legend()
        
        # Plot optimal action selection frequency
        window = 1000
        optimal_actions = np.array([1 if a == opt else 0 
                                  for a, opt in zip(self.action_history,
                                                  self.optimal_action_history)])
        optimal_percentage = np.convolve(optimal_actions,
                                       np.ones(window)/window,
                                       mode='valid')
        ax3.plot(optimal_percentage)
        ax3.set_title('Optimal Action Selection Frequency')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()

def test_non_stationary_bandit(episodes: int = 10000) -> NonStationaryBanditStats:
    
    bandit = NonStationaryBandit()
    
    for _ in range(episodes):
        action = bandit.choose_action()
        reward = bandit.get_reward(action)
        bandit.update(action, reward)
    
    # Plot results
    bandit.plot_performance()
    return bandit.get_statistics()

if __name__ == "__main__":
    # Run single experiment
    stats = test_non_stationary_bandit()
    print("\nFinal Statistics:")
    print(f"Average Reward: {stats.average_reward:.2f}")
    print(f"Cumulative Reward: {stats.cumulative_reward:.2f}")
    print(f"Optimal Action Percentage: {stats.selected_optimal_percentage:.2%}")
    
    # Run multiple experiments
    n_experiments = 5
    all_stats = []
    
    for i in range(n_experiments):
        print(f"\nRunning experiment {i+1}/{n_experiments}")
        stats = test_non_stationary_bandit()
        all_stats.append(stats)
    
    # Calculate aggregate statistics
    avg_reward = np.mean([s.average_reward for s in all_stats])
    avg_optimal = np.mean([s.selected_optimal_percentage for s in all_stats])
    
    print("\nAggregate Statistics:")
    print(f"Average Reward across experiments: {avg_reward:.2f}")
    print(f"Average Optimal Action Percentage: {avg_optimal:.2%}")