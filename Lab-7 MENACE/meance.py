# menace.py

import numpy as np
from typing import List, Tuple, Dict
import random
from dataclasses import dataclass

@dataclass
class GameState:
    board: List[List[int]]
    current_player: int

class MENACE:
   
    
    def __init__(self, initial_beads: int = 10):
       
        self.boxes: Dict[str, List[int]] = {}
        self.initial_beads = initial_beads
        self.moves_history: List[Tuple[str, int]] = []
        self.learning_rate = 0.1
        self.decay_factor = 0.95
        
    def get_state_key(self, board: List[List[int]]) -> str:
        """Convert board state to string key"""
        return ''.join([''.join(map(str, row)) for row in board])
    
    def initialize_box(self, state: str) -> None:
        """Initialize a new matchbox for a given state"""
        if state not in self.boxes:
            valid_moves = [i for i, val in enumerate(state) if val == '0']
            self.boxes[state] = [self.initial_beads] * len(valid_moves)
    
    def get_valid_moves(self, state: str) -> List[int]:
        """Get list of valid moves for current state"""
        return [i for i, val in enumerate(state) if val == '0']
    
    def choose_move(self, board: List[List[int]]) -> int:
        
        state = self.get_state_key(board)
        self.initialize_box(state)
        
        valid_moves = self.get_valid_moves(state)
        weights = self.boxes[state]
        
        if sum(weights) == 0:
            self.boxes[state] = [self.initial_beads] * len(valid_moves)
            weights = self.boxes[state]
        
        move = random.choices(valid_moves, weights=weights, k=1)[0]
        self.moves_history.append((state, move))
        return move
    
    def reward(self, reward_value: float) -> None:
       
        for idx, (state, move) in enumerate(self.moves_history):
            # Apply temporal difference learning
            decay = self.decay_factor ** (len(self.moves_history) - idx - 1)
            adjustment = int(reward_value * decay * self.learning_rate * self.initial_beads)
            
            move_idx = self.get_valid_moves(state).index(move)
            self.boxes[state][move_idx] = max(0, self.boxes[state][move_idx] + adjustment)
        
        self.moves_history.clear()
    
    def get_state_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistical information about current learning state"""
        stats = {}
        for state, beads in self.boxes.items():
            stats[state] = {
                'total_beads': sum(beads),
                'max_beads': max(beads),
                'min_beads': min(beads),
                'avg_beads': sum(beads) / len(beads) if beads else 0
            }
        return stats

def test_menace():
    """Test MENACE implementation"""
    menace = MENACE(initial_beads=10)
    
    # Create sample board
    board = [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]]
    
    # Make some moves
    for _ in range(5):
        move = menace.choose_move(board)
        print(f"Chosen move: {move}")
    
    # Simulate win
    menace.reward(1.0)
    
    # Print statistics
    stats = menace.get_state_statistics()
    print("\nLearning Statistics:")
    for state, stat in stats.items():
        print(f"State: {state}")
        print(f"Statistics: {stat}")

if __name__ == "__main__":
    test_menace()