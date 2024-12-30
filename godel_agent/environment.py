import gymnasium as gym
import numpy as np
from typing import Tuple, Any, List, Optional
from dataclasses import dataclass
from loguru import logger

@dataclass
class EnvironmentState:
    """Represents the state of the environment"""
    observation_history: List[Any] = None
    reward_history: List[float] = None
    step_count: int = 0

class AdaptiveEnvironment:
    """
    An environment that can dynamically adjust its complexity
    based on agent performance.
    """
    
    def __init__(self, env_id: str = "CartPole-v1", 
                 adaptation_rate: float = 0.1):
        """
        Initialize the adaptive environment.
        
        Args:
            env_id: Gymnasium environment ID
            adaptation_rate: Rate at which environment adapts to agent performance
        """
        self.env = gym.make(env_id)
        self.adaptation_rate = adaptation_rate
        self.state = EnvironmentState(
            observation_history=[],
            reward_history=[],
            step_count=0
        )
        
    def reset(self) -> Tuple[Any, dict]:
        """Reset the environment to initial state"""
        observation, info = self.env.reset()
        self.state.step_count = 0
        return observation, info
        
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take in the environment
            
        Returns:
            Tuple containing (observation, reward, terminated, truncated, info)
        """
        result = self.env.step(action)
        observation, reward, terminated, truncated, info = result
        
        # Record history
        self.state.observation_history.append(observation)
        self.state.reward_history.append(reward)
        self.state.step_count += 1
        
        return result
        
    def adapt_difficulty(self, agent_performance: float) -> None:
        """
        Adapt environment difficulty based on agent performance.
        
        Args:
            agent_performance: Float indicating agent's current performance
        """
        try:
            # This is a simple example - in practice, you'd want more sophisticated
            # adaptation logic based on your specific environment
            if hasattr(self.env, 'difficulty'):
                current_difficulty = self.env.difficulty
                
                # Increase difficulty if agent is performing well
                if agent_performance > 0.8:
                    new_difficulty = current_difficulty + self.adaptation_rate
                # Decrease difficulty if agent is struggling
                elif agent_performance < 0.2:
                    new_difficulty = current_difficulty - self.adaptation_rate
                else:
                    return
                    
                self.env.difficulty = np.clip(new_difficulty, 0.0, 1.0)
                logger.info(f"Adjusted environment difficulty to: {self.env.difficulty}")
                
        except Exception as e:
            logger.warning(f"Failed to adapt difficulty: {str(e)}")
            
    def get_state_summary(self) -> dict:
        """
        Get a summary of the environment's current state.
        
        Returns:
            Dict containing environment state summary
        """
        return {
            'total_steps': self.state.step_count,
            'average_reward': np.mean(self.state.reward_history) if self.state.reward_history else 0,
            'observation_shape': self.env.observation_space.shape,
            'action_shape': self.env.action_space.shape if hasattr(self.env.action_space, 'shape') else None
        }
        
    def close(self):
        """Clean up environment resources"""
        self.env.close()

if __name__ == "__main__":
    # Example usage
    env = AdaptiveEnvironment()
    obs, info = env.reset()
    
    for _ in range(100):
        action = env.env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
            
    print(env.get_state_summary())
    env.close()
