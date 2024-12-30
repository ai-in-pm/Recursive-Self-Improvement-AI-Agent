import os
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from loguru import logger
from .agent import GodelAgent
from .environment import AdaptiveEnvironment

@dataclass
class AgentConfig:
    """Configuration for a Gödel Agent instance"""
    name: str
    objective_type: str
    initial_policy: Optional[str] = None
    env_id: Optional[str] = None

class GodelAgentManager:
    """
    Manages the lifecycle of multiple Gödel Agents, including creation,
    training, and evaluation.
    """
    
    def __init__(self, workspace_dir: str = "./workspace"):
        """
        Initialize the agent manager.
        
        Args:
            workspace_dir: Directory for storing agent data and logs
        """
        self.workspace_dir = workspace_dir
        self.agents: Dict[str, GodelAgent] = {}
        self.environments: Dict[str, AdaptiveEnvironment] = {}
        
        # Create workspace directory if it doesn't exist
        os.makedirs(workspace_dir, exist_ok=True)
        os.makedirs(os.path.join(workspace_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(workspace_dir, "agents"), exist_ok=True)
        
        # Setup logging
        logger.add(
            os.path.join(workspace_dir, "logs", "agent_manager.log"),
            rotation="1 day"
        )
        
    def create_agent(self, config: AgentConfig) -> str:
        """
        Create a new Gödel Agent instance.
        
        Args:
            config: Configuration for the new agent
            
        Returns:
            str: ID of the created agent
        """
        try:
            # Create objective function based on type
            if config.objective_type == "environment":
                if not config.env_id:
                    raise ValueError("env_id required for environment objective")
                    
                env = AdaptiveEnvironment(config.env_id)
                self.environments[config.name] = env
                
                def objective_function(obs, action):
                    _, reward, _, _, _ = env.step(action)
                    return reward
                    
            elif config.objective_type == "optimization":
                # Example optimization objective
                def objective_function(obs, action):
                    return -((obs - action) ** 2)
            else:
                raise ValueError(f"Unknown objective type: {config.objective_type}")
            
            # Create agent
            agent = GodelAgent(
                objective_function=objective_function,
                initial_policy=eval(config.initial_policy) if config.initial_policy else None
            )
            
            self.agents[config.name] = agent
            
            # Save configuration
            self._save_agent_config(config)
            
            logger.info(f"Created agent: {config.name}")
            return config.name
            
        except Exception as e:
            logger.error(f"Failed to create agent: {str(e)}")
            raise
            
    def train_agent(self, agent_id: str, 
                   num_episodes: int = 100,
                   max_steps_per_episode: int = 1000) -> Dict[str, Any]:
        """
        Train a specific agent.
        
        Args:
            agent_id: ID of the agent to train
            num_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            Dict containing training results
        """
        try:
            agent = self.agents.get(agent_id)
            if not agent:
                raise ValueError(f"Agent not found: {agent_id}")
                
            env = self.environments.get(agent_id)
            if env:
                # Training with environment
                for episode in range(num_episodes):
                    obs, _ = env.reset()
                    episode_reward = 0
                    
                    for step in range(max_steps_per_episode):
                        action = agent.policy(obs)
                        next_obs, reward, done, truncated, _ = env.step(action)
                        episode_reward += reward
                        
                        if done or truncated:
                            break
                            
                        obs = next_obs
                    
                    # Optimize agent based on episode performance
                    agent.optimize([obs], max_iterations=10)
                    
                    logger.info(f"Episode {episode} reward: {episode_reward}")
                    
            else:
                # Training with optimization objective
                observations = [
                    [float(i) / max_steps_per_episode] 
                    for i in range(max_steps_per_episode)
                ]
                agent.optimize(observations, max_iterations=num_episodes)
            
            return agent.introspect()
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
            
    def evaluate_agent(self, agent_id: str, 
                      num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate an agent's performance.
        
        Args:
            agent_id: ID of the agent to evaluate
            num_episodes: Number of evaluation episodes
            
        Returns:
            Dict containing evaluation metrics
        """
        try:
            agent = self.agents.get(agent_id)
            if not agent:
                raise ValueError(f"Agent not found: {agent_id}")
                
            env = self.environments.get(agent_id)
            total_reward = 0.0
            
            if env:
                # Evaluate in environment
                for _ in range(num_episodes):
                    obs, _ = env.reset()
                    episode_reward = 0
                    done = False
                    
                    while not done:
                        action = agent.policy(obs)
                        obs, reward, done, _, _ = env.step(action)
                        episode_reward += reward
                        
                    total_reward += episode_reward
            else:
                # Evaluate optimization performance
                observations = [[float(i) / 100] for i in range(100)]
                total_reward = agent._evaluate_policy(observations)
            
            metrics = {
                'average_reward': total_reward / num_episodes,
                'policy_version': agent.state.policy_version
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
            
    def _save_agent_config(self, config: AgentConfig) -> None:
        """Save agent configuration to disk"""
        config_path = os.path.join(
            self.workspace_dir, "agents", f"{config.name}_config.json"
        )
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
            
    def load_agent(self, agent_id: str) -> None:
        """Load agent configuration from disk"""
        config_path = os.path.join(
            self.workspace_dir, "agents", f"{agent_id}_config.json"
        )
        with open(config_path, 'r') as f:
            config = AgentConfig(**json.load(f))
        self.create_agent(config)

if __name__ == "__main__":
    # Example usage
    manager = GodelAgentManager()
    
    # Create an optimization agent
    config = AgentConfig(
        name="optimizer_agent",
        objective_type="optimization"
    )
    agent_id = manager.create_agent(config)
    
    # Train and evaluate
    training_results = manager.train_agent(agent_id)
    eval_metrics = manager.evaluate_agent(agent_id)
    
    print("Training results:", training_results)
    print("Evaluation metrics:", eval_metrics)
