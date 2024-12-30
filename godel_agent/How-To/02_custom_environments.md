# Creating Custom Environments

Learn how to create custom environments for your Gödel Agent to interact with.

## Environment Structure

Custom environments should inherit from `AdaptiveEnvironment` and implement the following methods:

```python
from godel_agent.environment import AdaptiveEnvironment
import numpy as np

class CustomMathEnvironment(AdaptiveEnvironment):
    def __init__(self, difficulty: float = 0.5):
        super().__init__()
        self.difficulty = difficulty
        self.current_problem = None
        
    def reset(self):
        """Reset the environment and return initial observation"""
        self.current_problem = self._generate_problem()
        return self.current_problem, {}
        
    def step(self, action):
        """Process agent's action and return next state"""
        reward = self._calculate_reward(action)
        next_problem = self._generate_problem()
        done = reward > 0.95  # Success threshold
        
        return next_problem, reward, done, False, {}
        
    def _generate_problem(self):
        """Generate a math problem based on difficulty"""
        if self.difficulty < 0.3:
            # Simple addition
            a = np.random.randint(1, 10)
            b = np.random.randint(1, 10)
            return {'type': 'add', 'values': [a, b]}
        elif self.difficulty < 0.7:
            # Multiplication
            a = np.random.randint(2, 12)
            b = np.random.randint(2, 12)
            return {'type': 'multiply', 'values': [a, b]}
        else:
            # Quadratic equation
            a = np.random.randint(1, 5)
            b = np.random.randint(-10, 10)
            c = np.random.randint(-10, 10)
            return {'type': 'quadratic', 'values': [a, b, c]}
            
    def _calculate_reward(self, action):
        """Calculate reward based on agent's answer"""
        problem = self.current_problem
        if problem['type'] == 'add':
            correct = sum(problem['values'])
        elif problem['type'] == 'multiply':
            correct = problem['values'][0] * problem['values'][1]
        else:  # quadratic
            a, b, c = problem['values']
            # Reward based on how close the root approximation is
            correct = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
            
        return 1.0 / (1.0 + abs(action - correct))
```

## Using Custom Environments

Here's how to use your custom environment:

```python
from godel_agent.app import GodelAgentManager, AgentConfig

# Initialize manager and environment
manager = GodelAgentManager()
env = CustomMathEnvironment(difficulty=0.3)

# Create agent config
config = AgentConfig(
    name="math_solver",
    objective_type="environment",
    env_id="custom_math"  # Your environment identifier
)

# Create and train agent
agent_id = manager.create_agent(config)
manager.train_agent(agent_id, num_episodes=1000)
```

## Advanced Features

### Dynamic Difficulty Adjustment

The environment can adapt its difficulty based on agent performance:

```python
def adapt_difficulty(self, agent_performance: float):
    """Adjust difficulty based on agent's performance"""
    if agent_performance > 0.8:
        self.difficulty = min(1.0, self.difficulty + 0.1)
    elif agent_performance < 0.2:
        self.difficulty = max(0.0, self.difficulty - 0.1)
```

### Custom Observation Spaces

Define complex observation spaces for your environment:

```python
from gymnasium import spaces

class AdvancedEnvironment(AdaptiveEnvironment):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict({
            'state': spaces.Box(low=-np.inf, high=np.inf, shape=(10,)),
            'context': spaces.Discrete(5)
        })
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,))
```
