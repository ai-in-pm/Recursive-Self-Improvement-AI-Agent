# Getting Started with the Gödel Agent Framework

This guide will walk you through setting up and running your first self-improving AI agent.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Initial Setup

1. Clone the repository and set up the environment:
```bash
git clone https://github.com/yourusername/recursive-self-improvement.git
cd recursive-self-improvement
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure API keys:
```bash
cp .env.example .env
```
Edit `.env` file and add your API keys for various services.

## Basic Usage Example

Here's a simple example to create and train an agent:

```python
from godel_agent.app import GodelAgentManager, AgentConfig
from godel_agent.environment import AdaptiveEnvironment

# Initialize the manager
manager = GodelAgentManager()

# Create a simple optimization agent
config = AgentConfig(
    name="math_optimizer",
    objective_type="optimization"
)

# Create and get the agent ID
agent_id = manager.create_agent(config)

# Train the agent
training_results = manager.train_agent(
    agent_id,
    num_episodes=100,
    max_steps_per_episode=1000
)

# Evaluate performance
metrics = manager.evaluate_agent(agent_id)
print(f"Training Results: {training_results}")
print(f"Evaluation Metrics: {metrics}")
```

## Next Steps

- Check `02_custom_environments.md` for creating custom environments
- See `03_advanced_policies.md` for implementing advanced policies
- Explore `04_multi_agent_systems.md` for multi-agent setups
