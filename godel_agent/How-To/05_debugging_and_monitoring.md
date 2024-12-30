# Debugging and Monitoring Gödel Agents

Learn how to debug, monitor, and analyze your self-improving agents.

## Logging and Monitoring

### 1. Setting Up Advanced Logging

```python
from loguru import logger
import sys
import os

def setup_logging(workspace_dir: str):
    """Configure comprehensive logging system"""
    
    # Remove default handler
    logger.remove()
    
    # Add file handlers
    log_dir = os.path.join(workspace_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Main debug log
    logger.add(
        os.path.join(log_dir, "debug.log"),
        level="DEBUG",
        rotation="1 day"
    )
    
    # Error log
    logger.add(
        os.path.join(log_dir, "error.log"),
        level="ERROR",
        rotation="1 week"
    )
    
    # Performance metrics
    logger.add(
        os.path.join(log_dir, "performance.log"),
        level="INFO",
        filter=lambda record: "performance" in record["extra"],
        rotation="1 day"
    )
    
    # Console output
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"
    )
```

### 2. Performance Monitoring

```python
import time
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class PerformanceMetrics:
    episode: int
    reward: float
    policy_version: int
    execution_time: float
    memory_usage: float

class PerformanceMonitor:
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        
    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics"""
        self.metrics_history.append(metrics)
        logger.bind(performance=True).info(
            f"Episode {metrics.episode}: "
            f"Reward={metrics.reward:.4f}, "
            f"PolicyV{metrics.policy_version}, "
            f"Time={metrics.execution_time:.2f}s, "
            f"Memory={metrics.memory_usage:.2f}MB"
        )
        
    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends"""
        if not self.metrics_history:
            return {}
            
        # Calculate key statistics
        rewards = [m.reward for m in self.metrics_history]
        times = [m.execution_time for m in self.metrics_history]
        
        return {
            'avg_reward': sum(rewards) / len(rewards),
            'max_reward': max(rewards),
            'avg_time': sum(times) / len(times),
            'total_policies': self.metrics_history[-1].policy_version
        }
```

## Debugging Tools

### 1. Policy Debugger

```python
class PolicyDebugger:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.breakpoints = set()
        self.action_history = []
        
    def add_breakpoint(self, condition: callable):
        """Add debugging breakpoint"""
        self.breakpoints.add(condition)
        
    def debug_action(self, observation: Any, action: Any):
        """Debug single action"""
        self.action_history.append((observation, action))
        
        # Check breakpoints
        for condition in self.breakpoints:
            if condition(observation, action):
                self._handle_breakpoint(observation, action)
                
    def _handle_breakpoint(self, observation, action):
        """Handle triggered breakpoint"""
        logger.debug(
            f"Breakpoint triggered:\n"
            f"Observation: {observation}\n"
            f"Action: {action}\n"
            f"Recent history: {self.action_history[-5:]}"
        )
```

### 2. Memory Analyzer

```python
import psutil
import gc
from typing import Dict, List

class MemoryAnalyzer:
    def __init__(self):
        self.memory_snapshots: List[Dict] = []
        
    def take_snapshot(self):
        """Take memory usage snapshot"""
        snapshot = {
            'time': time.time(),
            'process': psutil.Process().memory_info().rss / 1024 / 1024,
            'python': self._get_python_memory()
        }
        self.memory_snapshots.append(snapshot)
        
    def _get_python_memory(self) -> Dict:
        """Get detailed Python memory usage"""
        gc.collect()
        return {
            'objects': len(gc.get_objects()),
            'refs': len(gc.get_referrers()),
            'garbage': len(gc.garbage)
        }
        
    def analyze_memory_usage(self):
        """Analyze memory usage patterns"""
        if not self.memory_snapshots:
            return {}
            
        # Calculate memory trends
        process_memory = [s['process'] for s in self.memory_snapshots]
        python_objects = [s['python']['objects'] for s in self.memory_snapshots]
        
        return {
            'peak_memory': max(process_memory),
            'avg_memory': sum(process_memory) / len(process_memory),
            'object_growth': python_objects[-1] - python_objects[0]
        }
```

## Usage Example

```python
from godel_agent.app import GodelAgentManager, AgentConfig

# Setup logging
setup_logging("./workspace")

# Initialize monitoring
monitor = PerformanceMonitor()
memory_analyzer = MemoryAnalyzer()

# Create agent
manager = GodelAgentManager()
config = AgentConfig(name="debug_agent", objective_type="optimization")
agent_id = manager.create_agent(config)

# Setup debugging
debugger = PolicyDebugger(agent_id)
debugger.add_breakpoint(
    lambda obs, act: act > 100  # Break on large actions
)

# Training loop with monitoring
for episode in range(100):
    start_time = time.time()
    memory_analyzer.take_snapshot()
    
    # Train agent
    results = manager.train_agent(agent_id, num_episodes=1)
    
    # Record metrics
    metrics = PerformanceMetrics(
        episode=episode,
        reward=results['reward'],
        policy_version=results['policy_version'],
        execution_time=time.time() - start_time,
        memory_usage=memory_analyzer.memory_snapshots[-1]['process']
    )
    monitor.record_metrics(metrics)
    
# Analyze results
performance_trends = monitor.analyze_trends()
memory_analysis = memory_analyzer.analyze_memory_usage()

logger.info(f"Performance Analysis: {performance_trends}")
logger.info(f"Memory Analysis: {memory_analysis}")
```
