import torch as th
import numpy as np

# Example 1: Action magnitude constraint
def action_magnitude_constraint(obs, actions, next_obs, rewards, dones):
    """Penalize large actions"""
    action_magnitude = th.norm(actions, dim=1)
    return th.maximum(th.zeros_like(action_magnitude), action_magnitude - 0.5)

# Example 2: Action smoothness constraint
def action_smoothness_constraint(obs, actions, next_obs, rewards, dones):
    """Penalize rapid action changes (needs action history)"""
    # Assuming obs contains previous action
    prev_actions = obs[:, -actions.shape[1]:]  # Extract prev action from obs
    action_change = th.norm(actions - prev_actions, dim=1)
    return th.maximum(th.zeros_like(action_change), action_change - 0.2)

# Example 3: State-dependent constraint
def position_constraint(obs, actions, next_obs, rewards, dones):
    """Keep agent within certain regions"""
    # Assuming first 2 dims are x,y position
    position = next_obs[:, :2]
    distance_from_center = th.norm(position, dim=1)
    return th.maximum(th.zeros_like(distance_from_center), distance_from_center - 2.0)

# Example 4: Safety constraint
def collision_constraint(obs, actions, next_obs, rewards, dones):
    """Penalize being close to obstacles"""
    # Assuming obs contains distance to nearest obstacle
    obstacle_distance = obs[:, -1]  # Last element is distance
    safety_threshold = 0.5
    return th.maximum(th.zeros_like(obstacle_distance), safety_threshold - obstacle_distance)

# Example 5: Resource constraint
def energy_constraint(obs, actions, next_obs, rewards, dones):
    """Limit energy consumption"""
    energy_cost = th.sum(actions**2, dim=1)  # Quadratic energy cost
    max_energy = 1.0
    return th.maximum(th.zeros_like(energy_cost), energy_cost - max_energy)

# Example 6: Multi-constraint (combine multiple)
def combined_constraint(obs, actions, next_obs, rewards, dones):
    """Combine multiple constraints"""
    action_cost = action_magnitude_constraint(obs, actions, next_obs, rewards, dones)
    energy_cost = energy_constraint(obs, actions, next_obs, rewards, dones)

    # Weighted combination
    total_cost = 0.7 * action_cost + 0.3 * energy_cost
    return total_cost

# Example 7: Time-varying constraint
class TimeVaryingConstraint:
    def __init__(self):
        self.timestep = 0

    def __call__(self, obs, actions, next_obs, rewards, dones):
        """Constraint that changes over time"""
        self.timestep += 1

        # Gradually relax constraint
        threshold = max(0.1, 1.0 - self.timestep / 10000)
        action_magnitude = th.norm(actions, dim=1)
        return th.maximum(th.zeros_like(action_magnitude), action_magnitude - threshold)

# Example 8: Probabilistic constraint
def probabilistic_constraint(obs, actions, next_obs, rewards, dones):
    """Constraint with uncertainty"""
    # Add noise to make constraint stochastic
    base_cost = th.norm(actions, dim=1)
    noise = th.randn_like(base_cost) * 0.1
    noisy_cost = base_cost + noise
    return th.maximum(th.zeros_like(noisy_cost), noisy_cost - 0.5)

# Usage examples:
"""
# Single constraint
model = CSAC(policy, env, cost_function=action_magnitude_constraint, constraint_threshold=0.1)

# Time-varying constraint
time_constraint = TimeVaryingConstraint()
model = CSAC(policy, env, cost_function=time_constraint, constraint_threshold=0.05)

# Multiple constraints combined
model = CSAC(policy, env, cost_function=combined_constraint, constraint_threshold=0.2)
"""