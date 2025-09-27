import gym
import numpy as np
import torch as th
from stable_baselines3 import CSAC

# Define a simple constraint cost function
def action_magnitude_constraint(observations, actions, next_observations, rewards, dones):
    """
    Constraint that penalizes large action magnitudes.
    Returns cost for each transition (higher cost = more constraint violation)
    """
    # Penalize actions with magnitude > 0.5
    # Note: inputs are PyTorch tensors on device, return tensor
    action_magnitude = th.norm(actions, dim=1)
    constraint_cost = th.maximum(th.zeros_like(action_magnitude), action_magnitude - 0.5)
    return constraint_cost

# Create environment
env = gym.make("Pendulum-v1")

# Create CSAC model with constraint
model = CSAC(
    "MlpPolicy",
    env,
    verbose=1,
    # Lagrangian SAC specific parameters
    constraint_threshold=0.1,  # Allow average constraint cost up to 0.1
    lagrange_lr=1e-3,          # Learning rate for Lagrange multiplier
    initial_lagrange_multiplier=1.0,  # Initial multiplier value
    cost_function=action_magnitude_constraint,  # Our constraint function
    # Standard SAC parameters
    learning_rate=3e-4,
    buffer_size=100000,
    batch_size=256,
    train_freq=1,
    gradient_steps=1,
    tensorboard_log="./logs/csac/",
)

# Train the model
print("Training CSAC with action magnitude constraints...")
model.learn(total_timesteps=50_000)

# Save the model
model.save("./models/csac_pendulum")

# Test the trained model
print("Testing trained model...")
obs = env.reset()
total_reward = 0
total_constraint_cost = 0
episodes = 0

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    # Calculate constraint cost for monitoring (convert to tensors for consistency)
    with th.no_grad():
        obs_tensor = th.tensor(obs.reshape(1, -1), dtype=th.float32)
        action_tensor = th.tensor(action.reshape(1, -1), dtype=th.float32)
        constraint_cost = action_magnitude_constraint(
            obs_tensor, action_tensor, obs_tensor,
            th.tensor([reward]), th.tensor([done])
        )[0].item()

    total_reward += reward
    total_constraint_cost += constraint_cost

    # Optional: render environment
    # env.render()

    if done:
        episodes += 1
        obs = env.reset()
        if episodes >= 10:  # Test for 10 episodes
            break

env.close()

print(f"Average reward per episode: {total_reward / max(episodes, 1):.2f}")
print(f"Average constraint cost per step: {total_constraint_cost / max(i+1, 1):.4f}")
print(f"Constraint threshold was: {model.constraint_threshold}")