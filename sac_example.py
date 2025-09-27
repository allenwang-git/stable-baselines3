import gym
import numpy as np
import torch as th
from stable_baselines3 import SAC

# Create environment
env = gym.make("Pendulum-v1")

# Create SAC model with exact same parameters as CSAC
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    # Exact same parameters as CSAC example
    learning_rate=3e-4,
    buffer_size=100000,
    batch_size=256,
    train_freq=1,
    gradient_steps=1,
    tensorboard_log="./logs/sac/",
)

# Train the model
print("Training SAC...")
model.learn(total_timesteps=50_000)

# Save the model
model.save("./models/sac_pendulum")

# Test the trained model
print("Testing trained model...")
obs = env.reset()
total_reward = 0
total_action_magnitude = 0  # Monitor same metric as CSAC
episodes = 0

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    # Calculate action magnitude for comparison with CSAC
    action_magnitude = np.linalg.norm(action)

    total_reward += reward
    total_action_magnitude += action_magnitude

    # Optional: render environment
    # env.render()

    if done:
        episodes += 1
        obs = env.reset()
        if episodes >= 10:  # Test for 10 episodes
            break

env.close()

print(f"Average reward per episode: {total_reward / max(episodes, 1):.2f}")
print(f"Average action magnitude per step: {total_action_magnitude / max(i+1, 1):.4f}")
print("No constraints applied (standard SAC)")