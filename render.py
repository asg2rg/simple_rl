from stable_baselines3 import PPO
from gymnasium_env.envs.car_and_target import CarAndTargetEnv

# 1. Create a renderable environment
env = CarAndTargetEnv(render_mode="human", max_episode_steps=100)

# 2. Load the saved model
model = PPO.load("simple_rl")

# 3. Reset the environment
obs, info = env.reset()

# 4. Run the trained policy
for _ in range(300):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()