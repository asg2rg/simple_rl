from stable_baselines3 import PPO
from gymnasium_env.envs.car_and_target import CarAndTargetEnv

env = CarAndTargetEnv(render_mode="human", max_episode_steps=100)

# Load the saved model
model = PPO.load("")

obs, info = env.reset()

for _ in range(500):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()