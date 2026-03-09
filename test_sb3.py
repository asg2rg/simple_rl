import gymnasium as gym
# import gymnasium_env
from gymnasium_env.envs.car_and_target import CarAndTargetEnv
from stable_baselines3 import A2C

# env = gym.make('gymnasium_env/CarAndTarget-v0', 
#                render_mode="human",
#                max_episode_steps=40)

env = CarAndTargetEnv(render_mode="human")                     

model = A2C("MlpPolicy", env,
            learning_rate=0.5,
            gamma=0.8,
            n_steps = 5)

print("----- TRAINING ------")
model.learn(total_timesteps=50,
            progress_bar=1)

print("----- RUNNING ------")


# obs, info = env.reset()
# for i in range(100):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, _, info = env.step(action)

#     # vec_env.render("human")
#     # VecEnv resets automatically
#     if done:
#       obs = env.reset()

# env.close()



vec_env = model.get_env()
obs = vec_env.reset()
for i in range(100):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    if done:
      obs = vec_env.reset()