import gymnasium as gym
import gymnasium_env
import time

# env = gym.make("CartPole-v1", render_mode="human")
# env = gym.make('gymnasium_env/GridWorld-v0', render_mode="human")
env = gym.make('gymnasium_env/CarAndTarget-v0', 
               render_mode="human")

# reset
observation, info = env.reset()

print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

k=0
while not episode_over:

    action = env.action_space.sample()  
    
    if k<20:
        action = 0
    elif k<40:
        action = 1
    elif k<60:
        action = 0
    elif k<80:
        action = 2
    else:
        action = 0

    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    episode_over = terminated or truncated

    # time.sleep(0.1)  # Pause for 3 seconds

    k+=1

    if k>=100:
        break
    print(k)


# print(f"Episode finished! Total reward: {total_reward}")
env.close()

print('DONE')
