from gymnasium_env.envs.grid_world import GridWorldEnv
from gymnasium_env.envs.car_and_target import CarAndTargetEnv

from gymnasium.envs.registration import register

register(
    id="gymnasium_env/GridWorld-v0",
    entry_point="gymnasium_env.envs:GridWorldEnv",
)


register(
    id="gymnasium_env/CarAndTarget-v0",
    entry_point="gymnasium_env.envs:CarAndTargetEnv"
)

