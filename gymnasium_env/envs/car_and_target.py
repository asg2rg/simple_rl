from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

xlim = (-20,20)
ylim = (-20,20)
dt = 0.2
carvelocity = 20
targetvelocity = 15

car_geom = {
    'color' : (120,120,120),
    'length' : 20,
    'radius' : 10,
    'width' : 5
}

target_geom = {
    'color' : (0,0,0),
    'length' : 20,
    'radius' : 10,
    'width' : 5
}

action_lookup = ['go_straight','turn_left','turn_right']


class CarAndTargetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, max_episode_steps=100):

        # print('init')

        # constants ................................
        self.window_size = 512  # The size of the PyGame window
        self.omega = 1.0        # [radm/s] the car's angular velocity
        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        self.target = np.array([400,100,np.pi])
        self.target_speed = targetvelocity
        # initialize the state ......................
        ic = np.array([100,400,0.0])    # x,y,alpha
        self.initial_state = ic.copy()
        self.state = ic.copy()

        # observation and action spaces ..............
        self.observation_space = spaces.Box(low=np.array([0.0, -np.pi]), 
                                            high=np.array([np.inf, np.pi]), dtype=np.float32)


        self.action_space = spaces.Discrete(3)   # go_straight, turn_left, turn_right

        # other ....................................
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        dx = self.target[0] - self.state[0]
        dy = self.target[1] - self.state[1]
        distance_to_target = np.sqrt(dx**2+dy**2)
        # angle_to_target = np.arctan2(dy,dx)
        target_angle = np.arctan2(dy, dx)
        heading_error = target_angle - self.state[2]

        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        return np.array([distance_to_target, heading_error])
        # return distance_to_target, angle_to_target
        # return np.array([distance_to_target, angle_to_target])

    def _get_info(self):
        return {
            "pos_error" : self.pos_error,
            "reward" : self.reward,
            "state" : self.state
        }

    def reset(self, seed=None, options=None):

        # print('reset')

        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.step_count = 0
        x = self.np_random.uniform(50, 450)
        y = self.np_random.uniform(50, 450)
        tar_x = self.np_random.uniform(50, 450)
        tar_y = self.np_random.uniform(50, 450)
        random_goal_x = self.np_random.uniform(50, 450)
        random_goal_y = self.np_random.uniform(50, 450)
        alpha = self.np_random.uniform(-np.pi, np.pi)   
        tar_alpha = self.np_random.uniform(-np.pi, np.pi)

        # randomize initial attributes 
        self.state = np.array([x, y, alpha])
        self.target = np.array([tar_x, tar_y, tar_alpha])
        self.target_goal = np.array([random_goal_x, random_goal_y])
        # Initialize the state 
        # self.state = self.initial_state.copy()
        self.pos_error = np.sqrt(np.sum((self.target - self.state)**2))
        self.reward = 0.0
        
        # render
        if self.render_mode == "human":
            self._render_frame()

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):

        # print('step')

        # translate action
        str_action = action_lookup[action]

        # update car position
        alpha = self.state[2] 
        if str_action == 'turn_left':
            alpha += dt * self.omega
        elif str_action == 'turn_right':
            alpha -= dt * self.omega

        if alpha > np.pi:
            alpha -= 2*np.pi
        if alpha < -np.pi:
            alpha += 2*np.pi
        
        direction = self.target_goal[:2] - self.target[:2]
        dist = np.linalg.norm(direction)

        if dist < 5:
            self.target_goal = np.array([
                self.np_random.uniform(50, 450),
                self.np_random.uniform(50, 450)])
            direction = self.target_goal - self.target[:2]
            dist = np.linalg.norm(direction)

        if dist > 1e-8:
            direction = direction / dist
            step_size = self.target_speed * dt
            move = min(step_size, dist)

            self.target[0] += move * direction[0]
            self.target[1] += move * direction[1]
            self.target[2] = np.arctan2(direction[1], direction[0])
            
        self.state[0] += dt * carvelocity * np.cos(alpha)
        self.state[1] += dt * carvelocity * np.sin(alpha)
        self.state[2] = alpha
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps

        obs = self._get_obs()
        distance_to_target = obs[0]
        heading_error = obs[1]

        # Calculatepoistion error and reward
        # self.pos_error = np.sqrt(np.sum((self.target - self.state)**2))
        # self.reward = 100 - self.pos_error
        self.reward = -0.1 * distance_to_target - 0.5 * abs(heading_error)
        # termnate if target reached
        # terminated = bool( np.sum((self.target-self.state)**2)<0.1 )
        terminated = distance_to_target < 5.0

        # get observation and info
        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, self.reward, terminated, truncated, info

    def render(self):

        # print('render')

        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_circle_line(self,canvas,state,geom):
        color = geom['color']
        length = geom['length']
        base = state[:2]
        alpha = state[2]
        tip = base+length*np.array([np.cos(alpha),np.sin(alpha)])

        pygame.draw.circle(canvas,color,base,geom['radius'])
        pygame.draw.line(canvas,color,base,tip,width=geom['width'])

    def _render_frame(self):

        # print('render frame')
 
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # car and target
        self._render_circle_line(canvas,self.state,car_geom)
        self._render_circle_line(canvas,self.target,target_geom)
        
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
