from stable_baselines3.common.env_checker import check_env
import gymnasium
from gymnasium import spaces
import numpy as np
# Path: modelTimetable/DRL/myEnv.ipynb
# Implementing the environment
# Reproduction of the cartpole environment
#
# Discription:
# Create a car in a two-dimensional plane with a width of 20, and the coordinates of
# the center point are the destination of the car to reach.
#
# State:
# The state of the car is represented by the coordinates of the center point of the car.(x,y)
# Action:
# The action of the car is represented by the speed of the car.(vx,vy)
# Reward:
# The reward is the distance between the car and the destination.
# Termination:
# The car reaches the destination.(0,0)
# truncation:
# The car is out of the screen.

'''
gymnasium is the main class that we will use to create our environment.

The gymnasium class has the following methods:
__init__(): This method is used to initialize the environment. It takes the following parameters:

step(): This method is used to take an action and return the next state, reward, and whether the episode is over. 
Physical engine
- input: action
- output: observation, reward,terminated,truncated,info

reset(): This method is used to reset the environment to its initial state.
- input: None
- output: observation

render(): This method is used to render the environment:
Image engine
- input: mode(default='human','human','rgb_array','ansi','rgb_array_list)
- output: None
eg:gymnasium.make('CartPole-v0',render_mode='human')

close(): This method is used to close the environment.
'''


class MyCar(gymnasium.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        self.target_x = 0
        self.target_y = 0
        self.size = 10
        # 0:stop, 1:up, 2:down, 3:left, 4:right
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            np.array([-self.size, -self.size]), np.array([self.size, self.size]))
        self.state = None
        self.info = {}

    def step(self, action):
        """
        Updates the state of the environment based on the given action and returns the next state, reward,
        terminated status, truncated status, and additional information.

        Parameters:
            action (int): The action to be taken in the environment.

        Returns:
            state (np.ndarray): The next state of the environment.
            reward (float): The reward obtained from the action.
            terminated (bool): Indicates whether the environment has reached a terminal state.
            truncated (bool): Indicates whether the episode has been truncated.
            info (dict): Additional information about the step.

        Raises:
            AssertionError: If the action is invalid.
        """
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))
        # update the state by the action
        x, y = self.state
        if action == 0:
            x += 0
            y += 0
        elif action == 1:
            x += 0
            y += 1
        elif action == 2:
            x += 0
            y += -1
        elif action == 3:
            x += -1
            y += 0
        elif action == 4:
            x += 1
            y += 0
        # the next state
        self.state = np.array([x, y])
        self.state = self.state.astype(np.float32)
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = {}
        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None):
        self.state = np.ceil(np.random.rand(2)*2*self.size)-self.size
        self.state = self.state.astype(np.float32)
        self.counts = 0
        self.info = {}
        return self.state, self.info

    def render(self, mode='human'):
        print(self.state)

    def close(self):
        return super().close()

    def _get_reward(self):
        return -np.sqrt(self.state[0]**2+self.state[1]**2)

    def _get_terminated(self):
        x, y = self.state
        return bool(x == self.target_x and y == self.target_y)

    def _get_truncated(self):
        x, y = self.state
        return bool(x < -self.size or x > self.size or y < -self.size or y > self.size)


env = MyCar()
check_env(env, warn=True)
