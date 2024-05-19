from stable_baselines3.common.env_checker import check_env
import pygame
import numpy as np
import math
from robot import Robot
import random
import gymnasium
from gymnasium import spaces
import numpy as np
# 颜色定义
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

STEPS_LIMIT = 300


class RobotEnv(gymnasium.Env):
    def __init__(self, screen_width, screen_height):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Robot Reinforcement Learning")
        self.robot = Robot(self.screen)
        self.score = 0
        self.info_font = pygame.font.SysFont(None, 36)
        # note 定义动作空间 0:vl+ 1:vl- 2:vr+ 3:vr- 4:do nothing
        self.action_space = spaces.Discrete(5)
        # note 定义状态空间 [robot_pos[0],robot_pos[1],target_pos[0],target_pos[1],vl,vr,theta,distance]
        self.observation_space = spaces.Box(
            np.array([-screen_width, -screen_height, -screen_width, -screen_height, -self.robot.vm, -self.robot.vm, 0, 0]), np.array([screen_width, screen_height, screen_width, screen_height, self.robot.vm, self.robot.vm, 2*math.pi, (2**0.5)*self.screen_width]))  # 定义观察空间
        self.target_pos = np.array([screen_width // 3, screen_height // 3])
        self.target_radius = 10

        self.distance = 0
        self.steps_count = 0
        self.left_wheel_speeds = []

        self.right_wheel_speeds = []
        self.clock = pygame.time.Clock()  # 创建Clock对象

    def reset(self, seed=None):
        self.robot.pos = np.array(
            [random.randint(0, self.screen_width), random.randint(0, self.screen_height)])
        self.target_pos = np.array(
            [random.randint(0, self.screen_width), random.randint(0, self.screen_height)])
        self.score = 0
        self.info = {}
        self.state = self._get_observation()
        return (self.state, self.info)

    def step(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))
        self.steps_count += 1
        self.robot.update_velocity(action)
        self.robot.update_position()

        reward = self._get_reward()
        self.state = self._get_observation()
        # 判断是否游戏结束（例如，机器人移出屏幕）
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        self.info = {}
        self.left_wheel_speeds.append(self.robot.vl)
        self.right_wheel_speeds.append(self.robot.vr)
        return self.state, reward, terminated, truncated, self.info
        # return self._get_observation(), reward, done, info

    def _get_observation(self):
        # 根据需要返回环境的观察值
        return np.array(
            [self.robot.pos[0], self.robot.pos[1], self.target_pos[0], self.target_pos[1], self.robot.vl, self.robot.vr, self.robot.theta, self.distance]).astype(np.float32)

    # def _generate_target_pos(self):
        随机生成目标点的位置
        # return np.array([random.randint(0, self.screen_width), random.randint(0, self.screen_height)])

    def render(self, mode='human'):
        # 清除屏幕
        self.screen.fill((255, 255, 255))
        # 绘制机器人
        self.robot.show(self.screen)
        # 绘制目标点
        pygame.draw.circle(self.screen, RED,
                           self.target_pos, self.target_radius)
        # 显示分数
        score_text = self.info_font.render(
            f"Score: {self.score}", True, (0, 0, 255))
        self.screen.blit(score_text, (10, 10))
        # 更新屏幕显示
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
        pygame.display.flip()
        self.clock.tick(30)

    def _get_terminated(self):
        if self.score == 1:
            pygame.quit()
            return True
        return False

    def touch_wall(self):
        return (self.robot.pos[0] <= 0 or self.robot.pos[0] >= self.screen_width or self.robot.pos[1] <= 0 or self.robot.pos[1] >= self.screen_height)

    def _get_truncated(self):
        # 限制新位置在屏幕内
        if self.touch_wall():
            # print("touch wall", self.steps_count)
            return True
        if self.steps_count >= STEPS_LIMIT:
            self.steps_count = 0
            return True
        return False

    def _get_reward(self):
        self.distance, collision = self.check_circle_to_circle_collision()
        distance_percentage = self.distance / self.screen_width * 100

        if collision:
            self.score += 1
            return 100
        elif distance_percentage <= 25:
            return -0.1
        elif distance_percentage <= 50:
            return -0.25
        elif distance_percentage <= 75:
            return -0.5
        else:
            return -1

    def check_circle_to_circle_collision(self):
        a = (self.robot.pos - self.target_pos)**2
        distance = a.sum()**0.5
        return distance, distance <= (self.robot.radius+self.target_radius)

    def close(self):
        # 关闭环境
        pygame.quit()


# check
# env = RobotEnv(screen_width=400, screen_height=400)
# check_env(env, warn=True)
