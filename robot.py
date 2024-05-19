import numpy as np
import pygame
import random
import math
# 初始化pygame
pygame.init()
# 设置屏幕大小
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))

# 设置标题
pygame.display.set_caption("2D Robot Game")
robot_image = pygame.image.load('robot.png')
original_width, original_height = robot_image.get_size()
scaled_width = original_width // 8
scaled_height = original_height // 8
robot_image = pygame.transform.scale(
    robot_image, (scaled_width, scaled_height))
robot_image = robot_image.convert_alpha()  # 转换图片以提高渲染效率并保留透明度

robot_image_rect = robot_image.get_rect()
robot_image_rect.center = (screen_width // 2, screen_height // 2)
# 颜色定义
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

DELTA_T = 1/10 * 5
MAX_V = 5
ACCELERATE = 1


class Robot:
    def __init__(self, screen) -> None:
        self.screen_width, self.screen_hight = screen.get_size()
        self.radius = 20
        self.color = BLUE
        self.init_pos = np.array([screen_width // 2, screen_height // 2])
        self.init_theta = 0
        self.vm = MAX_V
        self.vl = 0
        self.vr = 0
        self.vc = (self.vl + self.vr) / 2
        self.w = 0
        self.a = ACCELERATE

        self.pos = self.init_pos
        self.theta = self.init_theta
        self.image_theta = self.init_theta

    def update_velocity(self, key):
        # 更新机器人速度
        # if keys[pygame.K_UP]:
        #     robot.vl += robot.a*DELTA_T
        #     robot.vr += robot.a*DELTA_T
        # if keys[pygame.K_DOWN]:
        #     robot.vl -= robot.a*DELTA_T
        #     robot.vr -= robot.a*DELTA_T
        # if keys[pygame.K_LEFT]:
        #     self.vr += self.a*DELTA_T
        # if keys[pygame.K_RIGHT]:
        #     self.vl += self.a*DELTA_T
        if key == 0:
            self.vl += self.a*DELTA_T
        if key == 1:
            self.vl -= self.a*DELTA_T
        if key == 2:
            self.vr += self.a*DELTA_T
        if key == 3:
            self.vr -= self.a*DELTA_T
        if key == 4:
            pass
        # 限制速度不超过最大速度
        if self.vl >= self.vm:
            self.vl = self.vm
        # if self.vl < -self.vm:
        #     self.vl = -self.vm
        if self.vl < 0:
            self.vl = 0
        if self.vr >= self.vm:
            self.vr = self.vm
        # if self.vr < -self.vm:
        #     self.vr = -self.vm
        if self.vr < 0:
            self.vr = 0
        self.vc = (self.vl + self.vr) / 2
        self.w = (self.vr-self.vl)/self.radius

    def update_position(self):
        vc, theta = self.vc, self.theta
        self.pos[0] += vc * math.cos(theta) * DELTA_T
        # 注意因为坐标系方向改变，变化量的符号要仔细考虑
        self.pos[1] -= vc * math.sin(theta) * DELTA_T
        # 限制新位置在屏幕内
        if self.pos[0] <= 0:
            self.pos[0] = 0
        if self.pos[0] >= self.screen_width:
            self.pos[0] = self.screen_width
        if self.pos[1] <= 0:
            self.pos[1] = 0
        if self.pos[1] >= self.screen_hight:
            self.pos[1] = self.screen_hight
        self.theta += (self.w * DELTA_T)
        self.theta = normalize_angle(self.theta)

    def show(self, screen):
        # 旋转图片并更新其矩形区域的中心点
        self.image_theta = math.degrees(self.theta) - 90
        rotated_image = pygame.transform.rotate(
            robot_image, self.image_theta)
        robot_image_rect.center = self.pos
        rotated_image_rect = rotated_image.get_rect(
            center=robot_image_rect.center)

        # 绘制旋转后的图片
        screen.blit(rotated_image, rotated_image_rect.topleft)
        # pygame.draw.rect(screen, robot.color,
        #                  (robot.pos[0], robot.pos[1], robot.width, robot.height))
        # pygame.draw.line(screen, RED,
        #                  self.pos, start_head, 1)

        # 目标点的属性


info_font = pygame.font.SysFont(None, 36)  # 使用36号字体大小


def normalize_angle(angle):
    # 将角度规范化到 -2π 到 +2π 范围内
    return (angle + 4 * math.pi) % (2 * math.pi)


def show_data(name, pos, digit=3, *content):
    content_str = ''
    for x in content:
        content_str += (str(round(x, digit)) + ' ')
    name = info_font.render(
        f"{name}: ({content_str})", True, RED)
    screen.blit(name, pos)  # 在机器人坐标下方绘制目标点坐标


def check_circle_to_circle_collision():
    ...


def check_circle_to_rect_collision():
    ...
