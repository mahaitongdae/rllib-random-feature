"""
Code adopted from OpenAI Gym implementation
"""

import math
import gymnasium as gym
from gymnasium import spaces, logger
from gymnasium.utils import seeding
import numpy as np
# import autograd.numpy as anp
import numpy as anp
# from autograd import jacobian
from os import path
from gymnasium.error import *
from scipy.integrate import odeint
from typing import Optional, Union


class PendubotEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'render_fps': 30,
        'video.frames_per_second' : 50
    }

    screen_dim = 500
    screen = None
    clock = None
    isopen = True
    render_mode = 'human'

    def __init__(self, task="balance", initial_state=None):
        # set task
        self.task = task

        self.initial_state = initial_state

        self.last_u = None
        
        # gravity
        self.gravity = self.g = 9.8

        # pole 1
        self.m1 = 0.1
        self.l1 = self.len1 = 1.0 # actually half the pole's length
        self.L1 = 2 * self.l1
        self.I1 = (1/12) * self.m1 * (2 * self.len1)**2
        self.inertia_1= (self.I1 + self.m1 * self.len1**2)
        self.pm_len1 = self.m1 * self.len1

        # pole 2
        self.m2 = 0.1
        self.l2 = self.len2 = 1.0 # actually half the pole's length
        self.L2 = 2 * self.l2
        self.I2 = (1/12) * self.m2 * (2 * self.len2)**2
        self.inertia_2 = (self.I2 + self.m2 * self.len2**2)
        self.pm_len2 = self.m2 * self.len2

        # Other params
        self.force_mag = 0.5
        self.dt = 0.05  # seconds between state updates
        self.n_coords = 2

        # self.d1 = self.inertia_1 + self.m2 * self.L1**2
        # self.d2 = self.inertia_2
        # self.d3 = self.m2 * self.L1 * self.l2
        # self.d4 = self.m1 * self.l1 + self.m2 * self.L1
        # self.d5 = self.m2 * self.l2
        self.d1 = 0.089252
        self.d2 = 0.027630
        self.d3 = 0.023502
        self.d4 = 0.011204
        self.d5 = 0.002938

        # precompute the jacobian of the dynamics
        # self.jacobian = self._jacobian()

        # Angle at which to fail the episode
        self.x_threshold = np.pi / 4
        self.theta_threshold_radians = 12 * 2 * math.pi / 360

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(-1.0, 1.0)
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        if self.initial_state:
            self.state = self.initial_state
        else:
            self.state = self.np_random.normal(loc=0., scale=0.1, size=(2 * self.n_coords,))
            if self.task == "balance":
                self.state[0] += np.pi / 2
        self.steps_beyond_done = None
        return np.array(self.state)

    def is_done(self):
        th1, th2 = self.state[:self.n_coords]
        if self.task == "balance":
            done =  th1 < np.pi - self.theta_threshold_radians \
                    or th1 > np.pi + self.theta_threshold_radians \
                    or th2 < np.pi - self.theta_threshold_radians \
                    or th2 > np.pi + self.theta_threshold_radians
        else:
            bool = False
        return bool(done)

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # get state
        th1, th2, th1_dot, th2_dot = self.state
        th1 = self._unwrap_angle(th1)
        th2 = self._unwrap_angle(th2)

        # clip torque, update dynamics
        # u = np.clip(action, -self.force_mag, self.force_mag)
        u = action
        self.last_u = u
        # acc = self._dynamics(anp.array([th1, th2, th1_dot, th2_dot, u]))

        # integrate
        # th1_acc, th2_acc = acc

        # # update pole 1 position and angular velocity
        # th1_dot = th1_dot + self.dt * th1_acc
        # th1 = th1 + self.dt * th1_dot + 0.5 * th1_acc * self.dt**2
        #
        # # update pole 2 position and angular velocity
        # th2_dot = th2_dot + self.dt * th2_acc
        # th2 = th2 + self.dt * th2_dot + 0.5 * th2_acc * self.dt**2

        new_state = odeint(lambda s, t: self._dynamics(np.append(s, u)), self.state, [0., 0. + self.dt])[1]
        th1, th2, th1_dot, th2_dot = new_state

        # update state
        th1 = self._unwrap_angle(th1)
        th2 = self._unwrap_angle(th2)
        self.state = np.array([th1, th2, th1_dot, th2_dot])
        
        # done = self.is_done()
        terminated = False

        if not terminated:
            reward = - (th1 - np.pi / 2) ** 2 - th1_dot ** 2 - 0.01 * action ** 2
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            # reward = 1.0
            reward = - (th1 - np.pi / 2) ** 2 - th1_dot ** 2 - 0.01 * action ** 2
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_terminated += 1
            reward = -10.0

        return np.array(self.state), reward, terminated, done, {}

    def _dynamics(self, vec):
        """
        Calculate the accelerations
        """
        force = vec[-1]
        pos = vec[:self.n_coords]
        vel = vec[self.n_coords:-1]
        state = vec[:-1]
        Minv = self._Minv(pos)
        B = self._B()
        C = self._C(state)
        G = self._G(pos)
        acc = anp.dot(Minv, anp.dot(B, force) - anp.dot(C, vel.reshape((self.n_coords, 1))) - G)
        return np.append(vel, acc.flatten())

    def _F(self, vec):
        """
        Return derivative of state-space vector
        """
        qd = vec[self.n_coords:-1]
        qdd = self._dynamics(vec)
        return anp.array(list(qd) + list(qdd))

    def _M(self, pos):
        """
        Inertial Mass matrix
        """
        th1, th2 = pos
        m11 = self.d1 + self.d2 + 2 * self.d3 * anp.cos(th2)
        m21 = m12 = self.d2 + self.d3 * anp.cos(th2)
        m22 = self.d2

        mass_matrix = anp.array([[m11, m12],
                               [m21, m22]])
        return mass_matrix

    def _C(self, state):
        """
        Coriolis matrix
        """
        th1, th2, th1_dot, th2_dot = state
        c11 = -1. * self.d3 * anp.sin(th2) * th2_dot
        c12 = -self.d3 * anp.sin(th2) * (th2_dot + th1_dot)
        c21 = self.d3 * anp.sin(th2) * th1_dot
        c22 = 0.0
        return anp.array([[c11, c12],
                        [c21, c22]])

    def _G(self, pos):
        """
        Gravitional matrix
        """
        th1, th2 = pos
        g1 = self.d4 * anp.cos(th1) * self.g + self.d5 * self.g * anp.cos(th1 + th2)
        g2 = self.d5 * anp.cos(th1 + th2) * self.g
        return anp.array([[g1],
                        [g2]])

    def _B(self):
        """
        Force matrix
        """
        return anp.array([[1], [0]])

    # def _jacobian(self):
    #     """
    #     Return the Jacobian of the full state equation
    #     """
    #     return jacobian(self._F)

    def _linearize(self, vec):
        """
        Linearize the dynamics by first order Taylor expansion
        """
        f0 = self._F(vec)
        arr = self.jacobian(vec)
        A = arr[:, :-1]
        B = arr[:, -1].reshape((2 * self.n_coords, 1))
        return f0, A, B

    def _Minv(self, pos):
        """
        Invert the mass matrix
        """
        return anp.linalg.inv(self._M(pos))

    def total_energy(self, state=None):
        if state is None:
            state = self.state
        pos = state[:self.n_coords]
        vel = state[self.n_coords:]
        return self.kinetic_energy(pos, vel) + self.potential_energy(pos)

    def kinetic_energy(self, pos, vel):
        M = self._M(pos)
        vel = vel.reshape((self.n_coords, 1))
        return float(anp.dot(anp.dot(vel.T, M), vel))

    def potential_energy(self, pos):
        th1, th2 = pos
        return self.d4 * self.gravity * anp.sin(th1) + self.d5 * self.gravity * anp.sin(th1 + th2)

    def _unwrap_angle(self, theta):
        sign = (theta >=0)*1 - (theta < 0)*1
        theta = anp.abs(theta) % (2 * anp.pi)
        return sign*theta

    def integrate(self):
        """
        Integrate the equations of motion
        """
        raise NotImplementedError()

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0])
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0])
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        ## rod 2
        transformed_coords_2 = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + self.state[1])
            c = (int(c[0]) + rod_end[0], int(c[1]) + rod_end[1])
            transformed_coords_2.append(c)

        gfxdraw.aapolygon(self.surf, transformed_coords_2, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords_2, (204, 77, 77))

        rod_end_2 = (rod_length, 0)
        rod_end_2 = pygame.math.Vector2(rod_end_2).rotate_rad(self.state[0] + self.state[1])
        rod_end_2 = (int(rod_end_2[0] + rod_end[0]), int(rod_end_2[1] + rod_end[1]))
        gfxdraw.aacircle(
            self.surf, rod_end_2[0], rod_end_2[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end_2[0], rod_end_2[1], int(rod_width / 2), (204, 77, 77)
        )


        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)
        if self.last_u is not None:
            scale_img = pygame.transform.smoothscale(
                img,
                (10 * scale * np.abs(self.last_u) / 2, 10 * scale * np.abs(self.last_u) / 2),
            )
            is_flip = bool(self.last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.viewer:
            self.viewer.close()
        self.viewer = None

def energy_based_controller(env, kd = 1., kp = 1., ke = 1.5):

    th1, th2, th1_dot, th2_dot = env.state
    F_q_dotq = env.d2 * env.d3 * np.sin(th2) * (th1_dot + th2_dot) ** 2 + env.d3 ** 2 * np.cos(th2) * np.sin(th2) * th1_dot ** 2 \
               - env.d2 * env.d4 * env.g * np.cos(th1) + env.d3 * env.d5 * env.g * np.cos(th2) * np.cos(th1 + th2)
    th1_error = th1 - np.pi / 2
    energy_error = env.total_energy() - (env.d4 + env.d5) * env.g

    temp = env.d1 * env.d2 - env.d3 ** 2 * np.cos(th2) ** 2

    torque = (- kd * F_q_dotq - temp * (th1_dot + kp * th1_error)) / (temp * ke * energy_error + kd * env.d2)

    return torque


def main():
    env = PendubotEnv()
    env.reset()
    print(env.state)
    rewards = 0.
    for i in range(200):
        action = energy_based_controller(env)
        # print(action)
        _, reward, _, _ = env.step(action)

        rewards += reward
        env.render()
    print(rewards)

if __name__ == '__main__':
    main()
