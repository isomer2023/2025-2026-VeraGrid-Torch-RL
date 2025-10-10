import gymnasium as gym
import numpy as np
import win32com.client
from stable_baselines3 import PPO

class PowerGridEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.vg = win32com.client.Dispatch("VeraGrid.Application")
        self.vg.Visible = False
        self.vg.OpenProject(r"C:\path\to\IEEE118.vgjson")
        self.sim = self.vg.Simulation

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(118,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(10,))

    def reset(self, seed=None):
        self.sim.Reset()
        obs = self._get_state()
        return obs, {}

    def step(self, action):
        self._apply_control(action)
        self.sim.RunPowerFlow()
        obs = self._get_state()
        reward = -np.mean(np.abs(obs - 1.0))
        done = False
        return obs, reward, done, False, {}

    def _apply_control(self, action):
        # 例：控制部分发电机输出
        for i, g in enumerate(self.sim.Generators[:len(action)]):
            g.P = g.P * (1 + 0.05 * action[i])

    def _get_state(self):
        return np.array([bus.Voltage for bus in self.sim.Buses])

# 训练示例
env = PowerGridEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

