import gymnasium as gym
import numpy as np
import veragrid as vg  # VeraGrid 5.5.3
from stable_baselines3 import PPO

class PowerGridEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.sim = vg.Session()
        self.sim.load_case("IEEE118.vgjson")  # 载入118节点系统
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(n_obs,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(n_act,))

    def reset(self, seed=None):
        self.sim.reset()
        state = self._get_state()
        return state, {}

    def step(self, action):
        # 1. 将强化学习动作转换为控制信号（调节发电机、有载变压器等）
        self.sim.apply_control(action)
        # 2. 运行潮流或动态步
        self.sim.run_step()
        # 3. 获取新的状态
        obs = self._get_state()
        # 4. 计算奖励
        reward = -np.abs(self.sim.system_losses())
        done = self.sim.check_convergence()
        return obs, reward, done, False, {}

    def _get_state(self):
        return np.array(self.sim.get_bus_voltages())

# 训练
env = PowerGridEnv()
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./log")
model.learn(total_timesteps=1_000_000)
