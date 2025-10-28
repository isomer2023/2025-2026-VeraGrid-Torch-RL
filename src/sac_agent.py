# sac_agent.py
import os, torch, numpy as np, torch.optim as optim
from src.sac_networks import SquashedGaussianMLPActor, MLPQFunction

class SACAgent:
    def __init__(self, obs_dim, act_dim, act_limit,
                 hidden_sizes=(256,256),
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, polyak=0.995, target_entropy=None, device="cpu"):
        self.device = torch.device(device)
        self.act_limit = float(act_limit)
        self.actor = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, act_limit).to(self.device)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q1_targ = MLPQFunction(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q2_targ = MLPQFunction(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())
        self.pi_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=critic_lr)
        self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp().item()
        self.gamma = gamma; self.polyak = polyak
        self.target_entropy = -act_dim if target_entropy is None else target_entropy

    def select_action(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.actor.act(obs_t, deterministic=deterministic)[0]

    def update(self, batch):
        o  = torch.as_tensor(batch["obs"],  dtype=torch.float32, device=self.device)
        o2 = torch.as_tensor(batch["obs2"], dtype=torch.float32, device=self.device)
        a  = torch.as_tensor(batch["act"],  dtype=torch.float32, device=self.device)
        r  = torch.as_tensor(batch["rew"],  dtype=torch.float32, device=self.device)
        d  = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            a2, logp_a2 = self.actor.forward(o2)
            q1_t = self.q1_targ(o2, a2)
            q2_t = self.q2_targ(o2, a2)
            q_t  = torch.min(q1_t, q2_t) - self.log_alpha.exp() * logp_a2
            backup = r + self.gamma * (1 - d) * q_t

        q1 = self.q1(o, a)
        q2 = self.q2(o, a)
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()

        self.q1_opt.zero_grad(); loss_q1.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); loss_q2.backward(); self.q2_opt.step()

        for p in self.q1.parameters(): p.requires_grad = False
        for p in self.q2.parameters(): p.requires_grad = False

        pi, logp_pi = self.actor.forward(o)
        q1_pi = self.q1(o, pi)
        q2_pi = self.q2(o, pi)
        q_pi  = torch.min(q1_pi, q2_pi)
        loss_pi = (self.log_alpha.exp() * logp_pi - q_pi).mean()

        self.pi_opt.zero_grad(); loss_pi.backward(); self.pi_opt.step()

        for p in self.q1.parameters(): p.requires_grad = True
        for p in self.q2.parameters(): p.requires_grad = True

        # temperature alpha update
        alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        # target soft update
        with torch.no_grad():
            for p, p_t in zip(self.q1.parameters(), self.q1_targ.parameters()):
                p_t.data.mul_(self.polyak); p_t.data.add_((1 - self.polyak) * p.data)
            for p, p_t in zip(self.q2.parameters(), self.q2_targ.parameters()):
                p_t.data.mul_(self.polyak); p_t.data.add_((1 - self.polyak) * p.data)

    # --------- 保存 / 加载 ----------
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_targ": self.q1_targ.state_dict(),
            "q2_targ": self.q2_targ.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu().numpy(),
        }, path)

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_targ.load_state_dict(ckpt["q1_targ"])
        self.q2_targ.load_state_dict(ckpt["q2_targ"])
        self.log_alpha = torch.tensor(ckpt["log_alpha"], requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
