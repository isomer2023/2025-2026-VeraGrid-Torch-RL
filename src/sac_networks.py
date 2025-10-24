import torch
import torch.nn as nn

LOG_STD_MIN, LOG_STD_MAX = -20, 2

def mlp(sizes, act_fn=nn.ReLU, out_act=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = act_fn if j < len(sizes) - 2 else out_act
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), nn.ReLU, nn.ReLU)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = float(act_limit)

    def forward(self, obs, deterministic=False, with_logprob=True):
        h = self.net(obs)
        mu = self.mu_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mu, std)
        pi_action = mu if deterministic else dist.rsample()

        logp_pi = dist.log_prob(pi_action).sum(dim=-1) if with_logprob else None
        pi_action = torch.tanh(pi_action) * self.act_limit
        return pi_action, logp_pi

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        a, _ = self.forward(obs, deterministic, with_logprob=False)
        return a.cpu().numpy()

class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1])

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1)).squeeze(-1)