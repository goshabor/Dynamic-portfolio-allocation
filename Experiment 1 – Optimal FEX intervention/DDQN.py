from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim

import math
import random
import numpy as np

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

#for macbook: MPS, for PCs with Nvidia GPU: CUDA, else: CPU
USE_MPS = torch.backends.mps.is_available()
DEVICE = torch.device('mps' if USE_MPS else ('cuda' if torch.cuda.is_available() else 'cpu'))
torch.set_default_dtype(torch.float32 if DEVICE.type=='mps' else torch.float64)

### ------ Containers for parameters ------
#container for the FEX parameters
@dataclass
class FEXParameters: 
    w: float = 0.05
    mu: float = 0.25
    sigma: float = 0.3
    m: float = 0.02
    gamma: float = 3.0
    kappa: float = 1.0
    c: float = 0.1
    beta: float = 0.02
    T: float = 1.0
    
    R: float = 2.0 # S = [-R,R]

    n_t: int = 252
    n_zeta: int = 201

    train_x_min: float = -3.0
    train_x_max: float = 3.0

    target_min: float = -2.0
    target_max: float = 2.0

    min_abs_impulse: float = 0.0
    distance_reduction_tol: float = 1e-6

    @property
    def dt(self) -> float:
        return float(self.T)/float(self.n_t)

    @property
    def sqrt_dt(self) -> float:
        return math.sqrt(self.dt)

#container for the training part of the DDQN --> contains all hyperparameters
@dataclass
class TrainConfiguration: 
    seed: int = 42

    #neural network specific parameters
    episodes: int = 10000
    n_B: int = 512 #batch size
    M: int = 512 #number of Monte Carlo paths
    lr: float = 3e-4
    weight_decay: float = 0.0

    #target network
    target_update_interval: int = 500
    tau: float = 1.0 #hard update = 1, soft update < 1

    #buffer
    buffer_capacity: int = 500000
    warmup_transitions: int = 4096

    #exploration schedule
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: Optional[int] = None

    #samples initial state
    tail_sample_prob: float = 0.25
    tail_std: float = 1.35
    eval_interval_sample_prob: float = 0.35

    #network architecture and activation
    hidden_dims: Tuple[int, ...] = (512, 64)
    activation_str: str = 'relu'
    zero_bias: bool = True

    #silu specific hyperparameters
    silu_gain: float = 1.0
    output_gain: float = 1.0

    #loss
    loss: str = 'huber' #{'huber', 'mse'}
    huber_delta: float = 1.0
    grad_clip_norm: float = 10.0

    #for evaluation after training
    target_chunk_size: int = 64
    state_chunk_size: int = 4096

    #logging and checkpointing
    log_every: int = 25
    checkpoint_path: str = 'DDQN.pt'

### ------ Utility functions ------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if(torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)

def string_to_activation(activation_str):
    if(activation_str=='relu'): return nn.ReLU()
    if(activation_str=='silu'): return nn.SiLU()
    if(activation_str=='tanh'): return nn.Tanh()
    raise AssertionError(f'1: {activation_str} is unsupported: Choose either "relu", "silu" or "tanh"!')

def initialise_gains(activation_str, silu_gain=1.0, output_gain=1.0, zero_bias=True):
    if(activation_str == 'tanh'):
        hidden_init = 'xavier_uniform'
        hidden_gain = float(nn.init.calculate_gain('tanh'))
    elif(activation_str == 'relu'):
        hidden_init = 'kaiming_uniform'
        hidden_gain = float(nn.init.calculate_gain('relu'))
    elif(activation_str == 'silu'):
        hidden_init = 'xavier_uniform'
        hidden_gain = float(silu_gain)
    else: raise AssertionError(f'2: {activation_str} is unsupported: Choose either "relu", "silu", or "tanh"!')

    return {
        'activation': activation_str,
        'hidden_init': hidden_init,
        'hidden_gain': hidden_gain,
        'output_init': 'xavier_uniform',
        'output_gain': output_gain,
        'bias_init': 'zeros' if bool(zero_bias) else 'pytorch_linear_default',
    }

def initialise_linear_for_activation(layer, activation_str, is_output, output_gain=1.0, silu_gain=1.0, zero_bias=True):
    if is_output:
        nn.init.xavier_uniform_(layer.weight, gain=float(output_gain))
    elif(activation_str == 'tanh'):
        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("tanh"))
    elif(activation_str == 'relu'):
        nn.init.kaiming_uniform_(layer.weight, a=0.0, mode="fan_in", nonlinearity="relu")
    elif(activation_str == 'silu'):
        nn.init.xavier_uniform_(layer.weight, gain=float(silu_gain))
    else:
        raise AssertionError(f'{activation_str} is unsupported: choose "relu", "silu", or "tanh".')

    if zero_bias and layer.bias is not None:
        nn.init.zeros_(layer.bias)

def as_float_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to(device=DEVICE)
    return torch.as_tensor(x, device=DEVICE)

### ------ DDQN Network ------
class FEXDDQN(nn.Module):
    def __init__(self, params, hidden_dims, activation_str, silu_gain=1.0, output_gain=1.0, zero_bias=True):
        super().__init__()

        self.params = params
        self.hidden_dims = hidden_dims
        self.activation_str = activation_str

        self.silu_gain = silu_gain
        self.output_gain = output_gain
        self.zero_bias = zero_bias

        x_scale = max(abs(params.train_x_min - params.m), abs(params.train_x_max - params.m), 1.0)

        self.register_buffer('x_center', torch.tensor(params.m))
        self.register_buffer('x_scale', torch.tensor(x_scale))
        self.register_buffer('target_grid', torch.linspace(params.target_min, params.target_max, params.n_zeta))

        dimensions = [2, *self.hidden_dims, 1+int(params.n_zeta)]
        layers = []

        for i in range(len(dimensions)-2):
            layers.append(nn.Linear(dimensions[i], dimensions[i+1]))
            layers.append(string_to_activation(self.activation_str))
        
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()
    
    def reset_parameters(self):
        linear_layers = [module for module in self.net if isinstance(module, nn.Linear)]
        if(not linear_layers):
            return
        
        for layer in linear_layers[:-1]:
            initialise_linear_for_activation(layer=layer, activation_str=self.activation_str, is_output=False, silu_gain=self.silu_gain, output_gain=self.output_gain, zero_bias=self.zero_bias)
        initialise_linear_for_activation(layer=linear_layers[-1], activation_str=self.activation_str, is_output=True, silu_gain=self.silu_gain, output_gain=self.output_gain, zero_bias=self.zero_bias)
        
        for module in self.net:
            if(isinstance(module, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def initialise_info(self):
        return initialise_gains(self.activation_str)
    
    def encode_state(self, time_index, x):
        time_float = time_index.to(dtype=torch.float32)
        
        #scaling time and state
        t_scaled = time_float/self.params.n_t
        x_scaled = (x.to(dtype=torch.float32)-self.x_center)/self.x_scale

        return torch.stack((t_scaled, x_scaled), dim=-1)
    
    def forward(self, time_index, x):
        return self.net(self.encode_state(time_index=time_index, x=x))
    
    def metadata(self):
        return {
            'params': asdict(self.params),
            'hidden_dims': self.hidden_dims,
            'activation': self.activation_str,
            'silu_gain': self.silu_gain,
            'output_gain': self.output_gain,
            'zero_bias': self.zero_bias,
            'initialisation': self.initialise_info(),
            'init_info': self.initialise_info()
        }

### ------ Replay buffer ------
class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.time_index = torch.empty(self.capacity, dtype=torch.long, device=DEVICE)
        self.x = torch.empty(self.capacity, dtype=torch.float32, device=DEVICE)
        self.action = torch.empty(self.capacity, dtype=torch.long, device=DEVICE)
        self.reward = torch.empty(self.capacity, dtype=torch.float32, device=DEVICE)
        self.next_time_index = torch.empty(self.capacity, dtype=torch.long, device=DEVICE)
        self.next_x = torch.empty(self.capacity, dtype=torch.float32, device=DEVICE)
        self.done = torch.empty(self.capacity, dtype=torch.bool, device=DEVICE)
        self.pos = 0
        self.size = 0

    @torch.no_grad()
    def add_batch(self, time_index, x, action, reward, next_time_index, next_x, done):
        n = int(x.numel())
        if(n<=0):
            return
        if(n>self.capacity):
            time_index = time_index[-self.capacity :]
            x = x[-self.capacity :]
            action = action[-self.capacity :]
            reward = reward[-self.capacity :]
            next_time_index = next_time_index[-self.capacity :]
            next_x = next_x[-self.capacity :]
            done = done[-self.capacity :]
            n = self.capacity

        first = min(n, self.capacity - self.pos)
        second = n - first
        sl = slice(self.pos, self.pos + first)
        self.time_index[sl].copy_(time_index[:first])
        self.x[sl].copy_(x[:first])
        self.action[sl].copy_(action[:first])
        self.reward[sl].copy_(reward[:first])
        self.next_time_index[sl].copy_(next_time_index[:first])
        self.next_x[sl].copy_(next_x[:first])
        self.done[sl].copy_(done[:first])

        if(second > 0):
            sl2 = slice(0, second)
            self.time_index[sl2].copy_(time_index[first:])
            self.x[sl2].copy_(x[first:])
            self.action[sl2].copy_(action[first:])
            self.reward[sl2].copy_(reward[first:])
            self.next_time_index[sl2].copy_(next_time_index[first:])
            self.next_x[sl2].copy_(next_x[first:])
            self.done[sl2].copy_(done[first:])

        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    @torch.no_grad()
    def sample(self, n_B):
        idx = torch.randint(0, self.size, (n_B,), device=DEVICE)
        return self.time_index[idx], self.x[idx], self.action[idx], self.reward[idx], self.next_time_index[idx], self.next_x[idx], self.done[idx]

### ------ Action masks ------
@torch.no_grad()
def action_mask(params, x, target_grid):
    x = x.to(dtype=torch.float32)
    batch = int(x.numel())
    wait = torch.ones((batch,1), dtype=torch.bool, device=DEVICE)

    targets = target_grid.to(device=DEVICE, dtype=torch.float32).view(1, -1)
    x_col = x.view(-1, 1)
    impulse_abs = torch.abs(targets - x_col)
    valid = impulse_abs >= float(params.min_abs_impulse)

    current_dist = torch.abs(x_col - float(params.m))
    target_dist = torch.abs(targets - float(params.m))
    valid = valid & (target_dist <= current_dist - float(params.distance_reduction_tol))

    return torch.cat((wait, valid), dim=1)

@torch.no_grad()
def masked_argmax(q_values, mask): 
    very_negative = torch.finfo(q_values.dtype).min/4.0
    masked_q = torch.where(mask, q_values, torch.full_like(q_values, very_negative))
    return torch.argmax(masked_q, dim=1)

@torch.no_grad()
def random_valid_actions(mask):
    scores = torch.rand(mask.shape, device=DEVICE)
    scores = torch.where(mask, scores, torch.full_like(scores, -1.0))
    return torch.argmax(scores, dim=1)

@torch.no_grad()
def epsilon_greedy_actions(q_values, mask, epsilon):
    greedy = masked_argmax(q_values, mask)
    
    if(epsilon <= 0.0):
        return greedy
    
    random_actions = random_valid_actions(mask)
    choose_random = torch.rand(greedy.shape, device=DEVICE) < epsilon
    return torch.where(choose_random, random_actions, greedy)

def FEX_step(params, target_grid, time_index, x, action, noise=None):
    x = x.to(dtype=torch.float32)
    time_index = time_index.to(dtype=torch.long)
    action = action.to(dtype=torch.long)

    if noise is None:
        noise = torch.randn_like(x)
    else:
        noise = noise.to(device=DEVICE)

    is_impulse = action > 0
    safe_target_idx = torch.clamp(action - 1, min=0, max=int(params.n_zeta) - 1)
    selected_target = target_grid.to(device=DEVICE, dtype=torch.float32)[safe_target_idx]
    x_after_impulse = torch.where(is_impulse, selected_target, x)
    zeta = x_after_impulse - x

    dt = params.dt
    sqrt_dt = params.sqrt_dt

    t = time_index.to(dtype=torch.float32)*dt
    discount = torch.exp(-params.beta*t)

    running_reward = -discount*((x_after_impulse - params.m).square() + params.gamma*params.w**2)*dt
    impulse_reward = torch.where(is_impulse, -discount*(params.kappa*torch.abs(zeta) + params.c), torch.zeros_like(x))
    reward = running_reward + impulse_reward

    next_x = x_after_impulse - params.mu*params.w*dt + params.sigma*sqrt_dt*noise

    next_time_index = time_index + 1
    done = next_time_index >= int(params.n_t)
    return reward, next_time_index, next_x, done

### ------ Sampling helpers ------
@torch.no_grad()
def sample_time_index(params, M, include_terminal):
    high = int(params.n_t) + (1 if include_terminal else 0)
    return torch.randint(0, high, (int(M),), dtype=torch.long, device=DEVICE)

@torch.no_grad()
def sample_x(params, config, M):
    u = torch.rand(M, device=DEVICE)

    eval_prob = max(0.0, min(1.0, config.eval_interval_sample_prob))
    tail_prob = max(0.0, min(1.0 - eval_prob, config.tail_sample_prob))

    eval_mask = u < eval_prob
    tail_mask = (u >= eval_prob) & (u < eval_prob+tail_prob)

    train_lo = params.train_x_min
    train_hi = params.train_x_max
    target_lo = params.target_min
    target_hi = params.target_max

    uniform_vals = train_lo + (train_hi - train_lo)*torch.rand(M, device=DEVICE)
    tail_vals = params.m + config.tail_std*torch.randn(M, device=DEVICE)
    tail_vals = torch.clamp(tail_vals, train_lo, train_hi)
    eval_vals = target_lo + (target_hi - target_lo)*torch.rand(M, device=DEVICE)

    return torch.where(eval_mask, eval_vals, torch.where(tail_mask, tail_vals, uniform_vals))

def epsilon_at(episode, config):
    decay = config.epsilon_decay_episodes
    if(decay is None):
        decay = max(1, int(0.8 * int(config.episodes)))
    progress = min(1.0, episode/max(1, decay))
    return config.epsilon_start + progress*(config.epsilon_end - config.epsilon_start)

@torch.no_grad()
def update_target_network(online, target, tau):
    tau = float(tau)
    if(tau >= 1.0):
        target.load_state_dict(online.state_dict())
        return
    for target_param, online_param in zip(target.parameters(), online.parameters()):
        target_param.data.mul_(1.0 - tau).add_(online_param.data, alpha=tau)

    for target_buffer, online_buffer in zip(target.buffers(), online.buffers()):
        target_buffer.data.copy_(online_buffer.data)

### ------ Training procedure ------
def train(params, config):
    set_seed(config.seed)

    #same initialisation for both online and target networks
    model = FEXDDQN(
        params=params, 
        hidden_dims=config.hidden_dims, 
        activation_str=config.activation_str, 
        silu_gain=config.silu_gain,
        output_gain=config.output_gain,
        zero_bias=config.zero_bias,
    ).to(device=DEVICE)

    target_model = FEXDDQN(
        params=params, 
        hidden_dims=config.hidden_dims, 
        activation_str=config.activation_str, 
        silu_gain=config.silu_gain,
        output_gain=config.output_gain,
        zero_bias=config.zero_bias,
    ).to(device=DEVICE)
    
    update_target_network(online=model, target=target_model, tau=1.0)
    target_model.eval()

    optimiser = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    replay_buffer = ReplayBuffer(capacity=int(config.buffer_capacity))

    time_index = sample_time_index(params=params, M=int(config.M), include_terminal=False)
    x = sample_x(params=params, config=config, M=int(config.M))

    history = {
        'episode': [],
        'loss': [],
        'reward_mean': [],
        'td_abs': [],
        'q_mean': [],
        'target_mean': [],
        'epsilon': [],
        'buffer_size': [],
    }

    last_loss_t = None
    last_td_abs_t = None #td means temporal difference
    last_q_mean_t = None
    last_target_mean_t = None
    last_reward_mean_t = None

    for episode in range(1, int(config.episodes)+1): 
        epsilon = epsilon_at(episode=episode, config=config)

        #collecting one vectorised pathwise transition from each active episode
        model.eval()
        with torch.no_grad():
            Q = model(time_index, x)
            mask = action_mask(params, x, model.target_grid)
            action = epsilon_greedy_actions(q_values=Q, mask=mask, epsilon=epsilon)
            reward, next_time_index, next_x, done = FEX_step(params=params, target_grid=model.target_grid, time_index=time_index, x=x, action=action)

            replay_buffer.add_batch(time_index=time_index, x=x, action=action, reward=reward, next_time_index=next_time_index, next_x=next_x, done=done)
            last_reward_mean_t = reward.mean().detach()

            #continuous each time step unless terminal. Terminal paths are reset to randomised starts to cover the time and state domain
            reset_time = sample_time_index(params=params, M=int(config.M), include_terminal=False)
            reset_x = sample_x(params=params, config=config, M=int(config.M))

            time_index = torch.where(done, reset_time, next_time_index)
            x = torch.where(done, reset_x, next_x)

        #learning from replay buffer after warmup
        if(replay_buffer.size >= max(int(config.warmup_transitions), int(config.n_B))):
            model.train()
            b_time_index, b_x, b_action, b_reward, b_next_time_index, b_next_x, b_done = replay_buffer.sample((int(config.n_B)))

            #online model
            Q_all = model(b_time_index, b_x)
            Q_predicted = Q_all.gather(1, b_action.view(-1,1)).squeeze(1)

            #DDQN target
            with torch.no_grad():
                next_Q_online = model(b_next_time_index, b_next_x)
                next_mask = action_mask(params=params, x=b_next_x, target_grid=model.target_grid)
                next_action = masked_argmax(q_values=next_Q_online, mask=next_mask)
            
                next_Q_target_all = target_model(b_next_time_index, b_next_x)
                next_Q_target = next_Q_target_all.gather(1, next_action.view(-1,1)).squeeze(1)

                DDQN_target = b_reward + (~b_done).to(torch.float32)*next_Q_target

            if(config.loss.lower() == 'mse'):
                loss = F.mse_loss(Q_predicted, DDQN_target)
            else:
                abs_temporal_difference = torch.abs(Q_predicted - DDQN_target)
                loss = torch.where(abs_temporal_difference <= config.huber_delta, abs_temporal_difference.square()/2, config.huber_delta*(abs_temporal_difference-config.huber_delta/2)).mean()
            
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
            optimiser.step()

            with torch.no_grad():
                last_loss_t = loss.detach()
                last_td_abs_t = torch.mean(torch.abs(Q_predicted-DDQN_target)).detach()
                last_q_mean_t = Q_predicted.mean().detach()
                last_target_mean_t = DDQN_target.mean().detach()
        
        if(int(config.target_update_interval)>0 and episode%int(config.target_update_interval)==0):
            update_target_network(online=model, target=target_model, tau=config.tau)
        
        if(int(config.log_every) and episode%int(config.log_every)==0 or episode==1 or episode==int(config.episodes)):
            last_loss = float('nan') if last_loss_t is None else float(last_loss_t.detach().cpu())
            last_td_abs = float('nan') if last_td_abs_t is None else float(last_td_abs_t.detach().cpu())
            last_Q_mean = float('nan') if last_q_mean_t is None else float(last_q_mean_t.detach().cpu())
            last_target_mean = float('nan') if last_target_mean_t is None else float(last_target_mean_t.detach().cpu())
            last_reward_mean = float('nan') if last_reward_mean_t is None else float(last_reward_mean_t.detach().cpu())
            
            history['episode'].append(int(episode))
            history['loss'].append(last_loss)
            history['td_abs'].append(last_td_abs)
            history['reward_mean'].append(last_reward_mean)
            history['q_mean'].append(last_Q_mean)
            history['target_mean'].append(last_target_mean)
            history['epsilon'].append(epsilon)
            history['buffer_size'].append(replay_buffer.size)

            print(f'Episode: {episode:>7d}/{config.episodes} | lr={config.lr} | Loss={last_loss:.5e} | Mean reward={last_reward_mean:.5e} | Temporal difference (abs)={last_td_abs:.5e} | Mean Q-value={last_Q_mean:.5e} | Mean target={last_target_mean:.5e} | Epsilon={epsilon:.4f} | Replay buffer size={replay_buffer.size}')

    model.eval()
    if(config.checkpoint_path):
        save_model(model=model, path=config.checkpoint_path, config=config, history=history)

    return model, history

#utilities to save and load trained models
def save_model(model, path, config, history):
    path = Path(path)
    payload = {
        'state_dict': model.state_dict(),
        'metadata': model.metadata()
    }
    payload['config'] = asdict(config)
    payload['history'] = {k: list(v) for k,v in history.items()}
    torch.save(payload, str(path))

def load_model(path, device=DEVICE):
    payload = torch.load(str(path), map_location=device)
    meta = payload["metadata"]
    params = FEXParameters(**meta["params"])
    model = FEXDDQN(
        params=params,
        hidden_dims=tuple(meta.get("hidden_dims", (96, 96, 64))),
        activation_str=meta.get("activation", "silu"),
        silu_gain=float(meta.get("silu_gain", 1.0)),
        output_gain=float(meta.get("output_gain", 1.0)),
        zero_bias=bool(meta.get("zero_bias", True)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload

### ------- Model evaluation and comparison against QVI solver -------
#greedy policy and value evaluation
@torch.no_grad()
def policy_curve(model, x_grid, time_index, state_chunk_size):
    model.eval()
    params = model.params
    next(model.parameters()).device

    x_np = np.asarray(x_grid, dtype=np.float32).reshape(-1)
    n = len(x_np)

    chunk = int(state_chunk_size)

    action_arr = []
    intervention_arr = []
    advantage_arr = []
    zeta_arr = []
    target_out_arr = []
    Q_wait_arr = []
    Q_impulse_arr = []
    Q_values_arr = []

    for i in range(0, n, chunk):
        stop = min(n, i+chunk)
        x = torch.as_tensor(x_np[i:stop], device=DEVICE)
        t = torch.full((x.numel(),), int(time_index), dtype=torch.long, device=DEVICE)
        Q = model(t,x)
        
        mask = action_mask(params=params, x=x, target_grid=model.target_grid)
        a = masked_argmax(q_values=Q, mask=mask)
        Q_wait = Q[:,0]

        impulse_mask = mask[:,1:]
        impulse_Q_raw = Q[:, 1:]

        very_negative = torch.finfo(impulse_Q_raw.dtype).min/4
        impulse_Q_masked = torch.where(impulse_mask, impulse_Q_raw, torch.full_like(impulse_Q_raw, very_negative))
        Q_impulse_best = torch.where(torch.any(impulse_mask, dim=1), torch.max(impulse_Q_masked, dim=1).values, Q_wait)

        target_index = torch.clamp(a-1, min=0, max=int(params.n_zeta)-1)
        selected_target = torch.where(a > 0, model.target_grid[target_index], x)
        zeta = selected_target - x
        advantage = Q_impulse_best - Q_wait
        intervene = a > 0
        Q_value = torch.max(torch.where(mask, Q, torch.full_like(Q, very_negative)), dim=1).values

        action_arr.append(a.detach().cpu().numpy())
        intervention_arr.append(intervene.detach().cpu().numpy())
        advantage_arr.append(advantage.detach().cpu().numpy())
        zeta_arr.append(zeta.detach().cpu().numpy())
        target_out_arr.append(selected_target.detach().cpu().numpy())
        Q_wait_arr.append(Q_wait.detach().cpu().numpy())
        Q_impulse_arr.append(Q_impulse_best.detach().cpu().numpy())
        Q_values_arr.append(Q_value.detach().cpu().numpy())
    
    return {
        'x': x_np,
        'action': np.concatenate(action_arr),
        'intervene': np.concatenate(intervention_arr).astype(bool),
        'advantage': np.concatenate(advantage_arr),
        'zeta': np.concatenate(zeta_arr),
        'target': np.concatenate(target_out_arr),
        'Q_wait': np.concatenate(Q_wait_arr),
        'Q_imp': np.concatenate(Q_impulse_arr),
        'Q_value': np.concatenate(Q_values_arr),
    }

@torch.no_grad()
def q_value_curve(model, x_grid, time_index, state_chunk_size):
    policy = policy_curve(model=model, x_grid=x_grid, time_index=time_index, state_chunk_size=state_chunk_size)
    return {'x': policy['x'], 'value': policy['Q_value']}

@torch.no_grad()
def greedy_actions(model, time_index, x):
    Q = model(time_index, x)
    mask = action_mask(model.params, x, model.target_grid)
    return masked_argmax(Q, mask)

@torch.no_grad()
def make_antithetic_noise(rows, cols, antithetic=True):
    if(antithetic and cols >= 2 and cols%2 == 0):
        half = torch.randn((rows, cols // 2), dtype=torch.float32, device=DEVICE)
        return torch.cat((half, -half), dim=1)
    return torch.randn((rows, cols), dtype=torch.float32, device=DEVICE)

@torch.no_grad()
def MC_policy_curve(model, x_grid, n_paths_per_x=512, time_index=0, batch_x=32, path_batch_size=2048, antithetic=True, seed=None):
    if(seed is not None):
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

    model.eval()
    params = model.params
    x_np = np.asarray(x_grid).reshape(-1)
    n_x = x_np.size
    n_paths = n_paths_per_x

    values = np.empty(n_x)
    stds = np.empty(n_x)

    batch_x = max(1, int(batch_x))
    path_batch_size = max(1, int(path_batch_size))

    for outer_start in range(0, n_x, batch_x):
        outer_stop = min(n_x, outer_start + batch_x)
        current_x_np = x_np[outer_start:outer_stop]
        rows = current_x_np.size
        x0 = torch.as_tensor(current_x_np, device=DEVICE, dtype=torch.float32)

        sum_v = torch.zeros(rows, device=DEVICE)
        sumsq_v = torch.zeros(rows, device=DEVICE)
        paths_done = 0

        while paths_done < n_paths:
            cols = min(n_paths - paths_done, max(1, path_batch_size//rows))
            if(antithetic and cols > 1 and cols % 2 == 1):
                cols -= 1
            cols = max(1, cols)

            x = x0.view(rows, 1).repeat(1, cols).reshape(-1)
            t = torch.full((rows * cols,), int(time_index), dtype=torch.long, device=DEVICE)
            total_reward = torch.zeros(rows * cols, dtype=torch.float32, device=DEVICE)

            for i in range(int(time_index), int(params.n_t)):
                action = greedy_actions(model=model, time_index=t, x=x)
                noise = make_antithetic_noise(rows=rows, cols=cols, antithetic=antithetic).reshape(-1)
                reward, next_t, next_x, done = FEX_step(params=params, target_grid=model.target_grid, time_index=t, x=x, action=action, noise=noise)
                total_reward += reward
                x = next_x
                t = next_t

            vals = total_reward.view(rows, cols).to(dtype=torch.float32)
            sum_v += vals.sum(dim=1)
            sumsq_v += vals.square().sum(dim=1)
            paths_done += cols

        mean = sum_v/n_paths
        if(n_paths > 1):
            sample_var = torch.clamp((sumsq_v - sum_v.square()/n_paths) / (n_paths-1), min=0.0)
            std = torch.sqrt(sample_var/n_paths)
        else:
            std = torch.zeros_like(mean)

        values[outer_start:outer_stop] = mean.detach().cpu().numpy()
        stds[outer_start:outer_stop] = std.detach().cpu().numpy()

    return {'x': x_np.astype(np.float32), 'value': values, 'std': stds}