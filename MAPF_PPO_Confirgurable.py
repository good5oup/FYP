"""Configurable multi-agent grid MAPF with PPO and optional imitation/attention.

Edit ONLY the CONFIG section for normal experiments.
Install: pip install torch numpy matplotlib
Test: python mapf_ppo_configurable.py --test-only
Train: python mapf_ppo_configurable.py

Expert demonstrations are supervision, never forced execution. This is a discrete
MAPF research prototype; PPO does not guarantee convergence or optimality.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import copy
import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

# ============================================================================
# 1. CONFIGURATION - CHANGE EXPERIMENTS HERE (NO OTHER EDITS REQUIRED)
# ============================================================================
@dataclass
class ExperimentConfig:
    grid_size: int = 10
    starts: tuple = ((0, 0), (0, 9))
    goals: tuple = ((9, 9), (9, 0))
    obstacles: tuple = (
        (0,4), (0,6), (0,7), (1,2), (2,5), (2,8), (2,9),
        (3,0), (4,7), (5,2), (5,3), (5,5), (5,7),
        (6,6), (6,7), (8,0), (8,3), (8,5), (8,6), (9,2)
    )
    # Number of agents = len(starts) = len(goals); no separate number to update.
    use_attention: bool = True 
    use_imitation: bool = True
    imitation_coefficient: float = 0.10
    expert_prefix_steps: int = 18
    # Expert trajectories: one path per agent; all paths must have equal length,
    # start at starts, end at goals, and be jointly collision-free.
    # For a different map or agent count: replace paths, or set use_imitation=False.
    expert_paths: tuple = (
        ((0,0),(1,0),(2,0),(2,1),(3,1),(4,1),(5,1),(6,1),(7,1),(7,2),
         (7,3),(7,4),(8,4),(9,4),(9,5),(9,6),(9,7),(9,8),(9,9)),
        ((0,9),(1,9),(1,8),(1,7),(2,7),(3,7),(3,6),(4,6),(4,5),(4,4),
         (5,4),(6,4),(7,4),(7,3),(7,2),(8,2),(8,1),(9,1),(9,0))
    )
    use_static_action_mask: bool = True
    episodes: int = 5000
    max_steps: int = 75
    rollout_size: int = 256
    ppo_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    hidden_dim: int = 64
    attention_dim: int = 32
    attention_heads: int = 4
    eval_interval: int = 500
    eval_episodes: int = 20
    log_interval: int = 100
    seed: int = 42
    output_dir: str = 'MAPF_Results'
    show_plots: bool = False

CONFIG = ExperimentConfig()

ACTIONS = ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT')
DELTAS = ((-1,0), (1,0), (0,-1), (0,1), (0,0))


def validate_config(cfg):
    n = cfg.grid_size
    if n < 2 or len(cfg.starts) < 1 or len(cfg.starts) != len(cfg.goals):
        raise ValueError('Grid size >=2; starts and goals must contain the same nonzero number of agents.')
    if len(set(cfg.starts)) != len(cfg.starts) or len(set(cfg.goals)) != len(cfg.goals):
        raise ValueError('Agent starts and goals must each be distinct.')
    walls = set(cfg.obstacles)
    if len(walls) != len(cfg.obstacles):
        raise ValueError('Duplicate obstacles detected.')
    def inside(p): return len(p)==2 and all(isinstance(v,int) and 0 <= v < n for v in p)
    if not all(inside(p) for p in cfg.starts+cfg.goals+cfg.obstacles):
        raise ValueError('A coordinate is outside the grid.')
    if any(p in walls for p in cfg.starts+cfg.goals):
        raise ValueError('A start or goal overlaps an obstacle.')
    if cfg.attention_dim % cfg.attention_heads:
        raise ValueError('attention_dim must be divisible by attention_heads.')
    if cfg.use_imitation:
        if cfg.expert_prefix_steps < 1 or cfg.imitation_coefficient <= 0:
            raise ValueError('Enabled imitation requires a positive prefix length and coefficient.')
        if len(cfg.expert_paths) != len(cfg.starts):
            raise ValueError('Provide one expert path per agent, or disable imitation.')
        validate_expert_paths(cfg)
    if cfg.episodes < 1 or cfg.max_steps < 1 or cfg.rollout_size < 1 or cfg.eval_interval < 1 or cfg.eval_episodes < 1:
        raise ValueError('Training counts must be positive.')


# ============================================================================
# 2. ENVIRONMENT - ARBITRARY GRID SIZE AND NUMBER OF AGENTS
# ============================================================================
class GridMAPF:
    def __init__(self, cfg):
        self.cfg = cfg
        self.size = cfg.grid_size
        self.n_agents = len(cfg.starts)
        self.walls = frozenset(cfg.obstacles)
        self.reset()

    def free(self, p):
        r,c = p
        return 0 <= r < self.size and 0 <= c < self.size and p not in self.walls

    def reset(self):
        self.positions = tuple(self.cfg.starts)
        self.arrived = [p == g for p,g in zip(self.positions,self.cfg.goals)]
        self.steps = self.collisions = self.invalid = self.waits = 0
        return self.positions

    def onehot(self, p):
        a = torch.zeros(self.size**2)
        a[p[0]*self.size+p[1]] = 1.
        return a

    def walls_vector(self):
        a = torch.zeros(self.size**2)
        for r,c in self.walls: a[r*self.size+c] = 1.
        return a

    def actor_obs(self, i):
        # Channels: own position, own goal, all OTHER agents (occupancy), obstacles.
        other = torch.zeros(self.size**2)
        for j,p in enumerate(self.positions):
            if j != i: other[p[0]*self.size+p[1]] = 1.
        return torch.cat((self.onehot(self.positions[i]),self.onehot(self.cfg.goals[i]),
                          other,self.walls_vector()))

    def critic_obs(self):
        # Per-agent positions, per-agent goals, obstacle map.
        return torch.cat(tuple(self.onehot(p) for p in self.positions) +
                         tuple(self.onehot(g) for g in self.cfg.goals) + (self.walls_vector(),))

    def mask(self, i):
        r,c = self.positions[i]
        return torch.tensor([self.free((r+dr,c+dc)) for dr,dc in DELTAS],dtype=torch.bool)

    def step(self, actions):
        if len(actions) != self.n_agents or any(int(a) not in range(5) for a in actions):
            raise ValueError('Exactly one action in 0..4 is required per agent.')
        old = self.positions
        proposed = list(old)
        costs = [-1.]*self.n_agents
        invalid = [False]*self.n_agents
        collision = [False]*self.n_agents
        for i,a in enumerate(actions):
            a = int(a)
            if a == 4:
                self.waits += 1
                if old[i] == self.cfg.goals[i]: costs[i] = 0.
                continue
            dr,dc = DELTAS[a]
            nxt = (old[i][0]+dr,old[i][1]+dc)
            if self.free(nxt): proposed[i] = nxt
            else:
                invalid[i] = True
                costs[i] = -2.

        # Reject all moves into shared destinations, edge swaps, and occupied
        # stationary cells. Iterate to resolve cascading stationary conflicts.
        changed = True
        while changed:
            changed = False
            rejected = set()
            for i in range(self.n_agents):
                for j in range(i+1,self.n_agents):
                    if proposed[i] == proposed[j]:
                        if proposed[i] != old[i]: rejected.add(i)
                        if proposed[j] != old[j]: rejected.add(j)
                    if proposed[i] == old[j] and proposed[j] == old[i] and proposed[i] != old[i]:
                        rejected.update((i,j))
            for i in rejected:
                if proposed[i] != old[i]:
                    proposed[i] = old[i]
                    collision[i] = True
                    changed = True
        if len(set(proposed)) != self.n_agents:
            raise RuntimeError('Collision resolver left overlapping agents.')
        for i in range(self.n_agents):
            if collision[i]: costs[i] = -5.
            if proposed[i] == self.cfg.goals[i] and not self.arrived[i]:
                costs[i] += 10.
                self.arrived[i] = True
        self.positions = tuple(proposed)
        self.steps += 1
        self.invalid += sum(invalid)
        self.collisions += sum(collision)
        done = all(p == g for p,g in zip(self.positions,self.cfg.goals))
        truncated = self.steps >= self.cfg.max_steps and not done
        reward = sum(costs) + (30. if done else 0.)
        return self.positions,reward,done,truncated,{'invalid':sum(invalid),'collisions':sum(collision)}

    def display(self):
        print(f'\n{self.size}x{self.size} MAPF | {self.n_agents} agents')
        for r in range(self.size):
            cells=[]
            for c in range(self.size):
                p=(r,c)
                if p in self.positions: label=f'A{self.positions.index(p)+1}'
                elif p in self.walls: label='##'
                elif p in self.cfg.goals: label=f'G{self.cfg.goals.index(p)+1}'
                else: label='.'
                cells.append(f'{label:>4}')
            print(''.join(cells))


# ============================================================================
# 3. NETWORKS - MLP OR SPATIAL ATTENTION (SAME PPO)
# ============================================================================
class Encoder(nn.Module):
    def __init__(self, cfg, channels, actor=False):
        super().__init__()
        self.attention = cfg.use_attention
        self.channels = channels
        self.cells = cfg.grid_size**2
        self.actor = actor
        if self.attention:
            d = cfg.attention_dim
            self.project = nn.Linear(channels,d)
            self.pos = nn.Parameter(torch.randn(1,self.cells,d)*0.02)
            self.attn = nn.MultiheadAttention(d,cfg.attention_heads,batch_first=True)
            self.norm = nn.LayerNorm(d)
            self.out_dim = d
        else:
            d = cfg.hidden_dim
            self.net = nn.Sequential(nn.Linear(channels*self.cells,d),nn.ReLU(),
                                     nn.Linear(d,d),nn.ReLU())
            self.out_dim = d

    def forward(self,x):
        if not self.attention: return self.net(x)
        cells = x.reshape(-1,self.channels,self.cells).transpose(1,2)
        tokens = self.project(cells)+self.pos
        attended,_ = self.attn(tokens,tokens,tokens,need_weights=False)
        features = self.norm(tokens+attended)
        if self.actor:
            indices = cells[:,:,0].argmax(dim=1)
            return features[torch.arange(len(indices),device=indices.device),indices]
        return features.mean(dim=1)


class Actor(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.encoder = Encoder(cfg,4,actor=True)
        self.head = nn.Linear(self.encoder.out_dim,5)
    def forward(self,x):
        single = x.ndim == 1
        if single: x=x.unsqueeze(0)
        out = self.head(self.encoder(x))
        return out.squeeze(0) if single else out


class Critic(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.encoder = Encoder(cfg,2*len(cfg.starts)+1)
        self.head = nn.Linear(self.encoder.out_dim,1)
    def forward(self,x):
        single = x.ndim == 1
        if single: x=x.unsqueeze(0)
        out = self.head(self.encoder(x)).squeeze(-1)
        return out.squeeze(0) if single else out


# ============================================================================
# 4. EXPERT DEMONSTRATIONS - OPTIONAL CROSS-ENTROPY AUXILIARY LOSS
# ============================================================================
def validate_expert_paths(cfg):
    paths = cfg.expert_paths
    if not paths or len({len(p) for p in paths}) != 1:
        raise ValueError('Expert paths must have equal nonzero lengths.')
    if any(path[0]!=cfg.starts[i] or path[-1]!=cfg.goals[i] for i,path in enumerate(paths)):
        raise ValueError('Expert paths must match configured starts and goals.')
    if cfg.expert_prefix_steps > len(paths[0])-1:
        raise ValueError('Expert prefix longer than demonstration.')
    validation_cfg = copy.copy(cfg)
    validation_cfg.max_steps = max(cfg.max_steps, len(paths[0]) + 1)
    env = GridMAPF(validation_cfg)
    for t in range(len(paths[0])-1):
        joint=[]
        for path in paths:
            delta=(path[t+1][0]-path[t][0],path[t+1][1]-path[t][1])
            if delta not in DELTAS: raise ValueError(f'Non-adjacent expert move at step {t+1}.')
            joint.append(DELTAS.index(delta))
        _,_,done,trunc,info = env.step(joint)
        expected=tuple(path[t+1] for path in paths)
        if env.positions!=expected or info['invalid'] or info['collisions'] or trunc:
            raise ValueError(f'Invalid expert trajectory at step {t+1}: {env.positions} vs {expected}.')
        if done and t != len(paths[0])-2:
            raise ValueError('Expert path continues after joint completion.')
    if env.positions != tuple(cfg.goals): raise ValueError('Expert trajectory does not finish at goals.')


def expert_dataset(cfg):
    if not cfg.use_imitation: return None
    env = GridMAPF(cfg)
    obs,actions=[],[]
    for t in range(cfg.expert_prefix_steps):
        for i,path in enumerate(cfg.expert_paths):
            obs.append(env.actor_obs(i))
            delta=(path[t+1][0]-path[t][0],path[t+1][1]-path[t][1])
            actions.append(DELTAS.index(delta))
        env.step(actions[-env.n_agents:])
    return torch.stack(obs),torch.tensor(actions,dtype=torch.long)


def mask_from_observations(cfg,obs):
    n=cfg.grid_size
    out=[]
    walls=set(cfg.obstacles)
    for row in obs:
        r,c=divmod(int(row[:n*n].argmax()),n)
        out.append([0<=r+dr<n and 0<=c+dc<n and (r+dr,c+dc) not in walls for dr,dc in DELTAS])
    return torch.tensor(out,dtype=torch.bool)


def distribution(actor,obs,masks=None):
    logits=actor(obs)
    if masks is not None: logits=logits.masked_fill(~masks,-1e9)
    return Categorical(logits=logits)


# ============================================================================
# 5. PPO UPDATE - SHARED ACTOR, CENTRALIZED TEAM CRITIC
# ============================================================================
def gae(rewards,values,next_values,terminals,boundaries,cfg):
    out=torch.zeros(len(rewards))
    acc=0.
    for t in reversed(range(len(rewards))):
        delta=rewards[t]+cfg.gamma*(1.-float(terminals[t]))*next_values[t]-values[t]
        acc=delta+cfg.gamma*cfg.gae_lambda*(1.-float(boundaries[t]))*acc
        out[t]=acc
    return out


def update(actor,critic,actor_opt,critic_opt,buffer,expert,cfg):
    n=len(cfg.starts)
    obs=torch.stack(buffer['obs'])
    joint=torch.stack(buffer['joint'])
    next_joint=torch.stack(buffer['next_joint'])
    acts=torch.tensor(buffer['actions'],dtype=torch.long)
    old_logp=torch.stack(buffer['logp'])
    masks=torch.stack(buffer['masks'])
    with torch.no_grad():
        values=critic(joint)
        next_values=critic(next_joint)
        adv=gae(buffer['rewards'],values,next_values,buffer['terminals'],buffer['boundaries'],cfg)
        returns=adv+values
        adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8) if len(adv)>1 else adv
    last_il=None
    for _ in range(cfg.ppo_epochs):
        dist=distribution(actor,obs.reshape(-1,4*cfg.grid_size**2),
                          masks.reshape(-1,5) if cfg.use_static_action_mask else None)
        new_logp=dist.log_prob(acts.reshape(-1)).reshape(-1,n).sum(-1)
        ratio=torch.exp(new_logp-old_logp.sum(-1))
        clipped=torch.clamp(ratio,1-cfg.clip_epsilon,1+cfg.clip_epsilon)
        actor_loss=-torch.minimum(ratio*adv,clipped*adv).mean()-cfg.entropy_coefficient*dist.entropy().mean()
        if expert is not None:
            ex_obs,ex_actions=expert
            ex_mask=mask_from_observations(cfg,ex_obs) if cfg.use_static_action_mask else None
            il_loss=-distribution(actor,ex_obs,ex_mask).log_prob(ex_actions).mean()
            actor_loss=actor_loss+cfg.imitation_coefficient*il_loss
            last_il=float(il_loss.detach())
        actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(),0.5)
        actor_opt.step()
        critic_loss=F.mse_loss(critic(joint),returns)
        critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(),0.5)
        critic_opt.step()
    return last_il


def empty_buffer():
    return {k:[] for k in ('obs','joint','next_joint','actions','logp','masks','rewards','terminals','boundaries')}


# ============================================================================
# 6. EVALUATION, PLOTS, AND TRAINING
# ============================================================================
@torch.no_grad()
def evaluate(actor,cfg,greedy=True,seed=None):
    if seed is not None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return evaluate(actor,cfg,greedy)
    env=GridMAPF(cfg)
    path=[env.positions]
    total=0.
    for _ in range(cfg.max_steps):
        obs=torch.stack([env.actor_obs(i) for i in range(env.n_agents)])
        masks=torch.stack([env.mask(i) for i in range(env.n_agents)]) if cfg.use_static_action_mask else None
        dist=distribution(actor,obs,masks)
        actions=(dist.probs.argmax(-1) if greedy else dist.sample()).tolist()
        _,reward,done,trunc,_=env.step(actions)
        total+=reward
        path.append(env.positions)
        if done or trunc: break
    return dict(success=all(p==g for p,g in zip(env.positions,cfg.goals)),
                steps=env.steps,reward=total,collisions=env.collisions,
                invalid=env.invalid,waits=env.waits,path=path)


def plot_training(history,output,cfg):
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    x=np.arange(1,len(history)+1)
    for ax,key,title in zip(axes,('reward','success','steps'),('Team reward','Joint success (%)','Episode steps')):
        y=np.array([h[key] for h in history],dtype=float)
        if key=='success': y*=100
        ax.plot(x,y,alpha=.15)
        window=min(100,len(y))
        if window>1:
            ax.plot(x[window-1:],np.convolve(y,np.ones(window)/window,mode='valid'))
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output,dpi=150)
    if cfg.show_plots: plt.show()
    plt.close(fig)


def plot_paths(path,cfg,output):
    fig,ax=plt.subplots(figsize=(8,8))
    for r in range(cfg.grid_size):
        for c in range(cfg.grid_size):
            ax.add_patch(plt.Rectangle((c,r),1,1,facecolor='#64748b' if (r,c) in cfg.obstacles else '#f8fafc',edgecolor='#cbd5e1'))
    colors=plt.cm.tab10(np.linspace(0,1,max(2,len(cfg.starts))))
    for i in range(len(cfg.starts)):
        route=[joint[i] for joint in path]
        ax.plot([p[1]+.5 for p in route],[p[0]+.5 for p in route],'-o',color=colors[i],markersize=3,label=f'Agent {i+1}')
        sr,sc=cfg.starts[i]; gr,gc=cfg.goals[i]
        ax.text(sc+.5,sr+.5,f'A{i+1}',ha='center',va='center',fontsize=9,fontweight='bold')
        ax.scatter(gc+.5,gr+.5,marker='*',s=200,color=colors[i])
    ax.set_xlim(0,cfg.grid_size); ax.set_ylim(cfg.grid_size,0)
    ax.set_aspect('equal');ax.legend()
    ax.set_xticks(np.arange(.5,cfg.grid_size+.5),labels=range(cfg.grid_size))
    ax.set_yticks(np.arange(.5,cfg.grid_size+.5),labels=range(cfg.grid_size))
    ax.set_title('Greedy learned paths (no forced expert actions)')
    fig.tight_layout();fig.savefig(output,dpi=150)
    if cfg.show_plots: plt.show()
    plt.close(fig)


def display_final_results(result, cfg, best=None, checkpoint_path=None, first_success_episode=None):
    """Report all agents, team metrics, and best successful checkpoint."""
    print('\n' + '=' * 64)
    print(f'             {len(cfg.starts)}-AGENT MAPF RESULTS')
    print('=' * 64)
    for i in range(len(cfg.starts)):
        route = [joint[i] for joint in result['path']]
        reached = route[-1] == cfg.goals[i]
        print(f'\nAGENT {i+1} | {"GOAL REACHED" if reached else "GOAL NOT REACHED"}')
        print(f'  Start: {route[0]} | Goal: {cfg.goals[i]} | Final: {route[-1]}')
        print('  Path: ' + ' -> '.join(map(str, route)))
    print('\nTEAM PERFORMANCE')
    print(f'  All agents reached goals: {"YES" if result["success"] else "NO"}')
    print(f'  Makespan: {result["steps"]} steps')
    print(f'  Team reward: {result["reward"]:.2f}')
    print(f'  Collision attempts: {result["collisions"]}')
    print(f'  Invalid movements: {result["invalid"]}')
    print(f'  WAIT actions: {result["waits"]}')
    print('\nBEST SUCCESSFUL CHECKPOINT')
    if best is None:
        print('  No successful greedy checkpoint recorded.')
    else:
        print(f'  First successful evaluation: episode {first_success_episode}')
        print(f'  Best checkpoint episode: {best["episode"]}')
        print(f'  Best makespan: {best["steps"]} steps')
        print(f'  Checkpoint saved: {"YES" if checkpoint_path and Path(checkpoint_path).exists() else "NO"}')
        if checkpoint_path:
            print(f'  File: {checkpoint_path}')
    print('=' * 64)


def train(cfg):
    validate_config(cfg)
    random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    torch.set_num_threads(1)
    env=GridMAPF(cfg)
    env.display()
    actor=Actor(cfg);critic=Critic(cfg)
    actor_opt=torch.optim.Adam(actor.parameters(),lr=cfg.learning_rate)
    critic_opt=torch.optim.Adam(critic.parameters(),lr=cfg.learning_rate)
    expert=expert_dataset(cfg)
    print(f'Architecture: {"attention" if cfg.use_attention else "MLP"} | '
          f'Imitation: {"ON" if expert is not None else "OFF"} | '
          f'Expert joint steps: {cfg.expert_prefix_steps if expert is not None else 0} | '
          f'Examples: {len(expert[1]) if expert is not None else 0} | '
          f'Action mask: {cfg.use_static_action_mask}')
    buffer=empty_buffer();history=[];best=None;last_il=None;first_success_episode=None
    started=time.perf_counter()
    for ep in range(1,cfg.episodes+1):
        env.reset();total=0.;done=False
        for _ in range(cfg.max_steps):
            obs=torch.stack([env.actor_obs(i) for i in range(env.n_agents)])
            joint=env.critic_obs()
            masks=torch.stack([env.mask(i) for i in range(env.n_agents)])
            with torch.no_grad():
                dist=distribution(actor,obs,masks if cfg.use_static_action_mask else None)
                acts=dist.sample()
                logp=dist.log_prob(acts)
            _,reward,done,trunc,_=env.step(acts.tolist())
            buffer['obs'].append(obs);buffer['joint'].append(joint)
            buffer['next_joint'].append(env.critic_obs())
            buffer['actions'].append(acts.tolist());buffer['logp'].append(logp)
            buffer['masks'].append(masks);buffer['rewards'].append(reward)
            buffer['terminals'].append(done);buffer['boundaries'].append(done or trunc)
            total+=reward
            if done or trunc: break
        history.append(dict(reward=total,success=int(done),steps=env.steps,collisions=env.collisions))
        if len(buffer['rewards'])>=cfg.rollout_size:
            last_il=update(actor,critic,actor_opt,critic_opt,buffer,expert,cfg)
            buffer=empty_buffer()
        if ep==1 or ep%cfg.log_interval==0:
            h=history[-cfg.log_interval:]
            il_text=f' | Imitation CE {last_il:.4f}' if last_il is not None else ''
            print(f'Episode {ep:5} | Reward {np.mean([v["reward"] for v in h]):7.1f} | '
                  f'Success {np.mean([v["success"] for v in h])*100:5.1f}% | '
                  f'Steps {np.mean([v["steps"] for v in h]):5.1f}{il_text}')
        if ep%cfg.eval_interval==0:
            greedy=evaluate(actor,cfg)
            count = cfg.eval_episodes
            sampled=[evaluate(actor,cfg,False,seed=cfg.seed+ep*100+i) for i in range(count)]
            success_count=sum(v['success'] for v in sampled)
            rate=100.0*success_count/count
            print(f'  Full-task eval: greedy={greedy["success"]}, steps={greedy["steps"]}, '
                  f'stochastic={rate:.1f}% ({success_count}/{count})')
            if greedy['success']:
                if first_success_episode is None:
                    first_success_episode=ep
                if best is None or greedy['steps']<best['steps']:
                    best={'steps':greedy['steps'],'episode':ep,'actor':copy.deepcopy(actor.state_dict()),
                          'critic':copy.deepcopy(critic.state_dict())}
                    print(f'  NEW BEST SUCCESSFUL CHECKPOINT: episode {ep}, makespan {greedy["steps"]}')
    if buffer['rewards']:
        update(actor,critic,actor_opt,critic_opt,buffer,expert,cfg)
    if best:
        actor.load_state_dict(best['actor']);critic.load_state_dict(best['critic'])
        print(f'Best successful checkpoint: episode {best["episode"]}, makespan {best["steps"]}')
    result=evaluate(actor,cfg)
    folder=Path(cfg.output_dir);folder.mkdir(parents=True,exist_ok=True)
    tag=f'{cfg.grid_size}x{cfg.grid_size}_{env.n_agents}agents_{"attention" if cfg.use_attention else "mlp"}_{"il"+str(cfg.expert_prefix_steps) if cfg.use_imitation else "noil"}_seed{cfg.seed}'
    checkpoint_path = folder / f'{tag}_best_successful.pt' if best is not None else None
    if best is not None:
        torch.save({'actor':best['actor'],'critic':best['critic'], 'config':asdict(cfg),
                    'episode':best['episode'],'makespan':best['steps']},checkpoint_path)
    torch.save({'actor':actor.state_dict(),'critic':critic.state_dict(),'config':asdict(cfg)},folder/f'{tag}.pt')
    display_final_results(result,cfg,best,checkpoint_path,first_success_episode)
    (folder/f'{tag}_history.json').write_text(json.dumps(history,indent=2))
    plot_training(history,folder/f'{tag}_training.png',cfg)
    plot_paths(result['path'],cfg,folder/f'{tag}_paths.png')
    print(f'Time: {time.perf_counter()-started:.1f}s | Saved results: {folder.resolve()}')
    return result


# ============================================================================
# 7. QUICK CHECKS - TEST BEFORE A LONG TRAINING RUN
# ============================================================================
def smoke_tests():
    validate_config(CONFIG)
    env=GridMAPF(CONFIG)
    n=CONFIG.grid_size**2
    assert env.actor_obs(0).shape==(4*n,)
    assert env.critic_obs().shape==((2*env.n_agents+1)*n,)
    expert=expert_dataset(CONFIG)
    if expert is not None:
        assert expert[0].shape==(CONFIG.expert_prefix_steps*env.n_agents,4*n)
    for attention in (False,True):
        cfg=copy.deepcopy(CONFIG);cfg.use_attention=attention
        actor=Actor(cfg);critic=Critic(cfg)
        obs=torch.stack([env.actor_obs(i) for i in range(env.n_agents)])
        assert actor(obs).shape==(env.n_agents,5)
        assert critic(env.critic_obs()).ndim==0
        if expert is not None:
            masks=mask_from_observations(cfg,expert[0])
            loss=-distribution(actor,expert[0],masks).log_prob(expert[1]).mean()
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())
    # Test configurable 3-agent geometry and collision rejection without expert labels.
    cfg=copy.deepcopy(CONFIG)
    cfg.grid_size=5;cfg.starts=((0,0),(0,2),(4,4));cfg.goals=((4,0),(4,2),(0,4))
    cfg.obstacles=();cfg.use_imitation=False;cfg.expert_paths=()
    validate_config(cfg)
    env=GridMAPF(cfg)
    assert env.critic_obs().shape==(7*25,)
    env.step([3,2,4])
    assert env.positions==cfg.starts and env.collisions==2
    print('Smoke tests passed: configuration, N-agent environment, attention/MLP, expert gradients.')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--test-only',action='store_true')
    args=parser.parse_args()
    smoke_tests()
    if not args.test_only: train(CONFIG)
