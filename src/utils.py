from collections import OrderedDict
import os
import torch
import gymnasium as gym
from nsfr.common import get_nsfr_model, get_blender_nsfr_model
from nsfr.utils.common import load_module
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

# from huggingface_sb3 import load_from_hub, push_to_hub


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NeuralBlenderActor(nn.Module):
    """
    Neural Blender Actor;
    a neural network that takes an image as input and outputs a probability distribution over policies.
    """

    def __init__(self, out_size=2):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, out_size), std=0.01)

    def forward(self, x):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        return logits

    def get_action_probs(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


class NeuralBlenderMLP(nn.Module):
    """
    Neural Blender MLP;
    a neural network that takes a vector as input and outputs a probability distribution over policies.
    """

    def __init__(self, num_in_features, out_size=2, hidden_sizes=[64, 64]):
        super().__init__()
        layers = []
        last_size = num_in_features
        for size in hidden_sizes:
            layers.append(layer_init(nn.Linear(last_size, size)))
            layers.append(nn.Tanh())
            last_size = size
        self.network = nn.Sequential(*layers)
        self.actor = layer_init(nn.Linear(last_size, out_size), std=0.01)

    def forward(self, x):
        # Flatten input: (B, ...) -> (B, -1)
        x = x.float().reshape(x.shape[0], -1)
        hidden = self.network(x)
        logits = self.actor(hidden)
        return logits


class CNNActor(nn.Module):
    """
    Neural Blender Actor;
    a neural network that takes an image as input and outputs a probability distribution over actions.
    """

    def __init__(
        self,
        n_actions=18,
    ):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def forward(self, x):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        return probs.probs


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

    def forward(self, x):
        return self.network(x / 255.0)


class MLPQNetwork(nn.Module):
    def __init__(self, n_actions, num_in_features=4):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(num_in_features, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=1.0),
        )

    def forward(self, x):
        return self.network(x.float())


class MLPValueNetwork(nn.Module):
    def __init__(self, num_in_features=4):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(num_in_features, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, x):
        return self.network(x.float())


class QNetwork(nn.Module):
    def __init__(self, n_actions=18):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, n_actions), std=1.0),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def get_neural_agent(env_name, n_actions, device, arch_name=None, hidden_sizes=[64, 64]):
    """
    Get a neural agent based on the environment and architecture name.
    If arch_name is provided, it tries to load that specific architecture.
    Otherwise, it defaults to environment-specific mlp.py or CNNActor.
    """
    if arch_name == "mlp":
        mlp_module_path = f"in/envs/{env_name}/mlp.py"
        if os.path.exists(mlp_module_path):
            module = load_module(mlp_module_path)
            return module.MLP(device=device, out_size=n_actions, hidden_sizes=hidden_sizes).to(device)
        else:
            # Generic MLP if env-specific one doesn't exist? 
            # For now, let's just use a default 64x64 if we can't find it.
            # But the user might prefer an error or fallback.
            pass
    
    if arch_name == "cnn":
        return CNNActor(n_actions=n_actions).to(device)

    # Fallback to existing logic: check for mlp.py, then fallback to CNNActor
    mlp_module_path = f"in/envs/{env_name}/mlp.py"
    if os.path.exists(mlp_module_path):
        module = load_module(mlp_module_path)
        return module.MLP(device=device, out_size=n_actions, hidden_sizes=hidden_sizes).to(device)
    else:
        return CNNActor(n_actions=n_actions).to(device)


def get_blender(
    env,
    blender_rules,
    device,
    train=True,
    blender_mode="logic",
    reasoner="nsfr",
    explain=False,
    out_size=2,
    architecture="cnn",
):
    """
    Load a Blender model.
    Args:
        env (gym.Env): Environment.
        blender_rules (str): Path to Blender rules.
        device (torch.device): Device.
        train (bool): Whether to train the model.
        blender_mode (str): Mode of Blender. Possible values are "logic" and "neural".
        reasoner (str): Reasoner. Possible values are "nsfr" and "neumann".
        explain (bool): Whether to explain the model.
        out_size (int): Number of outputs for neural blender.
        architecture (str): Architecture for neural blender ("cnn" or "mlp").
    Returns:
        Blender: Blender model.
    """
    assert blender_mode in ["logic", "neural"]
    if blender_mode == "logic":
        if reasoner == "nsfr":
            return get_blender_nsfr_model(
                env.name, blender_rules, device, train=train, explain=explain
            )
        elif reasoner == "neumann":
            from neumann.common import get_neumann_model, get_blender_neumann_model

            return get_blender_neumann_model(
                env.name, blender_rules, device, train=train, explain=explain
            )
    if blender_mode == "neural":
        if architecture == "cnn":
            net = NeuralBlenderActor(out_size=out_size)
        else:
            # Assume mlp
            dummy_logic, dummy_neural = env.reset()
            # Calculate total features after flattening
            num_in_features = np.prod(dummy_logic.shape[1:]) 
            net = NeuralBlenderMLP(num_in_features=num_in_features, out_size=out_size)
        net.to(device)
        return net


def load_cleanrl_envs(env_id, run_name=None, capture_video=False, num_envs=1):
    from src.blendrl.env_utils import make_env as apply_wrappers

    def make_env(env_id, seed, capture_video, run_name):
        def thunk():
            env = gym.make(env_id)
            return apply_wrappers(env)
        return thunk

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, i, capture_video, run_name) for i in range(num_envs)],
    )
    return envs


def load_cleanrl_agent(pretrained, device):
    # from cleanrl.cleanrl.ppo_atari import Agent
    agent = CNNActor(n_actions=18)  # , device=device, verbose=1)
    if pretrained:
        try:
            agent.load_state_dict(torch.load("cleanrl/out/ppo_Seaquest-v4_1.pth"))
            agent.to(device)
        except RuntimeError:
            agent.load_state_dict(
                torch.load(
                    "cleanrl/out/ppo_Seaquest-v4_1.pth",
                    map_location=torch.device("cpu"),
                )
            )
    else:
        agent.to(device)
    return agent


def load_logic_ppo(agent, path):
    new_actor_dic = OrderedDict()
    new_critic_dic = OrderedDict()
    dic = torch.load(path)
    for name, value in dic.items():
        if "actor." in name:
            new_name = name.replace("actor.", "")
            new_actor_dic[new_name] = value
        if "critic." in name:
            new_name = name.replace("critic.", "")
            new_critic_dic[new_name] = value
    agent.logic_actor.load_state_dict(new_actor_dic)
    agent.logic_critic.load_state_dict(new_critic_dic)
    return agent


import time
import lightning as L
from src.blendrl.env_vectorized import VectorizedNudgeBaseEnv

class EnvironmentEvaluatorCallback(L.Callback):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.eval_env = None
        self.logged_intervals = set()
        self.train_start_time = None
        self.cumulative_eval_time = 0

    def on_train_start(self, trainer, pl_module):
        self.train_start_time = time.time()
        # Trigger evaluation for point 0
        if 0 not in self.logged_intervals:
            self.evaluate_and_log(trainer, pl_module, transitions=0)
            self.logged_intervals.add(0)

    def on_train_epoch_end(self, trainer, pl_module):
        interval_size = max(1, pl_module.cfg.total_timesteps // pl_module.cfg.intervals_count)
        eval_interval_epochs = pl_module.cfg.agent.get("eval_interval_epochs")
        
        should_eval = False
        target_transitions = 0

        if pl_module.cfg.mode.type == "online":
            current_transitions = pl_module.global_step_count
            # Check all possible intervals up to the current progress
            for i in range(1, pl_module.cfg.intervals_count + 1):
                target_transitions = i * interval_size
                if current_transitions >= target_transitions and i not in self.logged_intervals:
                    if i == pl_module.cfg.intervals_count:
                        target_transitions = pl_module.cfg.total_timesteps
                    should_eval = True
                    self.logged_intervals.add(i)
                    break
        else:
            # Offline Mode
            epochs_per_interval = pl_module.cfg.agent.get("epochs_per_interval", 1)
            current_epoch = pl_module.current_epoch + 1
            
            # 1. Standard Interval Evaluation (Dataset Scaling)
            current_interval = current_epoch // epochs_per_interval
            target_transitions = interval_size * current_interval
            
            if current_interval > 0 and current_interval not in self.logged_intervals:
                should_eval = True
                self.logged_intervals.add(current_interval)
            
            # 2. High-Frequency Evaluation (Convergence Verification)
            # If we are in a verification run (e.g. intervals_count=1)
            if eval_interval_epochs and current_epoch % eval_interval_epochs == 0:
                should_eval = True
                # We use a unique key for frequent evals to not conflict with standard intervals
                # and use current transitions (which might be the full dataset size)
                target_transitions = interval_size * max(1, current_interval)

        if should_eval:
            self.evaluate_and_log(trainer, pl_module, transitions=target_transitions)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Validation is now redundant since we handle everything in on_train_epoch_end
        pass

    def evaluate_and_log(self, trainer, pl_module, transitions):
        eval_start = time.time()
        avg_reward, std_reward = self.evaluate(trainer, pl_module)
        eval_end = time.time()
        
        eval_duration = eval_end - eval_start
        self.cumulative_eval_time += eval_duration
        
        # Force transitions to be a clean integer
        transitions = int(round(transitions))

        metrics = {
            "eval/reward": avg_reward,
            "eval/reward_std": std_reward,
            "transitions": float(transitions)
        }
        
        if self.train_start_time is not None:
            current_total_time = eval_end - self.train_start_time
            pure_training_time = current_total_time - self.cumulative_eval_time
            metrics["time/eval"] = eval_duration
            metrics["time/train"] = pure_training_time
            metrics["time/total"] = current_total_time

        if pl_module.cfg.mode.type == "offline":
            metrics["epoch"] = float(pl_module.current_epoch)
            
        trainer.logger.log_metrics(metrics, step=transitions)
        
        pl_module.log("eval/reward", avg_reward, prog_bar=True, on_step=False, on_epoch=True)
        pl_module.log("transitions", float(transitions), logger=False, prog_bar=True)

        print(f"Evaluation at {transitions} transitions: Avg Reward = {avg_reward} (+/- {std_reward})")

    def evaluate(self, trainer, pl_module):
        cfg = self.cfg
        pl_module.eval()
        
        # Helper to extract algorithm name robustly
        def get_algo_name_robust(acfg):
            from omegaconf import DictConfig
            if isinstance(acfg, (dict, DictConfig)):
                if "algorithm" in acfg:
                    return acfg.algorithm
                if "agent" in acfg:
                    res = get_algo_name_robust(acfg.agent)
                    if res: return res
                if "name" in acfg:
                    return acfg.name
            return None

        base_algo_name = get_algo_name_robust(cfg.agent)

        if self.eval_env is None:
            self.eval_env = VectorizedNudgeBaseEnv.from_name(
                cfg.env.name, 
                n_envs=min(10, cfg.env.num_envs if cfg.mode.type == "online" else 10), 
                mode=base_algo_name if base_algo_name else cfg.env.name, 
                seed=cfg.seed + 100
            )
        
        eval_total_rewards = []
        n_eval_envs = self.eval_env.n_envs
        eval_cumulative_rewards = np.zeros(n_eval_envs)
        
        logic_obs, obs = self.eval_env.reset()
        obs = torch.Tensor(obs).to(pl_module.device)
        logic_obs = torch.Tensor(logic_obs).to(pl_module.device)
        
        # Determine if the agent needs logic_obs
        is_hybrid = base_algo_name and "blendrl" in base_algo_name
        
        while len(eval_total_rewards) < cfg.eval_episodes:
            with torch.no_grad():
                # Call get_action_and_value on the pl_module (agent) directly
                # All agents now have a consistent (obs, logic_obs, action) signature
                res = pl_module.get_action_and_value(obs, logic_obs)
                action = res[0]
            
            (next_logic, next_obs), reward, terminations, truncations, infos = self.eval_env.step(action.cpu().numpy())
            obs = torch.Tensor(next_obs).to(pl_module.device)
            logic_obs = torch.Tensor(next_logic).to(pl_module.device)
            
            for k in range(n_eval_envs):
                eval_cumulative_rewards[k] += reward[k]
                if terminations[k] or truncations[k]:
                    eval_total_rewards.append(eval_cumulative_rewards[k])
                    eval_cumulative_rewards[k] = 0
                    if len(eval_total_rewards) >= cfg.eval_episodes:
                        break
        
        return np.mean(eval_total_rewards), np.std(eval_total_rewards)

    def on_fit_end(self, trainer, pl_module):
        if self.eval_env:
            self.eval_env.close()
