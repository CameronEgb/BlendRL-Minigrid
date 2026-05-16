import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import numpy as np
import os
from typing import Any, Dict, Optional
from blendrl.env_vectorized import VectorizedNudgeBaseEnv

class PPOAgent(L.LightningModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        self.lr = self.get_cfg("lr", 3e-4)
        self.num_envs = self.get_cfg("num_envs", 4)
        self.num_steps = self.get_cfg("num_steps", 128)
        self.update_epochs = self.get_cfg("update_epochs", self.get_cfg("ppo_epochs", 10))
        self.batch_size = self.get_cfg("batch_size", 64) # Minibatch size
        self.num_minibatches = self.get_cfg("num_minibatches", None)
        if self.num_minibatches is not None:
            self.batch_size = (self.num_envs * self.num_steps) // self.num_minibatches

        self.gamma = self.get_cfg("gamma", 0.99)
        self.gae_lambda = self.get_cfg("gae_lambda", 0.95)
        self.clip_coef = self.get_cfg("clip_coef", 0.2)
        self.ent_coef = self.get_cfg("ent_coef", 0.01)
        self.vf_coef = self.get_cfg("vf_coef", 0.5)
        self.max_grad_norm = self.get_cfg("max_grad_norm", 0.5)
        self.anneal_lr = self.get_cfg("anneal_lr", True)
        self.norm_adv = self.get_cfg("norm_adv", True)
        self.clip_vloss = self.get_cfg("clip_vloss", True)

        # We'll initialize the model and environments in setup() or __init__
        from blendrl.env_vectorized import VectorizedNudgeBaseEnv
        self.env = VectorizedNudgeBaseEnv.from_name(
            cfg.env.name, 
            n_envs=self.num_envs, 
            mode=self.get_cfg("algorithm", "ppo"), 
            seed=cfg.seed
        )
        
        # Get action space and observation space
        dummy_logic, dummy_neural = self.env.reset()
        self.observation_space = dummy_neural.shape[1:]
        self.logic_observation_space = dummy_logic.shape[1:]
        self.n_actions = self.env.n_actions if not callable(self.env.n_actions) else self.env.n_actions()
        
        # Define the network
        from src.utils import get_neural_agent
        self.model = get_neural_agent(
            cfg.env.name, 
            self.n_actions, 
            self.device, 
            arch_name=cfg.env.architecture, hidden_sizes=self.get_cfg("hidden_sizes", [64, 64])
        )

        self.automatic_optimization = False
        
        # Storage for rollouts
        self.register_buffer("obs", torch.zeros((self.num_steps, self.num_envs) + self.observation_space))
        self.register_buffer("actions", torch.zeros((self.num_steps, self.num_envs)))
        self.register_buffer("logprobs", torch.zeros((self.num_steps, self.num_envs)))
        self.register_buffer("rewards", torch.zeros((self.num_steps, self.num_envs)))
        self.register_buffer("terminations", torch.zeros((self.num_steps, self.num_envs)))
        self.register_buffer("truncations", torch.zeros((self.num_steps, self.num_envs)))
        self.register_buffer("values", torch.zeros((self.num_steps, self.num_envs)))
        
        self.next_obs = dummy_neural
        self.next_logic_obs = dummy_logic
        self.next_done = torch.zeros(self.num_envs)
        self.next_terminated = torch.zeros(self.num_envs)
        self.next_truncated = torch.zeros(self.num_envs)
        
        self.global_step_count = 0

    def get_cfg(self, key, default=None):
        cfg = self.cfg
        # Check agent sub-config (including nested), then env sub-config, then top level
        if hasattr(cfg, "agent"):
            if key in cfg.agent:
                return cfg.agent[key]
            if "agent" in cfg.agent and key in cfg.agent.agent:
                return cfg.agent.agent[key]
        if hasattr(cfg, "env") and key in cfg.env:
            return cfg.env[key]
        if key in cfg:
            return cfg[key]
        return default
        
    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        # Collect rollout
        cfg = self.cfg
        if cfg.mode.type == "offline":
            return
            
        self.model.eval()
        
        # Learning rate annealing
        if self.anneal_lr:
            frac = 1.0 - (self.current_epoch / self.trainer.max_epochs)
            lrnow = frac * self.lr
            for pg in self.optimizers().param_groups:
                pg['lr'] = lrnow
        
        # Ensure next_obs/done are on the correct device
        self.next_obs = self.next_obs.to(self.device)
        self.next_terminated = self.next_terminated.to(self.device)
        self.next_truncated = self.next_truncated.to(self.device)

        with torch.no_grad():
            for step in range(self.num_steps):
                self.global_step_count += self.num_envs
                self.obs[step] = self.next_obs
                self.terminations[step] = self.next_terminated
                self.truncations[step] = self.next_truncated
                
                old_logic_obs = self.next_logic_obs
                
                # Pure PPO (neural only) for now
                action, logprob, _, value = self.model.get_action_and_value(self.next_obs)
                self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob
                
                (real_next_logic, real_next_obs), reward, terminations, truncations, infos = self.env.step(action.cpu().numpy())
                
                self.rewards[step] = torch.tensor(reward, device=self.next_obs.device).view(-1)
                
                self.next_obs = torch.Tensor(real_next_obs).to(self.next_obs.device)
                self.next_logic_obs = torch.Tensor(real_next_logic).to(self.next_obs.device)
                self.next_terminated = torch.Tensor(terminations).to(self.next_obs.device)
                self.next_truncated = torch.Tensor(truncations).to(self.next_obs.device)

                for idx, item in enumerate(infos):
                    if "episode" in item.keys():
                        ep_reward = item["episode"]["r"]
                        ep_length = item["episode"]["l"]
                        transitions_to_log = min(float(self.global_step_count), float(self.cfg.total_timesteps))
                        self.logger.log_metrics(
                            {"train/reward": float(ep_reward), "train/length": float(ep_length), "transitions": transitions_to_log},
                            step=self.global_step_count
                        )
                        print(f"global_step={self.global_step_count}, episodic_return={ep_reward}")
                    
                    # Bootstrapping for truncations: Handle the case where the episode was truncated
                    # by injecting the final observation's value into the advantages calculation.
                    # We store the final value in the info dict if we want to be precise.
                    if "final_observation" in item:
                        final_obs = torch.Tensor(item["final_observation"]).to(self.device).unsqueeze(0)
                        final_value = self.model.get_value(final_obs).item()
                        # This is a bit tricky to store in the standard rollout buffer.
                        # For now, we'll rely on the fact that next_obs is the start of the next episode,
                        # which is not perfect but better than 0. 
                        # A better way is to modify the reward: r = r + gamma * V(final_obs)
                        self.rewards[step][idx] += self.gamma * final_value
                
                if cfg.save_dataset:
                    if not hasattr(self, "dataset_writer"):
                        from src.dataset_utils import DatasetWriter
                        chunk_size = cfg.total_timesteps // cfg.intervals_count
                        save_dir = os.path.join("results/datasets", cfg.experiment_id, cfg.agent.name)
                        self.dataset_writer = DatasetWriter(
                            save_dir=save_dir,
                            env_name=cfg.env.name,
                            chunk_size=chunk_size
                        )
                    self.dataset_writer.batch_add(
                        obs=self.obs[step],
                        logic_obs=old_logic_obs,
                        action=action,
                        reward=reward,
                        next_obs=self.next_obs,
                        next_logic_obs=self.next_logic_obs,
                        done=np.logical_or(terminations, truncations)
                    )
                
        self.model.train()
        
        # Compute advantages
        with torch.no_grad():
            next_value = self.model.get_value(self.next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards)
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - self.next_terminated
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.terminations[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            
            self.returns = advantages + self.values
            self.advantages = advantages

    def training_step(self, batch, batch_idx):
        # Flatten rollout
        b_obs = self.obs.reshape((-1,) + self.observation_space)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape(-1)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)
        
        optimizer = self.optimizers()
        b_inds = np.arange(self.num_envs * self.num_steps)
        
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), self.batch_size):
                end = start + self.batch_size
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = self.model.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                
                optimizer.zero_grad()
                self.manual_backward(loss)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()
                
                self.log("losses/policy_loss", pg_loss)
                self.log("losses/value_loss", v_loss)
                self.log("losses/entropy", entropy_loss)
                self.log("losses/total_loss", loss)

    def on_train_end(self):
        if hasattr(self, "dataset_writer"):
            self.dataset_writer.close()
            
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5)


    def validation_step(self, batch, batch_idx):
        # validation_step now only handles calculating validation loss on a held-out batch
        # For PPO online, this is just a placeholder.
        pass
