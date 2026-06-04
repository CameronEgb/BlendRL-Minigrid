import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import numpy as np
import os
from typing import Any, Dict, Optional
from src.methods.ppo_agent import PPOAgent
from src.blendrl.agents.blender_agent import BlenderActorCritic

class BlendRLAgent(PPOAgent):
    def __init__(self, cfg: Dict[str, Any]):
        # We need to call L.LightningModule.__init__ directly to avoid PPOAgent's 
        # complex __init__ which sets up a separate environment.
        super(PPOAgent, self).__init__() 
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.lr = self.get_cfg("lr", 3e-4)
        self.logic_lr = self.get_cfg("logic_lr", self.lr)
        self.blender_lr = self.get_cfg("blender_lr", self.lr)
        self.num_envs = self.get_cfg("num_envs", 4)
        self.num_steps = self.get_cfg("num_steps", 128)
        self.update_epochs = self.get_cfg("update_epochs", self.get_cfg("ppo_epochs", 10))
        self.batch_size = self.get_cfg("batch_size", 64)
        self.num_minibatches = self.get_cfg("num_minibatches", None)
        if self.num_minibatches is not None:
            self.batch_size = (self.num_envs * self.num_steps) // self.num_minibatches

        self.gamma = self.get_cfg("gamma", 0.99)
        self.gae_lambda = self.get_cfg("gae_lambda", 0.95)
        self.clip_coef = self.get_cfg("clip_coef", 0.2)
        self.ent_coef = self.get_cfg("ent_coef", 0.01)
        self.blend_ent_coef = self.get_cfg("blend_ent_coef", 0.01)
        self.vf_coef = self.get_cfg("vf_coef", 0.5)
        self.max_grad_norm = self.get_cfg("max_grad_norm", 0.5)
        self.anneal_lr = self.get_cfg("anneal_lr", True)
        self.norm_adv = self.get_cfg("norm_adv", True)
        self.clip_vloss = self.get_cfg("clip_vloss", True)
        
        from src.blendrl.env_vectorized import VectorizedNudgeBaseEnv
        self.env = VectorizedNudgeBaseEnv.from_name(
            cfg.env.name, 
            n_envs=self.num_envs, 
            mode=self.get_cfg("algorithm", cfg.env.name), 
            seed=self.get_cfg("seed", cfg.seed)
        )
        
        dummy_logic, dummy_neural = self.env.reset()
        self.observation_space = dummy_neural.shape[1:]
        self.logic_observation_space = dummy_logic.shape[1:]
        self.n_actions = self.env.n_actions if not callable(self.env.n_actions) else self.env.n_actions()
        
        self.model = BlenderActorCritic(
            self.env,
            self.get_cfg("rules", cfg.env.rules),
            self.get_cfg("actor_mode", "hybrid"),
            self.get_cfg("blender_mode", "logic"),
            self.get_cfg("blend_function", "softmax"),
            self.get_cfg("reasoner", cfg.env.reasoner),
            self.device,
            architecture=self.get_cfg("architecture", cfg.env.architecture),
            cfg=cfg.agent
        )
        
        self.automatic_optimization = False
        
        # Storage for rollouts
        self.obs = torch.zeros((self.num_steps, self.num_envs) + self.observation_space)
        self.logic_obs = torch.zeros((self.num_steps, self.num_envs) + self.logic_observation_space)
        self.actions = torch.zeros((self.num_steps, self.num_envs))
        self.logprobs = torch.zeros((self.num_steps, self.num_envs))
        self.rewards = torch.zeros((self.num_steps, self.num_envs))
        self.terminations = torch.zeros((self.num_steps, self.num_envs))
        self.truncations = torch.zeros((self.num_steps, self.num_envs))
        self.values = torch.zeros((self.num_steps, self.num_envs))
        
        self.next_obs = dummy_neural
        self.next_logic_obs = dummy_logic
        self.next_terminated = torch.zeros(self.num_envs)
        self.next_truncated = torch.zeros(self.num_envs)
        
        self.global_step_count = 0

    def get_cfg(self, key, default=None):
        from src.methods.ppo_agent import PPOAgent
        return PPOAgent.get_cfg(self, key, default)

    def get_action_and_value(self, obs, logic_obs=None, action=None):
        return self.model.get_action_and_value(obs, logic_obs, action)

    def get_value(self, obs, logic_obs=None):
        return self.model.get_value(obs, logic_obs)

    def on_fit_start(self):
        self.model.to(self.device)

    def on_train_epoch_start(self):
        # 1. Standard PPO Rollout Collection
        cfg = self.cfg
        if cfg.mode.type == "offline":
            return
            
        # Hard stop if we've reached total timesteps
        if self.global_step_count >= cfg.total_timesteps:
            self.trainer.should_stop = True
            return

        self.model.eval()
        
        # Learning rate annealing
        if self.anneal_lr:
            frac = 1.0 - (self.current_epoch / self.trainer.max_epochs)
            lrnow = frac * self.lr
            logic_lrnow = frac * self.logic_lr
            blender_lrnow = frac * self.blender_lr
            
            # Param groups: 0: policy_modules, 1: logic_critic, 2: blender
            opts = self.optimizers()
            opts.param_groups[0]['lr'] = lrnow
            opts.param_groups[1]['lr'] = lrnow
            opts.param_groups[2]['lr'] = blender_lrnow

        # 2. Check for CEW self-organization at interval starts
        # For online training, we organize at epoch 0 if possible
        if self.current_epoch == 0:
             try:
                 batch = self.trainer.datamodule.reader.sample(1000)
                 organize_obs = batch["logic_obs"] if batch["logic_obs"] is not None else batch["obs"]
                 self.model.self_organize_cew_modules(organize_obs)
                 # Re-init optimizer
                 self.trainer.strategy.optimizers[0] = self.configure_optimizers()
             except:
                 pass # Might not have data yet

        with torch.no_grad():
            for step in range(self.num_steps):
                self.global_step_count += self.num_envs
                self.obs[step] = self.next_obs
                self.logic_obs[step] = self.next_logic_obs
                self.terminations[step] = self.next_terminated
                self.truncations[step] = self.next_truncated
                
                old_logic_obs = self.next_logic_obs
                
                action, logprob, _, _, value = self.model.get_action_and_value(
                    self.next_obs.to(self.device), 
                    self.next_logic_obs.to(self.device)
                )
                self.values[step] = value.flatten().cpu()
                self.actions[step] = action.cpu()
                self.logprobs[step] = logprob.cpu()
                
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
                    
                    if "final_observation" in item:
                        # Extract both neural and logic parts for bootstrapping
                        final_neural = torch.Tensor(item["final_observation"][1]).to(self.device).unsqueeze(0)
                        final_logic = torch.Tensor(item["final_observation"][0]).to(self.device).unsqueeze(0)
                        final_value = self.model.get_value(final_neural, final_logic).item()
                        self.rewards[step][idx] += self.gamma * final_value
                
                if cfg.save_dataset:
                    if not hasattr(self, "dataset_writer"):
                        from src.dataset_utils import DatasetWriter
                        chunk_size = cfg.total_timesteps // cfg.intervals_count
                        save_dir = cfg.dataset_path
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
            next_value = self.model.get_value(
                self.next_obs.to(self.device), 
                self.next_logic_obs.to(self.device)
            ).reshape(1, -1).cpu()
            
            advantages = torch.zeros_like(self.rewards)
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - self.next_terminated.cpu()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.terminations[t + 1].cpu()
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            self.returns = advantages + self.values
            self.advantages = advantages

    def training_step(self, batch, batch_idx):
        b_obs = self.obs.reshape((-1,) + self.observation_space).to(self.device)
        b_logic_obs = self.logic_obs.reshape((-1,) + self.logic_observation_space).to(self.device)
        b_logprobs = self.logprobs.reshape(-1).to(self.device)
        b_actions = self.actions.reshape(-1).to(self.device)
        b_advantages = self.advantages.reshape(-1).to(self.device)
        b_returns = self.returns.reshape(-1).to(self.device)
        b_values = self.values.reshape(-1).to(self.device)
        
        optimizer = self.optimizers()
        b_inds = np.arange(self.num_envs * self.num_steps)
        
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), self.batch_size):
                end = start + self.batch_size
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, blend_entropy, newvalue = self.model.get_action_and_value(
                    b_obs[mb_inds], 
                    b_logic_obs[mb_inds], 
                    b_actions.long()[mb_inds]
                )
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
                
                joint_entropy_loss = -self.ent_coef * entropy.mean() - self.blend_ent_coef * blend_entropy.mean()
                loss = pg_loss + joint_entropy_loss + v_loss * self.vf_coef
                
                optimizer.zero_grad()
                self.manual_backward(loss)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()
                
                self.log("losses/policy_loss", pg_loss)
                self.log("losses/value_loss", v_loss)
                self.log("losses/entropy", entropy.mean())
                self.log("losses/blend_entropy", blend_entropy.mean())
                self.log("losses/total_loss", loss)

    def configure_optimizers(self):
        return optim.Adam([
            {"params": self.model.policy_modules.parameters(), "lr": self.lr},
            {"params": self.model.logic_critic.parameters(), "lr": self.lr},
            {"params": self.model.blender.parameters(), "lr": self.blender_lr},
        ], lr=self.lr, eps=1e-5)

