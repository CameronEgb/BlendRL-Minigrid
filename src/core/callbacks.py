import time
import torch
import numpy as np
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
            for i in range(1, pl_module.cfg.intervals_count + 1):
                target_transitions = i * interval_size
                if current_transitions >= target_transitions and i not in self.logged_intervals:
                    if i == pl_module.cfg.intervals_count:
                        target_transitions = pl_module.cfg.total_timesteps
                    should_eval = True
                    self.logged_intervals.add(i)
                    break
        else:
            epochs_per_interval = pl_module.cfg.agent.get("epochs_per_interval", 1)
            current_epoch = pl_module.current_epoch + 1
            
            current_interval = current_epoch // epochs_per_interval
            target_transitions = interval_size * current_interval
            
            if current_interval > 0 and current_interval not in self.logged_intervals:
                should_eval = True
                self.logged_intervals.add(current_interval)
            
            if eval_interval_epochs and current_epoch % eval_interval_epochs == 0:
                should_eval = True
                target_transitions = interval_size * max(1, current_interval)

        if should_eval:
            self.evaluate_and_log(trainer, pl_module, transitions=target_transitions)

    def evaluate_and_log(self, trainer, pl_module, transitions):
        eval_start = time.time()
        avg_reward, std_reward = self.evaluate(trainer, pl_module)
        eval_end = time.time()
        
        eval_duration = eval_end - eval_start
        self.cumulative_eval_time += eval_duration
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
            
        log_step = trainer.global_step if hasattr(trainer, "global_step") else int(transitions)
        trainer.logger.log_metrics(metrics, step=log_step)
        
        pl_module.log("eval/reward", avg_reward, prog_bar=True, on_step=False, on_epoch=True)
        pl_module.log("transitions", float(transitions), logger=False, prog_bar=True)

        print(f"Evaluation at {transitions} transitions: Avg Reward = {avg_reward} (+/- {std_reward})")

    def evaluate(self, trainer, pl_module):
        cfg = self.cfg
        pl_module.eval()
        
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
            eval_n_envs = cfg.env.get("eval_n_envs", 20) if hasattr(cfg.env, "get") else 20
            target_n_envs = min(cfg.eval_episodes, eval_n_envs)
            self.eval_env = VectorizedNudgeBaseEnv.from_name(
                cfg.env.name, 
                n_envs=target_n_envs, 
                mode=base_algo_name if base_algo_name else cfg.env.name, 
                seed=cfg.seed + 100
            )
        
        eval_total_rewards = []
        n_eval_envs = self.eval_env.n_envs
        eval_cumulative_rewards = np.zeros(n_eval_envs)
        
        logic_obs, obs = self.eval_env.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32, device=pl_module.device)
        logic_obs = torch.as_tensor(logic_obs, dtype=torch.float32, device=pl_module.device)
        
        while len(eval_total_rewards) < cfg.eval_episodes:
            with torch.no_grad():
                res = pl_module.get_action_and_value(obs, logic_obs)
                action = res[0]
            
            (next_logic, next_obs), reward, terminations, truncations, infos = self.eval_env.step(action.cpu().numpy())
            obs = torch.as_tensor(next_obs, dtype=torch.float32, device=pl_module.device)
            logic_obs = torch.as_tensor(next_logic, dtype=torch.float32, device=pl_module.device)
            
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
