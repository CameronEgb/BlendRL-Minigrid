import random
import pickle
from pathlib import Path
import os
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from nudge.agents.logic_agent import NsfrActorCritic
from nudge.agents.neural_agent import NeuralPPO, ActorCritic
from nudge.torch_utils import softor

# from nudge.env import NudgeBaseEnv
from torch.distributions.categorical import Categorical
from nsfr.utils.common import load_module
from nsfr.common import get_nsfr_model

from src.utils import get_blender, load_cleanrl_agent, get_neural_agent
from nudge.utils import print_program

from src.methods.cew_utils import run_CLIP, run_ECM, rule_creation, run_FYD, MultiFLC

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)


class BlenderActor(nn.Module):
    """
    BlendeRL actor that combines heterogeneous policy modules.

    Args:
        env: environment
        policy_modules: list of policy modules (NSFR, CEW, Neural, etc.)
        module_types: list of types for each module
        blender: blending policy
        actor_mode: actor mode, one of ["hybrid", "logic", "neural"]
        blender_mode: blender mode, one of ["logic", "neural"]
        blend_function: blending function, one of ["softmax", "gumbel_softmax"]
        device: device
    """

    def __init__(
        self,
        env,
        policy_modules,
        module_types,
        blender,
        actor_mode,
        blender_mode,
        blend_function,
        device=None,
        explain=False,
    ):
        """
        Initialize a BlendeRL agent.
        """
        super(BlenderActor, self).__init__()
        self.env = env
        self.policy_modules = nn.ModuleList(policy_modules)
        self.module_types = module_types
        self.blender = blender
        self.actor_mode = actor_mode
        self.blender_mode = blender_mode
        self.blend_function = blend_function
        self.device = device
        self.explain = explain
        
        # Build mappings for logic-based blender if needed
        self.blender_id_to_pred_indices = self._build_blender_id_dict()

    def _build_blender_id_dict(self):
        """
        Initialize a dictionary that maps blender mode id to predicate indices.
        Returns:
            blender_id_to_pred_indices: dictionary that maps blender mode id to predicate indices
        """
        if self.blender_mode == "neural":
            return {}
            
        blender_mode_names = [f"agent_{i}" for i in range(len(self.policy_modules))]
        # Compatibility with legacy names if only 2 modules (neural, logic)
        if len(self.policy_modules) == 2:
            blender_mode_names = ["neural_agent", "logic_agent"]
            
        blender_id_to_pred_indices = {i: [] for i in range(len(blender_mode_names))}
        
        if hasattr(self.blender, "get_prednames"):
            for j, pred_name in enumerate(self.blender.get_prednames()):
                for i, mode_name in enumerate(blender_mode_names):
                    if mode_name in pred_name:
                        blender_id_to_pred_indices[i].append(j)
        return blender_id_to_pred_indices

    def _map_logic_output(self, q, module):
        action_names = self.env.get_action_meanings()
        
        if hasattr(module, "prednames"):
            mapped_q = torch.zeros(q.size(0), len(action_names), device=q.device)
            for idx, action_name in enumerate(action_names):
                if action_name in module.prednames:
                    pred_idx = module.prednames.index(action_name)
                    mapped_q[:, idx] = q[:, pred_idx]
        elif q.size(1) == len(action_names):
            mapped_q = q
        else:
            return q
                
        # Normalize into a valid action probability distribution:
        # If any action predicate is active, normalize proportionally across valid action candidates.
        # If no action predicate is active (sum == 0, meaning logic is silent/unmatched):
        # Default to safe action (e.g. "withhold" / "noop") rather than uniform 50/50 random.
        sum_q = mapped_q.sum(dim=-1, keepdim=True)
        active_mask = (sum_q > 1e-6)
        
        if "withhold" in action_names:
            default_probs = torch.zeros_like(mapped_q)
            default_probs[:, action_names.index("withhold")] = 1.0
        elif "noop" in action_names:
            default_probs = torch.zeros_like(mapped_q)
            default_probs[:, action_names.index("noop")] = 1.0
        else:
            default_probs = torch.full_like(mapped_q, 1.0 / len(action_names))
            
        normalized_q = torch.where(active_mask, mapped_q / torch.clamp(sum_q, min=1e-6), default_probs)
        return normalized_q

    def get_explanation(self, neural_state, logic_state, action):
        """
        Get the explanation of the blending weights.
        """
        # TODO: Update for heterogeneous modules
        return None, None, self.to_blender_policy_distribution(neural_state, logic_state)[0].detach().cpu().numpy()

    def compute_action_probs_hybrid(self, neural_state, logic_state):
        """
        Compute action probabilities by blending all modules.
        """
        batch_size = neural_state.size(0)
        module_probs = []
        
        for i, module in enumerate(self.policy_modules):
            m_type = self.module_types[i]
            if m_type == "neural":
                probs = module.get_action_probs(neural_state)
            else:
                # logic or cew
                probs = self._map_logic_output(module.get_action_probs(logic_state), module)
            module_probs.append(probs)
            
        # weights size: B * N_modules
        weights = self.to_blender_policy_distribution(neural_state, logic_state)
        self.w_policy = weights[0]
        
        action_probs = torch.zeros(batch_size, self.env.n_actions, device=neural_state.device)
        for i, m_probs in enumerate(module_probs):
            action_probs += weights[:, i].unsqueeze(1) * m_probs.to(neural_state.device)
            
        return action_probs, weights

    def compute_action_probs_logic(self, logic_state):
        """
        Compute action probabilities using only non-neural modules.
        """
        # Determine expected neural input size from blender if possible
        n_in = 1
        if hasattr(self.blender, "network") and len(self.blender.network) > 0:
            n_in = self.blender.network[0].in_features
        elif hasattr(self.blender, "fc"): # logic blender
            n_in = 1 # dummy
            
        dummy_neural = torch.zeros(logic_state.size(0), n_in).to(logic_state.device) 
        weights = self.to_blender_policy_distribution(dummy_neural, logic_state)
        
        # Zero out neural modules and re-normalize
        for i, m_type in enumerate(self.module_types):
            if m_type == "neural":
                weights[:, i] = 0.0
        
        weights_sum = weights.sum(dim=1, keepdim=True)
        weights = weights / torch.clamp(weights_sum, min=1e-12)
        self.w_policy = weights[0]
        
        action_probs = torch.zeros(logic_state.size(0), self.env.n_actions, device=logic_state.device)
        for i, module in enumerate(self.policy_modules):
            if self.module_types[i] != "neural":
                probs = self._map_logic_output(module.get_action_probs(logic_state), module)
                action_probs += weights[:, i].unsqueeze(1) * probs
            
        return action_probs, weights

    def compute_action_probs_neural(self, neural_state):
        """
        Compute action probabilities using only neural modules.
        """
        # Determine expected logic input size from blender
        l_in = 1
        if hasattr(self.blender, "network") and len(self.blender.network) > 0:
             l_in = self.blender.network[0].in_features
        
        dummy_logic = torch.zeros(neural_state.size(0), l_in).to(neural_state.device)
        weights = self.to_blender_policy_distribution(neural_state, dummy_logic)
        
        # Zero out logic modules and re-normalize
        for i, m_type in enumerate(self.module_types):
            if m_type != "neural":
                weights[:, i] = 0.0
        
        weights_sum = weights.sum(dim=1, keepdim=True)
        weights = weights / torch.clamp(weights_sum, min=1e-12)
        self.w_policy = weights[0]
        
        action_probs = torch.zeros(neural_state.size(0), self.env.n_actions, device=neural_state.device)
        for i, module in enumerate(self.policy_modules):
            if self.module_types[i] == "neural":
                probs = module.get_action_probs(neural_state)
                action_probs += weights[:, i].unsqueeze(1) * probs
                
        return action_probs, weights

    def to_blender_policy_distribution(self, neural_state, logic_state):
        """
        Merge policies using the blender function.
        """
        if self.blender_mode == "logic":
            policy_probs = self.blender(logic_state)
            batch_size = policy_probs.size(0)
            mode_probs = []
            n_modes = len(self.policy_modules)
            for i in range(n_modes):
                indices = torch.tensor(self.blender_id_to_pred_indices.get(i, []), device=policy_probs.device)
                if len(indices) == 0:
                    mode_probs.append(torch.zeros(batch_size, 1, device=policy_probs.device))
                    continue
                indices = indices.expand(batch_size, -1)
                gathered = torch.gather(policy_probs, 1, indices)
                merged = softor(gathered, dim=1)
                mode_probs.append(merged)
            
            probs = torch.stack(mode_probs, dim=1).squeeze(-1)
            logits = torch.logit(probs, eps=0.01)
        else:
            # Neural blender
            if len(neural_state.shape) == 2: # vector
                 logits = self.blender(logic_state)
            else:
                 logits = self.blender(neural_state)
        
        if self.blend_function == "softmax":
            return torch.softmax(logits, dim=1)
        else:
            return F.gumbel_softmax(logits, dim=1, hard=True)

    def forward(self, neural_state, logic_state):
        if self.actor_mode == "hybrid":
            return self.compute_action_probs_hybrid(neural_state, logic_state)
        elif self.actor_mode == "logic":
            return self.compute_action_probs_logic(logic_state)
        else:
            return self.compute_action_probs_neural(neural_state)

    def get_q_values(self, neural_state, logic_state=None):
        """Compute Q-values by blending Q-values from all modules."""
        if logic_state is None:
            if neural_state.ndim == 2:
                logic_state = neural_state.unsqueeze(1).repeat(1, 2, 1)
            else:
                logic_state = neural_state
        batch_size = neural_state.size(0)
        module_q_values = []
        
        for i, module in enumerate(self.policy_modules):
            m_type = self.module_types[i]
            if m_type == "neural":
                if hasattr(module, "get_q_values"):
                    q = module.get_q_values(neural_state)
                elif hasattr(module, "forward"):
                    q = module(neural_state) # Assuming forward returns Q-values for Q-networks
                else:
                    q = torch.zeros(batch_size, self.env.n_actions, device=neural_state.device)
            else:
                # logic or cew
                if hasattr(module, "get_q_values"):
                    q = module.get_q_values(logic_state)
                elif m_type == "cew":
                    q = module(logic_state) # MultiFLC forward returns Q-values
                else:
                    # Logic modules usually return probs, treat as Q-values [0, 1]
                    q = self._map_logic_output(module.get_action_probs(logic_state), module)
            module_q_values.append(q)
            
        weights = self.to_blender_policy_distribution(neural_state, logic_state)
        
        q_values = torch.zeros(batch_size, self.env.n_actions, device=neural_state.device)
        for i, m_q in enumerate(module_q_values):
            q_values += weights[:, i].unsqueeze(1) * m_q.to(neural_state.device)
            
        return q_values


class BlenderActorCritic(nn.Module):
    """
    BlendeRL actor-critic that supports heterogeneous policy modules.
    """

    def __init__(
        self,
        env,
        rules,
        actor_mode,
        blender_mode,
        blend_function,
        reasoner,
        device,
        architecture=None,
        rng=None,
        explain=False,
        modules=None,
        cfg=None, # For accessing other agent hyperparams
    ):
        super(BlenderActorCritic, self).__init__()
        self.device = device
        self.rng = random.Random() if rng is None else rng
        self.env = env
        self.cfg = cfg
        hidden_sizes = [64, 64]
        if cfg:
            if "hidden_sizes" in cfg:
                hidden_sizes = list(cfg["hidden_sizes"])
            elif "agent" in cfg and "hidden_sizes" in cfg["agent"]:
                hidden_sizes = list(cfg["agent"]["hidden_sizes"])
        
        self.actor_mode = actor_mode
        self.blender_mode = blender_mode
        self.blend_function = blend_function
        self.reasoner = reasoner
        self.architecture = architecture
        self.explain = explain

        self.policy_modules = nn.ModuleList()
        self.module_types = []
        
        dummy_logic, dummy_neural = env.reset()
        neural_in_features = dummy_neural.shape[-1]

        # 1. Parse modules from argument or config
        modules_list = modules if modules is not None else (cfg.get("modules") if cfg and "modules" in cfg else None)
        
        if modules_list:
            for m_cfg in modules_list:
                m_type = m_cfg.type
                if m_type == "nsfr" or m_type == "neumann":
                    m_rules = m_cfg.rules
                    if self.reasoner == "neumann":
                        from neumann.common import get_neumann_model
                        m = get_neumann_model(env.name, m_rules, device=device, train=True, explain=self.explain)
                    else:
                        m = get_nsfr_model(env.name, m_rules, device=device, train=True, explain=self.explain)
                    self.policy_modules.append(m)
                    self.module_types.append("logic")
                elif m_type == "cew":
                    # Placeholder CEW module, will be self-organized later
                    # Determine input size from env
                    n_inputs = np.prod(dummy_logic.shape[1:])
                    m = MultiFLC(
                        n_inputs=n_inputs, 
                        n_outputs=env.n_actions,
                        antecedents=[],
                        rules=[]
                    ).to(device)
                    self.policy_modules.append(m)
                    self.module_types.append("cew")
                elif m_type == "neural":
                    m = get_neural_agent(env.name, env.n_actions, device, arch_name=self.architecture, hidden_sizes=hidden_sizes, num_in_features=neural_in_features)
                    self.policy_modules.append(m)
                    self.module_types.append("neural")
        else:
            # Backward compatibility with 'rules' string
            if isinstance(rules, str) and "," in rules:
                rulesets = [r.strip() for r in rules.split(",")]
            elif isinstance(rules, list):
                rulesets = rules
            else:
                rulesets = [rules]
            
            # Add Neural module first
            self.policy_modules.append(get_neural_agent(env.name, env.n_actions, device, arch_name=self.architecture, hidden_sizes=hidden_sizes, num_in_features=neural_in_features))
            self.module_types.append("neural")
            
            # Add Logic modules
            for r in rulesets:
                if self.reasoner == "neumann":
                    from neumann.common import get_neumann_model
                    la = get_neumann_model(env.name, r, device=device, train=True, explain=self.explain)
                else:
                    la = get_nsfr_model(env.name, r, device=device, train=True, explain=self.explain)
                self.policy_modules.append(la)
                self.module_types.append("logic")

        out_size = len(self.policy_modules)
        
        # Use first logic module's rules for blender if logic-based
        blender_rules = rulesets[0] if 'rulesets' in locals() else (rules if isinstance(rules, str) else rules[0])
        
        self.blender = get_blender(
            env,
            blender_rules,
            device,
            blender_mode=self.blender_mode,
            train=True,
            explain=self.explain,
            out_size=out_size,
            architecture=self.architecture if self.architecture else "cnn"
        )
        
        # Load logic critic (MLP)
        mlp_module_path = f"in/envs/{env.name}/mlp.py"
        if os.path.exists(mlp_module_path):
            module = load_module(mlp_module_path)
            mlp_cls = getattr(module, "StandardMLP", getattr(module, "MLP", None))
            if mlp_cls:
                self.logic_critic = mlp_cls(device=device, out_size=1, logic=True, hidden_sizes=hidden_sizes)
            else:
                self.logic_critic = None
        else:
            self.logic_critic = None 

        self.actor = BlenderActor(
            env,
            self.policy_modules,
            self.module_types,
            self.blender,
            self.actor_mode,
            self.blender_mode,
            self.blend_function,
            device=device,
        )

    def get_cfg(self, key, default=None):
        """Helper to get a config value from either cfg or cfg.agent."""
        if self.cfg is None: return default
        if key in self.cfg: return self.cfg[key]
        if "agent" in self.cfg and key in self.cfg["agent"]:
            return self.cfg["agent"][key]
        return default

    def self_organize_cew_modules(self, dataset_sample_obs):
        """Triggers self-organization for any CEW modules in the architecture.
        Returns True if any module was physically replaced (architecture changed).
        """
        any_changed = False
        for i, m in enumerate(self.policy_modules):
            if self.module_types[i] == "cew":
                print(f"Self-organizing CEW module {i}...")
                obs = dataset_sample_obs.cpu().numpy()
                # Flatten obs if it has more than 2 dimensions (B, entities, features) -> (B, entities*features)
                if len(obs.shape) > 2:
                    obs = obs.reshape(obs.shape[0], -1)
                
                mins = obs.min(axis=0)
                maxes = obs.max(axis=0)
                
                # CLIP
                antecedents = run_CLIP(obs, mins, maxes)
                # ECM
                dthr = self.get_cfg("ecm_dthr", 0.05)
                clusters = run_ECM(obs, [], dthr)
                reduced_X = np.array([c.center for c in clusters])
                # WM
                antecedents, rules = rule_creation(reduced_X, antecedents)
                
                # FYD (optional)
                if self.get_cfg("fyd", False):
                    top_k = self.get_cfg("fyd_top_k", None)
                    rules, antecedents = run_FYD(rules, obs, antecedents, top_k=top_k)
                
                # Check if architecture changed
                current_rules = getattr(m.flcs[0], "links", None)
                if current_rules is not None and current_rules.shape[1] == len(rules):
                    # Simple heuristic: if rule count is same, check if antecedents count is same
                    if m.flcs[0].transformed_len == sum(len(p_ants) for p_ants in antecedents):
                        print(f"CEW module {i} architecture stable ({len(rules)} rules). Skipping reset.")
                        continue

                # Re-initialize MultiFLC in place
                from src.methods.cew_utils import MultiFLC
                n_in = np.prod(obs.shape[1:])
                # Determine current device from existing parameters
                current_device = next(self.parameters()).device
                new_m = MultiFLC(
                    n_inputs=n_in,
                    n_outputs=self.env.n_actions,
                    antecedents=antecedents,
                    rules=rules
                ).to(current_device)
                
                # Replace the module in ModuleList
                self.policy_modules[i] = new_m
                # Also update the actor's reference
                self.actor.policy_modules[i] = new_m
                print(f"CEW module {i} self-organized with {len(rules)} rules. Weights reset.")
                any_changed = True
        return any_changed

    def forward(self, neural_state, logic_state=None, action=None):
        return self.get_action_and_value(neural_state, logic_state, action=action)

    def get_action_and_value(self, neural_state, logic_state=None, action=None):
        if logic_state is None and neural_state.ndim == 2:
            logic_state = neural_state.unsqueeze(1).repeat(1, 2, 1)
        action_probs, blending_weights = self.actor(neural_state, logic_state)
        dist = Categorical(action_probs)
        blend_dist = Categorical(blending_weights)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)

        blended_value = self.get_value(neural_state, logic_state, blending_weights=blending_weights)

        return action, logprob, dist.entropy(), blend_dist.entropy(), blended_value

    def get_q_values(self, neural_state, logic_state=None):
        # Extract well-calibrated continuous Q-values from neural module for Bellman updates
        for i, module in enumerate(self.policy_modules):
            if self.module_types[i] == "neural":
                if hasattr(module, "get_q_values"):
                    return module.get_q_values(neural_state)
                elif hasattr(module, "actor"):
                    x = neural_state.float().reshape(neural_state.shape[0], -1)
                    hidden = module.network(x)
                    return module.actor(hidden)
                elif hasattr(module, "forward"):
                    return module(neural_state)
        return self.actor.get_q_values(neural_state, logic_state)

    def get_value(self, neural_state, logic_state, blending_weights=None):
        if blending_weights is None:
            _, blending_weights = self.actor(neural_state, logic_state)
            
        neural_value = self.get_neural_value(neural_state).squeeze(1)
        logic_value = self.get_logic_value(logic_state).squeeze(1)
        
        # Weighted value blending
        # w_neural_sum * V_neural + w_logic_sum * V_logic
        neural_weight_sum = 0
        logic_weight_sum = 0
        for i, m_type in enumerate(self.module_types):
            if m_type == "neural":
                neural_weight_sum += blending_weights[:, i]
            else:
                logic_weight_sum += blending_weights[:, i]
        
        blended_value = (
            neural_weight_sum * neural_value
            + logic_weight_sum * logic_value
        ).unsqueeze(1)
        return blended_value

    def get_neural_value(self, neural_state):
        # Find the first neural module to use its critic
        for i, m in enumerate(self.policy_modules):
            if self.module_types[i] == "neural":
                return m.get_value(neural_state)
        return torch.zeros(neural_state.size(0), 1, device=neural_state.device)

    def get_logic_value(self, logic_state):
        if self.logic_critic:
            if hasattr(self.logic_critic, "num_in_features"):
                flat_size = np.prod(logic_state.shape[1:])
                if flat_size != self.logic_critic.num_in_features and logic_state.ndim > 2:
                    if logic_state.shape[-1] == self.logic_critic.num_in_features:
                        logic_state = logic_state[:, 0, :]
            return self.logic_critic(logic_state)
        return torch.zeros(logic_state.size(0), 1, device=logic_state.device)

    def save(self, checkpoint_path, directory: Path, step_list, reward_list, weight_list):
        torch.save(self.state_dict(), checkpoint_path)
        with open(directory / "data.pkl", "wb") as f:
            pickle.dump(step_list, f)
            pickle.dump(reward_list, f)
            pickle.dump(weight_list, f)
