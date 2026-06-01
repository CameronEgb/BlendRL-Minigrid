import random
import pickle
from pathlib import Path
import os


import torch
import torch.nn as nn
import torch.nn.functional as F
from nudge.agents.logic_agent import NsfrActorCritic
from nudge.agents.neural_agent import NeuralPPO, ActorCritic
from nudge.torch_utils import softor

# from nudge.env import NudgeBaseEnv
from torch.distributions.categorical import Categorical


from torch.distributions import Categorical
from nsfr.utils.common import load_module
from nsfr.common import get_nsfr_model

from src.utils import get_blender, load_cleanrl_agent, get_neural_agent
from nudge.utils import print_program

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
    BlendeRL actor that combines neural and multiple logic policies.

    Args:
        env: environment
        neural_actor: neural policy
        logic_actors: list of logic policies
        blender: blending policy
        actor_mode: actor mode, one of ["hybrid", "logic", "neural"]
        blender_mode: blender mode, one of ["logic", "neural"]
        blend_function: blending function, one of ["softmax", "gumbel_softmax"]
        device: device
    """

    def __init__(
        self,
        env,
        neural_actor,
        logic_actors,
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
        self.neural_actor = neural_actor
        if isinstance(logic_actors, nn.ModuleList):
            self.logic_actors = logic_actors
        elif isinstance(logic_actors, nn.Module):
            self.logic_actors = nn.ModuleList([logic_actors])
        else:
            self.logic_actors = nn.ModuleList(logic_actors)
        self.blender = blender
        self.actor_mode = actor_mode
        self.blender_mode = blender_mode
        self.blend_function = blend_function
        self.device = device
        self.explain = explain
        
        # Build action mappings for each logic actor
        self.logic_actor_mappings = [self._build_action_id_dict(la) for la in self.logic_actors]
        self.blender_id_to_pred_indices = self._build_blender_id_dict()

    def _build_blender_id_dict(self):
        """
        Initialize a dictionary that maps blender mode id to predicate indices.
        0: neural, 1: logic_1, 2: logic_2, ...
        Returns:
            blender_id_to_pred_indices: dictionary that maps blender mode id to predicate indices
        """
        if self.blender_mode == "neural":
            return {} # Not needed for neural blender
            
        blender_mode_names = ["neural_agent"] + [f"logic_agent_{i}" for i in range(len(self.logic_actors))]
        # Compatibility with legacy single logic actor
        if len(self.logic_actors) == 1:
            blender_mode_names = ["neural_agent", "logic_agent"]
            
        blender_id_to_pred_indices = {i: [] for i in range(len(blender_mode_names))}
        
        for j, pred_name in enumerate(self.blender.get_prednames()):
            for i, mode_name in enumerate(blender_mode_names):
                # Use robust matching for names like 'logic_agent/1'
                if mode_name in pred_name:
                    blender_id_to_pred_indices[i].append(j)
        return blender_id_to_pred_indices

    def _build_action_id_dict(self, logic_actor):
        """
        Initialize a dictionary that maps environment action id to predicate indices for a specific logic actor.
        Returns:
            env_action_id_to_action_pred_indices: dictionary that maps environment action id to predicate indices
        """
        env_action_names = list(self.env.pred2action.keys())
        env_action_id_to_action_pred_indices = {}
        for i, env_action_name in enumerate(env_action_names):
            env_action_id_to_action_pred_indices[i] = []

        atoms = logic_actor.atoms
        for i, env_action_name in enumerate(env_action_names):
            exist_flag = False
            for j, atom in enumerate(atoms):
                if env_action_name == atom.pred.name:
                    env_action_id_to_action_pred_indices[i].append(j)
                    exist_flag = True
            if not exist_flag:
                # Map to a dummy index (last index + 1)
                dummy_index = len(atoms)
                env_action_id_to_action_pred_indices[i].append(dummy_index)
        return env_action_id_to_action_pred_indices

    def get_explanation(self, neural_state, logic_state, action):
        """
        Get the explanation of the blending weights.
        """
        # TODO: Update for multiple logic actors
        neural_explanation = self.get_neural_explanation(neural_state, action)
        # For now, just explain the first logic actor
        logic_explanation = self.get_logic_explanation(logic_state, action, 0)
        weights = self.to_blender_policy_distribution(neural_state, logic_state)[0]
        return neural_explanation, logic_explanation, weights.detach().cpu().numpy()

    def get_neural_explanation(self, neural_state, action):
        self.neural_actor.eval()
        baseline = torch.zeros_like(neural_state).to(neural_state.device)
        ig = IntegratedGradients(self.neural_actor)
        attributions, delta = ig.attribute(
            neural_state, baseline, target=action, return_convergence_delta=True
        )
        minimum = attributions.min()
        maximum = attributions.max()
        attributions = (attributions - minimum) / (maximum - minimum)
        return attributions

    def get_logic_explanation(self, logic_state, action, logic_idx=0):
        self.raw_action_probs_list[logic_idx].max().backward()
        atom_attributes = self.logic_actors[logic_idx].dummy_zeros.grad
        # normalize to [0, 1]
        minimum = atom_attributes.min()
        maximum = atom_attributes.max()
        atom_attributes = (atom_attributes - minimum) / (maximum - minimum)
        self.logic_actors[logic_idx].dummy_zeros.grad.zero_()
        return atom_attributes

    def compute_action_probs_hybrid(self, neural_state, logic_state):
        """
        Compute action probabilities for hybrid actor.
        """
        batch_size = neural_state.size(0)
        neural_action_probs = self.to_neural_action_distribution(neural_state)
        
        self.logic_action_probs_list = []
        for i, logic_actor in enumerate(self.logic_actors):
            # We call logic_actor(logic_state) to perform reasoning, 
            # but we use V_T directly to avoid prednames errors
            try:
                logic_actor(logic_state)
            except AssertionError as e:
                # Catch "right not found" or similar if prednames check fails inside forward
                if "not found" not in str(e):
                    raise e
            
            if hasattr(logic_actor, "get_all_valuations"):
                V_T = logic_actor.get_all_valuations()
            else:
                V_T = logic_actor.V_T
            
            probs = self.to_action_distribution_from_valuation(V_T, i)
            self.logic_action_probs_list.append(probs)
            
        self.neural_action_probs = neural_action_probs
        
        # weights size: B * (1 + N_logic)
        weights = self.to_blender_policy_distribution(neural_state, logic_state)
        self.w_policy = weights[0]
        
        action_probs = weights[:, 0].unsqueeze(1) * neural_action_probs
        for i, logic_probs in enumerate(self.logic_action_probs_list):
            action_probs += weights[:, i+1].unsqueeze(1) * logic_probs
            
        return action_probs, weights

    def compute_action_probs_logic(self, logic_state):
        """
        Compute action probabilities using only logic actors (combined).
        """
        # Create a dummy neural state
        dummy_neural = torch.zeros(1, 1).to(self.device) 
        weights = self.to_blender_policy_distribution(dummy_neural, logic_state)
        # Zero out neural weight and re-normalize
        weights[:, 0] = 0.0
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        self.w_policy = weights[0]
        
        action_probs = torch.zeros(logic_state.size(0), self.env.n_actions, device=logic_state.device)
        for i, logic_actor in enumerate(self.logic_actors):
            try:
                logic_actor(logic_state)
            except AssertionError:
                pass
            
            if hasattr(logic_actor, "get_all_valuations"):
                V_T = logic_actor.get_all_valuations()
            else:
                V_T = logic_actor.V_T
            
            probs = self.to_action_distribution_from_valuation(V_T, i)
            action_probs += weights[:, i+1].unsqueeze(1) * probs
            
        return action_probs, weights

    def compute_action_probs_neural(self, neural_state):
        """
        Compute action probabilities using only neural actor.
        """
        weights = torch.zeros(neural_state.size(0), 1 + len(self.logic_actors), device=neural_state.device)
        weights[:, 0] = 1.0
        self.w_policy = weights[0]
        neural_action_probs = self.to_neural_action_distribution(neural_state)
        return neural_action_probs, weights

    def to_blender_policy_distribution(self, neural_state, logic_state):
        """
        Merge neural and logic policies using the blender funciton.
        """
        assert self.blender_mode in ["logic", "neural"]
        assert self.blend_function in ["softmax", "gumbel_softmax"]

        if self.blender_mode == "logic":
            policy_probs = self.blender(logic_state)
            
            # Map logic predicates to blender modes
            batch_size = policy_probs.size(0)
            mode_probs = []
            n_modes = 1 + len(self.logic_actors)
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
            # Neural blender returns logits directly now
            if len(neural_state.shape) == 2: # vector
                 logits = self.blender(logic_state) # MLP blender takes logic_state (vector)
            else:
                 logits = self.blender(neural_state)
        
        if self.blend_function == "softmax":
            return torch.softmax(logits, dim=1)
        else:
            return F.gumbel_softmax(logits, dim=1, hard=True)

    def to_action_distribution_from_valuation(self, valuation, logic_idx=0):
        """
        Converts full valuation tensor to an action distribution for a specific logic actor.
        """
        batch_size = valuation.size(0)
        env_action_names = list(self.env.pred2action.keys())

        # Add a dummy zero column at the end for actions not present in the atoms
        valuation_with_dummy = torch.cat(
            [valuation, torch.zeros(batch_size, 1, device=valuation.device)], dim=1
        )
        # Store for explanations
        if not hasattr(self, "raw_action_probs_list"):
            self.raw_action_probs_list = [None] * len(self.logic_actors)
        self.raw_action_probs_list[logic_idx] = valuation_with_dummy
        
        valuation_logits = torch.logit(valuation_with_dummy, eps=0.01)
        dist_values = []
        mapping = self.logic_actor_mappings[logic_idx]
        
        for i in range(len(env_action_names)):
            indices = torch.tensor(mapping[i], device=valuation.device)
            indices = indices.expand(batch_size, -1)
            gathered = torch.gather(valuation_logits, 1, indices)
            merged = softor(gathered, dim=1)
            dist_values.append(merged)

        action_values = torch.stack(dist_values, dim=1)
        action_dist = torch.softmax(action_values, dim=1)
        return self.reshape_action_distribution(action_dist)

    def to_neural_action_distribution(self, neural_state):
        hidden = self.neural_actor.network(neural_state)
        logits = self.neural_actor.actor(hidden)
        probs = Categorical(logits=logits)
        return probs.probs

    def reshape_action_distribution(self, action_dist):
        batch_size = action_dist.size(0)
        if action_dist.size(1) < self.env.n_raw_actions:
            zeros = torch.zeros(
                batch_size,
                self.env.n_raw_actions - action_dist.size(1),
                device=action_dist.device,
                requires_grad=True,
            )
            action_dist = torch.cat([action_dist, zeros], dim=1)
        return action_dist

    def forward(self, neural_state, logic_state):
        if self.actor_mode == "hybrid":
            return self.compute_action_probs_hybrid(neural_state, logic_state)
        elif self.actor_mode == "logic":
            return self.compute_action_probs_logic(logic_state)
        else:
            return self.compute_action_probs_neural(neural_state)


class BlenderActorCritic(nn.Module):
    """
    BlendeRL actor-critic that combines neural and multiple logic policies.

    Args:
        env: environment
        rules: rules (can be a comma-separated string for multiple rulesets)
        actor_mode: actor mode, one of ["hybrid", "logic", "neural"]
        blender_mode: blender mode, one of ["logic", "neural"]
        blend_function: blending function, one of ["softmax", "gumbel_softmax"]
        device: device
        rng: random number generator
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
    ):
        super(BlenderActorCritic, self).__init__()
        self.device = device
        self.rng = random.Random() if rng is None else rng
        self.actor_mode = actor_mode
        self.blender_mode = blender_mode
        self.blend_function = blend_function
        self.env = env
        self.rules = rules
        self.explain = explain
        
        if isinstance(rules, str) and "," in rules:
            self.rulesets = [r.strip() for r in rules.split(",")]
        elif isinstance(rules, list):
            self.rulesets = rules
        else:
            self.rulesets = [rules]

        self.visual_neural_actor = get_neural_agent(self.env.name, self.env.n_actions, device, arch_name=architecture)
        
        self.logic_actors = nn.ModuleList()
        for r in self.rulesets:
            if reasoner == "neumann":
                from neumann.common import get_neumann_model
                la = get_neumann_model(env.name, r, device=device, train=True, explain=explain)
            else:
                la = get_nsfr_model(env.name, r, device=device, train=True, explain=explain)
            self.logic_actors.append(la)
        
        # Shortcut for single logic actor compatibility
        self.logic_actor = self.logic_actors[0]
        
        out_size = 1 + len(self.logic_actors)
        self.blender = get_blender(
            env,
            self.rulesets[0],
            device,
            blender_mode=blender_mode,
            train=True,
            explain=explain,
            out_size=out_size,
            architecture=architecture if architecture else "cnn"
        )
        
        # Load logic critic (MLP)
        mlp_module_path = f"in/envs/{env.name}/mlp.py"
        if os.path.exists(mlp_module_path):
            module = load_module(mlp_module_path)
            self.logic_critic = module.MLP(device=device, out_size=1, logic=True)
        else:
            self.logic_critic = None 

        self.actor = BlenderActor(
            env,
            self.visual_neural_actor,
            self.logic_actors,
            self.blender,
            actor_mode,
            blender_mode,
            blend_function,
            device=device,
        )

        # the number of actual actions on the environment
        self.num_actions = len(self.env.pred2action.keys())

        self.uniform = Categorical(
            torch.tensor(
                [1.0 / self.num_actions for _ in range(self.num_actions)], device=device
            )
        )
        self.upprior = Categorical(
            torch.tensor(
                [0.9]
                + [0.1 / (self.num_actions - 1) for _ in range(self.num_actions - 1)],
                device=device,
            )
        )

    def _print(self):
        """
        Print the weighted logic rules for actor and blender.
        """
        if self.blender_mode == "logic":
            print("==== Blender ====")
            print_program(self.blender)
        for i, la in enumerate(self.logic_actors):
            print(f"==== Logic Policy {i} ({self.rulesets[i]}) ====")
            print_program(la)

    def get_policy_weights(self):
        """
        Get the blending policy weights stored in the latest forward computation.

        Returns:
            weights: blending policy weights
        """
        return self.actor.w_policy

    def forward(self):
        raise NotImplementedError

    def get_explanation(self, neural_state, logic_state):
        """
        Get the explanation of the blending weights.

        Args:
            neural_state: neural state
            logic_state: logic state
        Returns:
            explanation: explanation
        """
        self.actor.get_explanation(neural_state, logic_state)

    def act(self, neural_state, logic_state, epsilon=0.0):
        """
        Compute an action using the actor. Used only by the play script (render.py).

        Args:
            neural_state: neural state
            logic_state: logic state
            epsilon: epsilon for e-greedy
        Returns:
            action: action
            action_logprob: action log probability
        """
        action_probs, blending_weights = self.actor(neural_state, logic_state)

        # e-greedy
        if self.rng.random() < epsilon:
            # random action with epsilon probability
            dist = self.uniform
            action = dist.sample()
        else:
            dist = Categorical(action_probs)
            # action = (action_probs[0] == max(action_probs[0])).nonzero(as_tuple=True)[0].squeeze(0).to(self.device)
            action = dist.sample()
            # print(action)
            if torch.numel(action) > 1:
                action = action[0]
        # action = dist.sample()
        action_logprob = dist.log_prob(action)
        action_prob = torch.exp(action_logprob)
        return action.detach(), action_prob  # action_logprob.detach()

    def get_prednames(self):
        """
        Get the predicate names representing actions.
        Returns:
            prednames: predicate names
        """
        # Return prednames of the first logic actor for compatibility
        return self.logic_actor.get_prednames()

    def get_action_and_value(self, neural_state, logic_state, action=None):
        """
        Compute an action and value.
        Args:
            neural_state: neural state
            logic_state: logic state
            action: action
        Returns:
            action: action
            logprob: log probability
            entropy: entropy
            value: value
        """
        # Compute action probabilities using blenderl actor
        # size: n_envs * n_actions
        action_probs, blending_weights = self.actor(neural_state, logic_state)
        dist = Categorical(action_probs)
        blend_dist = Categorical(blending_weights)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)

        # Compute state values using each neural and logic value function
        # size: n_envs * 1
        neural_value = self.get_neural_value(neural_state).squeeze(1)
        logic_value = self.get_logic_value(logic_state).squeeze(1)
        
        # blend the values using blending weights
        # neural_weight * neural_value + sum(logic_weights) * logic_value
        # Since logic modules share the same logic_critic for now.
        logic_weight_sum = blending_weights[:, 1:].sum(dim=1)
        blended_value = (
            blending_weights[:, 0] * neural_value
            + logic_weight_sum * logic_value
        ).unsqueeze(1)

        return action, logprob, dist.entropy(), blend_dist.entropy(), blended_value

    def get_neural_value(self, neural_state):
        """
        Compute the value using the neural value function from a RGB state.
        Args:
            neural_state: neural state
        Returns:
            value: value
        """
        value = self.visual_neural_actor.get_value(neural_state)
        return value

    def get_logic_value(self, logic_state):
        """
        Compute the value using the logic value function from a OCAtari state.
        Args:
            logic_state: logic state
        Returns:
            value: value
        """
        value = self.logic_critic(logic_state)
        return value

    def get_value(self, neural_state, logic_state):
        """
        Compute the value using the blending value function.
        Args:
            neural_state: neural state
            logic_state: logic state
        Returns:
            value: value
        """
        _, _, _, _, value = self.get_action_and_value(neural_state, logic_state)
        return value

    def save(
        self, checkpoint_path, directory: Path, step_list, reward_list, weight_list
    ):
        """
        Save the model.

        Args:
            checkpoint_path: checkpoint path
            directory: directory
            step_list: step list
            reward_list: reward list
            weight_list: weight list
        """
        torch.save(self.state_dict(), checkpoint_path)
        with open(directory / "data.pkl", "wb") as f:
            pickle.dump(step_list, f)
            pickle.dump(reward_list, f)
            pickle.dump(weight_list, f)
