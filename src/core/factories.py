import os
import torch
import numpy as np
import gymnasium as gym
from nsfr.utils.common import load_module
from collections import OrderedDict
from src.models.architectures import CNNActor, NeuralBlenderActor, NeuralBlenderMLP

def get_neural_agent(env_name, n_actions, device, arch_name=None, hidden_sizes=[64, 64], num_in_features=None, **kwargs):
    if arch_name in ["cross_attention", "cross_attention_transformer", "sepsis_cross_attention"]:
        transformer_module_path = f"in/envs/{env_name}/transformer.py"
        if not os.path.exists(transformer_module_path):
            raise FileNotFoundError(f"Requested transformer architecture but {transformer_module_path} does not exist.")
        module = load_module(transformer_module_path)
        if hasattr(module, "CrossAttentionPolicy"):
            return module.CrossAttentionPolicy(device=device, out_size=n_actions, num_in_features=num_in_features).to(device)
        return module.CrossAttentionSepsisPolicy(device=device, out_size=n_actions, num_in_features=num_in_features).to(device)

    if arch_name in ["transformer", "sepsis_transformer"]:
        transformer_module_path = f"in/envs/{env_name}/transformer.py"
        if not os.path.exists(transformer_module_path):
            raise FileNotFoundError(f"Requested transformer architecture but {transformer_module_path} does not exist.")
        module = load_module(transformer_module_path)
        if hasattr(module, "TransformerPolicy"):
            return module.TransformerPolicy(device=device, out_size=n_actions, num_in_features=num_in_features).to(device)
        return module.SepsisTransformerPolicy(device=device, out_size=n_actions, num_in_features=num_in_features).to(device)

    if arch_name in ["dueling_resnet", "resnet"]:
        mlp_module_path = f"in/envs/{env_name}/mlp.py"
        if not os.path.exists(mlp_module_path):
            raise FileNotFoundError(f"Requested resnet architecture but {mlp_module_path} does not exist.")
        module = load_module(mlp_module_path)
        if hasattr(module, "DuelingResNetMLP"):
            return module.DuelingResNetMLP(device=device, out_size=n_actions, hidden_sizes=hidden_sizes, num_in_features=num_in_features).to(device)
        return module.MLP(device=device, out_size=n_actions, hidden_sizes=hidden_sizes, num_in_features=num_in_features).to(device)

    def _safe_instantiate(module_class, **kwargs):
        import inspect
        sig = inspect.signature(module_class.__init__)
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return module_class(**valid_kwargs)

    if arch_name in ["mlp", "dnn", "standard_mlp"]:
        mlp_module_path = f"in/envs/{env_name}/mlp.py"
        if not os.path.exists(mlp_module_path):
            raise FileNotFoundError(f"Requested MLP architecture but {mlp_module_path} does not exist.")
        module = load_module(mlp_module_path)
        if hasattr(module, "StandardMLP"):
            return _safe_instantiate(module.StandardMLP, device=device, out_size=n_actions, hidden_sizes=hidden_sizes, num_in_features=num_in_features).to(device)
        return _safe_instantiate(module.MLP, device=device, out_size=n_actions, hidden_sizes=hidden_sizes, num_in_features=num_in_features).to(device)
    
    if arch_name == "cnn":
        return CNNActor(n_actions=n_actions).to(device)

    raise ValueError(f"Unknown architecture '{arch_name}' requested for environment '{env_name}'.")

def get_blender(env, blender_rules, device, train=True, blender_mode="logic", reasoner="nsfr", explain=False, out_size=2, architecture="cnn"):
    assert blender_mode in ["logic", "neural"]
    if blender_mode == "logic":
        if reasoner == "nsfr":
            from nsfr.common import get_blender_nsfr_model
            return get_blender_nsfr_model(env.name, blender_rules, device, train=train, explain=explain)
        elif reasoner == "neumann":
            from neumann.common import get_neumann_model, get_blender_neumann_model
            return get_blender_neumann_model(env.name, blender_rules, device, train=train, explain=explain)
    if blender_mode == "neural":
        if architecture == "cnn":
            net = NeuralBlenderActor(out_size=out_size)
        else:
            dummy_logic, dummy_neural = env.reset()
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

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, i, capture_video, run_name) for i in range(num_envs)],
    )
    return envs

def load_cleanrl_agent(pretrained, device):
    agent = CNNActor(n_actions=18)
    if pretrained:
        try:
            agent.load_state_dict(torch.load("cleanrl/out/ppo_Seaquest-v4_1.pth"))
            agent.to(device)
        except RuntimeError:
            agent.load_state_dict(torch.load("cleanrl/out/ppo_Seaquest-v4_1.pth", map_location=torch.device("cpu")))
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
