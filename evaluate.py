import tyro
import torch as th
import numpy as np
from nudge.utils import load_model
from nudge.env import NudgeBaseEnv

def evaluate(
        env_name: str,
        agent_path: str,
        episodes: int = 5,
        device: str = "cpu",
        seed: int = 0,
):
    # Load agent & environment
    model = load_model(agent_path, env_kwargs_override={}, device=device)
    env = NudgeBaseEnv.from_name(env_name, mode="eval", seed=seed)
    print("Loaded model & environment:", env_name)

    returns = []

    for ep in range(episodes):
        obs_logic, obs_nn = env.reset()
        obs_logic = th.tensor(obs_logic, dtype=th.float32, device=model.device)
        obs_nn = th.tensor(obs_nn, dtype=th.float32, device=model.device)

        # ensure shape: (1, 4, 84, 84)
        if obs_nn.dim() == 3:
            obs_nn = obs_nn.unsqueeze(0)

        done = False
        ep_return = 0
        ep_len = 0

        while not done:
            # Agent chooses action using Blender policy
            action, logprob = model.act(obs_nn, obs_logic)

            # Step environment
            (new_logic, new_nn), rewards, truncs, dones, infos = env.step(int(action))

            shaped_reward = rewards[0]
            ep_return += shaped_reward
            ep_len += 1

            # update for next step
            obs_logic = th.tensor(new_logic, dtype=th.float32, device=model.device)
            obs_nn = th.tensor(new_nn, dtype=th.float32, device=model.device)
            if obs_nn.dim() == 3:
                obs_nn = obs_nn.unsqueeze(0)



            done = dones[0]

        print(f"Episode {ep+1}: return={ep_return:.3f}, length={ep_len}")
        returns.append(ep_return)

    print("\n=== Evaluation Summary ===")
    print(f"Mean return: {np.mean(returns):.3f}")
    print(f"Std return : {np.std(returns):.3f}")
    print(f"Min return : {np.min(returns):.3f}")
    print(f"Max return : {np.max(returns):.3f}")


def main():
    tyro.cli(evaluate)

if __name__ == "__main__":
    main()
