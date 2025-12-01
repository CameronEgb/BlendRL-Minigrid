import tyro
import torch as th
import numpy as np
from nudge.utils import load_model
from nudge.env import NudgeBaseEnv
from nsfr.fol.logic import Clause, Atom
from nsfr.fol.data_utils import DataUtils
from nsfr.utils.logic import get_index_by_predname # Added for utility
# from nsfr.fol.language import Language # No longer needed directly

def debug_logic(
        env_name: str = "minigrid",
        agent_path: str = "out/runs/minigrid_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_4_steps_64__0",
        episodes: int = 1,
        device: str = "cpu",
        seed: int = 0,
):
    # Load agent & environment
    print(f"Loading model from {agent_path} for env {env_name}")
    model = load_model(agent_path, env_kwargs_override={}, device=device)
    env = NudgeBaseEnv.from_name(env_name, mode="eval", seed=seed)
    print("Loaded model & environment")

    nsfr = model.actor.logic_actor
    print("Logic Actor Predicates:", nsfr.prednames)

    # Load clauses to identify body predicates
    lark_path = "nsfr/nsfr/fol/exp.lark" 
    lang_base_path = "in/envs/" 
    dataset_name_for_du = f"{env_name}/logic/default" # Corrected dataset path to include logic/default
    
    # Initialize DataUtils and load language
    du = DataUtils(lark_path=lark_path, lang_base_path=lang_base_path, dataset=dataset_name_for_du)
    lang = du.load_language() # Load language via DataUtils
    
    clauses_file_path = f"in/envs/{env_name}/logic/default/clauses.txt" 
    with open(clauses_file_path, "r") as f:
        clause_expressions = [line.strip() for line in f.readlines()] # Strip newlines
    
    # Use DataUtils to parse clauses with the loaded language
    parsed_clauses = [du.parse_clause(expr, lang) for expr in clause_expressions]
    
    body_preds_to_monitor = set()
    for clause_obj in parsed_clauses: 
        if isinstance(clause_obj, Clause):
            for body_literal in clause_obj.body:
                if isinstance(body_literal, Atom):
                    body_preds_to_monitor.add(body_literal.pred.name)

    print("Monitoring body predicates:", body_preds_to_monitor)

    for ep in range(episodes):
        print(f"=== Episode {ep+1} ===")
        obs_logic, obs_nn = env.reset()
        obs_logic = th.tensor(obs_logic, dtype=th.float32, device=model.device)
        obs_nn = th.tensor(obs_nn, dtype=th.float32, device=model.device)

        if obs_nn.dim() == 3:
            obs_nn = obs_nn.unsqueeze(0)

        done = False
        step = 0
        
        # Run for a few steps to observe values
        max_steps = 20 
        
        while not done and step < max_steps:
            step += 1
            # Agent chooses action using Blender policy
            # This triggers the forward pass in logic actor which populates V_0 and V_T
            action, logprob = model.act(obs_nn, obs_logic)

            # Access the logic actor's valuations
            print(f"\n--- Step {step} ---")
            
            # Print Logic Policy Predicate Values (V_T)
            print("Logic Policy Output (V_T):")
            pred_vals = {
                pred: nsfr.get_predicate_valuation(pred, initial_valuation=False)
                for pred in nsfr.prednames
            }
            for pred, val in pred_vals.items():
                print(f"  {pred}: {val:.4f}")

            # Also inspect V_0 (Inputs) to see if they are changing
            print("Logic Input (V_0) - Values for body predicates:")
            # Iterate through all ground atoms and check if their predicate name is in our monitored list
            for atom_obj in nsfr.atoms:
                if atom_obj.pred.name in body_preds_to_monitor:
                    val = 0.0
                    try:
                        # Directly get valuation from V_0 using the atom's index
                        target_index = nsfr.atoms.index(atom_obj)
                        val = nsfr.V_0[:, target_index].item()
                    except ValueError:
                        pass # Should not happen if atom_obj is in nsfr.atoms
                    print(f"  {atom_obj}: {val:.4f}")
            
            print("Logic Input (V_0) - Top active atoms:")
            v0 = nsfr.V_0[0].detach().cpu().numpy() # Batch 0
            top_idxs = np.argsort(-v0)[:10] # Top 10
            for idx in top_idxs:
                val = v0[idx]
                if val > 0.01:
                    atom = nsfr.atoms[idx]
                    print(f"  {atom}: {val:.4f}")

                        
            # Step environment
            (new_logic, new_nn), rewards, truncs, dones, infos = env.step(int(action))

            # update for next step
            obs_logic = th.tensor(new_logic, dtype=th.float32, device=model.device)
            obs_nn = th.tensor(new_nn, dtype=th.float32, device=model.device)
            if obs_nn.dim() == 3:
                obs_nn = obs_nn.unsqueeze(0)

            done = dones[0]

        if done:
            print("Episode finished.")
        else:
            print(f"Stopped after {max_steps} steps.")

def main():
    tyro.cli(debug_logic)

if __name__ == "__main__":
    main()
