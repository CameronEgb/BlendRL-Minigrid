MiniGrid Dynamic Obstacles BaselinesThis project contains scripts to generate baselines for the MiniGrid-Dynamic-Obstacles-8x8-v0 environment. It supports Deep RL methods (PPO, A2C, DQN) and heuristic baselines (Random, A* Oracle).Comparison to NeSy RLWhen comparing your Neuro-Symbolic (NeSy) method, use these specific baselines to demonstrate different strengths:A* (Oracle): The Upper Bound. It cheats by looking at the map grid to plan paths. Your method should aim to approach this performance but using only pixel/symbolic observations, not internal grid access.Random: The Lower Bound.PPO/A2C/DQN: The Competitive Baselines. These show what standard "black box" neural networks can achieve.Installationpip install -r requirements.txt
Usage1. Run a BaselineDeep RL Agents (Training):# Best general performer
python train_baseline.py --algo ppo --obstacles 6 --steps 500000

# Lighter alternative to PPO
python train_baseline.py --algo a2c --obstacles 6 --steps 500000
Heuristic Agents (Evaluation):These do not "train" but run for the specified steps to generate logs for comparison plots.# Upper Bound (Oracle)
python train_baseline.py --algo astar --obstacles 6 --steps 100000

# Lower Bound (Random)
python train_baseline.py --algo random --obstacles 6 --steps 100000
2. Visualize ResultsUse the provided plotter to see how they stack up.# Plot everything together
python plot_results.py --dirs ./logs/ppo_obst6_* ./logs/astar_obst6_* ./logs/random_obst6_*
Algorithm Performance EstimatesEnvironment: MiniGrid-Dynamic-Obstacles-8x8-v0AlgorithmTypeEst. Success RateEst. Wall TimeNotesA*Oracle~90-99%FastReplans every step. Might fail if surrounded.PPODeep RL~70-90%15 minsRobust, standard baseline.A2CDeep RL~60-80%10 minsFaster training steps than PPO.DQNDeep RL~40-70%45 minsHard to tune for dynamic environments.RandomBaseline<5%InstantUseful for debugging plotting logic.Note: The "Success Rate" heavily depends on the number of obstacles. At 4 obstacles, PPO might hit 95%. At 10 obstacles, it might drop to 50%.