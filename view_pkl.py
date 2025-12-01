import pickle

with open("out/runs/minigrid_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_4_steps_64_num_balls_2__0/checkpoints/training_log.pkl", "rb") as f:
    obj = pickle.load(f)

print(type(obj))
print(obj)

for i, item in enumerate(obj):
    print(f"Item {i}: type={type(item)}")

