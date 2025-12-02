import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# Load CSVs (expects logic.csv and neural.csv with: step,value)
# ------------------------------------------------------------
logic = pd.read_csv("logic.csv")
neural = pd.read_csv("neural.csv")

steps = logic.iloc[:, 0]
logic_vals = logic.iloc[:, 1]
neural_vals = neural.iloc[:, 1]

# ------------------------------------------------------------
# Raw choice: neural = 1, logic = 0
# ------------------------------------------------------------
raw_choice = (neural_vals > logic_vals).astype(int)

# ------------------------------------------------------------
# TensorBoard-style smoothing (EMA)
# ------------------------------------------------------------
def ema(series, smoothing=0.9):
    out = []
    last = float(series.iloc[0])
    for x in series:
        last = last * smoothing + (1 - smoothing) * float(x)
        out.append(last)
    return np.array(out)

smooth_choice = ema(raw_choice, smoothing=0.9)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(18,6))

# --- vertical colored bars (option 1 style) ---
for s, c in zip(steps, raw_choice):
    plt.axvline(
        s,
        color="#1f77b4" if c == 0 else "#ff7f0e",   # logic(0)=blue, neural(1)=orange
        alpha=0.30,
        linewidth=2
    )

# --- plot smoothed line ---
plt.plot(steps, smooth_choice, color="#1f77b4", linewidth=3)

# ------------------------------------------------------------
# Labels, title, grid
# ------------------------------------------------------------
plt.title("Model Choice Over Steps", fontsize=24, pad=20)
plt.xlabel("Step", fontsize=18)
plt.ylabel("Neural (1) vs Logic (0)", fontsize=18)
plt.grid(True, linestyle="--", alpha=0.4)

# No legend — per request

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
plt.savefig("model_choice_no_legend.png", dpi=300, bbox_inches="tight")
print("Saved: model_choice_no_legend.png")
