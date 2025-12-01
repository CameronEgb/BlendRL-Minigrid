import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import glob
import numpy as np

def moving_average(values, window):
    """Smooths the data using a moving average."""
    return values.rolling(window=window, min_periods=1).mean()

def plot_logs(log_dirs, window=100):
    plt.figure(figsize=(14, 6))

    # --- Plot 1: Rewards ---
    plt.subplot(1, 2, 1)
    
    for log_dir in log_dirs:
        # Find the monitor.csv file in the log directory
        monitor_files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
        if not monitor_files:
            print(f"No monitor file found in {log_dir}")
            continue
            
        # Standard SB3 monitor files have 1 header line metadata, then headers
        try:
            df = pd.read_csv(monitor_files[0], skiprows=1)
        except:
            print(f"Could not read {monitor_files[0]}")
            continue

        # Check if data exists
        if df.empty:
            continue
            
        # Calculate cumulative steps
        if 'l' in df.columns: # 'l' is episode length
            df['timesteps'] = df['l'].cumsum()
        else:
            df['timesteps'] = df.index # Fallback
            
        # Smooth rewards ('r')
        smoothed_rewards = moving_average(df['r'], window)
        
        # Label extraction
        label = os.path.basename(os.path.normpath(log_dir))
        
        plt.plot(df['timesteps'], smoothed_rewards, label=label)

    plt.title(f"Training Reward (Smoothed window={window})")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Plot 2: Episode Length (Proxy for efficiency) ---
    plt.subplot(1, 2, 2)
    for log_dir in log_dirs:
        monitor_files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
        if not monitor_files: continue
        df = pd.read_csv(monitor_files[0], skiprows=1)
        if df.empty: continue
        if 'l' in df.columns:
            df['timesteps'] = df['l'].cumsum()
            smoothed_len = moving_average(df['l'], window)
            label = os.path.basename(os.path.normpath(log_dir))
            plt.plot(df['timesteps'], smoothed_len, label=label)

    plt.title("Episode Length (Steps to Goal/Fail)")
    plt.xlabel("Timesteps")
    plt.ylabel("Steps")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training logs")
    parser.add_argument("--dirs", nargs='+', required=True, help="List of log directories to plot")
    parser.add_argument("--window", type=int, default=50, help="Smoothing window size")
    
    args = parser.parse_args()
    plot_logs(args.dirs, args.window)