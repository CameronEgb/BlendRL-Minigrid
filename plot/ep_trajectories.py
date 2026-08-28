#!/usr/bin/env python3
from typing import Optional
from pathlib import Path

from plot.base import BasePlotter
from plot.ep_cache import get_ep_eval_data
from src.early_prediction.eval_logic import plot_ep_shock_over_tau

class EpTrajectoriesPlotter(BasePlotter):
    def __init__(self):
        super().__init__("ep_trajectories")

    def run(self, exp_id: str, cli_overrides: Optional[dict] = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        if not cfg.get("enabled", True):
            return

        data = get_ep_eval_data(exp_id, cfg, group, output_dir)
        if not data:
            return
            
        ep_shock_results = data.get("ep_shock_results")
        
        if ep_shock_results:
            print(f"Generating EP Trajectories Plots in {output_dir}")
            plot_ep_shock_over_tau(ep_shock_results, output_dir)
