#!/usr/bin/env python3
from typing import Optional
from pathlib import Path

from plot.base import BasePlotter
from plot.ep_cache import get_ep_eval_data
from src.early_prediction.eval_logic import plot_agreement_vs_shock, plot_agreement_vs_shock_deciles

class EpAgreementPlotter(BasePlotter):
    def __init__(self):
        super().__init__("ep_agreement")

    def run(self, exp_id: str, cli_overrides: Optional[dict] = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        if not cfg.get("enabled", True):
            return

        data = get_ep_eval_data(exp_id, cfg, group, output_dir)
        if not data:
            return
            
        rl_agreements = data.get("rl_agreements")
        y = data.get("y")
        
        if rl_agreements:
            print(f"Generating Agreement Plots in {output_dir}")
            plot_agreement_vs_shock(rl_agreements, y, output_dir)
            plot_agreement_vs_shock_deciles(rl_agreements, y, output_dir)
