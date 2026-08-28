#!/usr/bin/env python3
from typing import Optional
from pathlib import Path

from plot.base import BasePlotter
from plot.ep_cache import get_ep_eval_data
from src.early_prediction.eval_logic import write_counterfactual_table

class EpCounterfactualTablePlotter(BasePlotter):
    def __init__(self):
        super().__init__("ep_counterfactual_table")

    def run(self, exp_id: str, cli_overrides: Optional[dict] = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        if not cfg.get("enabled", True):
            return

        data = get_ep_eval_data(exp_id, cfg, group, output_dir)
        if not data:
            return
            
        cf_data = data.get("cf_data")
        
        if cf_data:
            print(f"Generating Counterfactual Tables in {output_dir}")
            csv_path = output_dir / "counterfactual_summary.csv"
            txt_path = output_dir / "counterfactual_summary.txt"
            write_counterfactual_table(cf_data, csv_path, txt_path)
