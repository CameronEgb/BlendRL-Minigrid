#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List

from plot.base import BasePlotter, clean_label, get_style_info, moving_average

class ConvergencePlotter(BasePlotter):
    def __init__(self):
        super().__init__("convergence")

    def run(self, exp_id: str, cli_overrides: Optional[dict] = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        metrics = cfg.get("metrics", ["eval/reward", "train/reward", "train/length"])
        self.plot_metric_series(exp_id, group, output_dir, metrics, cfg)

