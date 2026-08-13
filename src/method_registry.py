"""Unified method style registry for the NeSyRL project.

Single source of truth for method display names, colors, line styles, and markers.
Both the plotting system (plot/base.py) and EP evaluation (eval.py) import from here.

To add a new architecture, add ONE entry to METHOD_STYLE below.
"""
import re
from typing import Tuple, Optional

METHOD_STYLE = {
    # key (checkpoint/log dir name)    label                       color          marker   linestyle
    "cql":                          {"label": "CQL (Neural Only)",    "color": "tab:blue",   "marker": "o", "linestyle": "-"},
    "blendrl_cql_human_neural":     {"label": "BlendRL Human+Neural", "color": "tab:orange", "marker": "s", "linestyle": "-"},
    "blendrl_cql_human_cew":        {"label": "BlendRL Human+CEW",   "color": "tab:green",  "marker": "^", "linestyle": "--"},
    "blendrl_cql_cew_only":         {"label": "BlendRL CEW Only",     "color": "tab:red",    "marker": "D", "linestyle": "-"},
    "clinician":                    {"label": "Clinician (Dataset)",  "color": "tab:purple", "marker": "X", "linestyle": "-"},
    # Standard RL methods
    "ppo":                          {"label": "PPO",                  "color": "black",      "marker": "o", "linestyle": "--"},
    "iql":                          {"label": "IQL",                  "color": "#1f77b4",    "marker": "d", "linestyle": "-"},
    "blendrl":                      {"label": "BlendRL",              "color": "#2ca02c",    "marker": "^", "linestyle": "-"},
    "blendrl_iql":                  {"label": "BlendRL-IQL",          "color": "#d62728",    "marker": "s", "linestyle": "-"},
    "cew":                          {"label": "CEW",                  "color": "#ff7f0e",    "marker": "h", "linestyle": "-"},
    "cew_fyd":                      {"label": "CEW+FYD",              "color": "#9467bd",    "marker": "p", "linestyle": "-"},
}

_DEFAULT_STYLE = {"label": None, "color": None, "marker": "o", "linestyle": "-"}


def get_style(name: str) -> dict:
    """Look up style by exact match, then by longest prefix match.
    
    Examples:
        get_style("cql")                  -> exact match
        get_style("ppo_cp_tuned")          -> prefix match on "ppo"
        get_style("blendrl_iql_cp_tuned")  -> prefix match on "blendrl_iql"
        get_style("unknown_method")        -> default with label=name
    """
    if name in METHOD_STYLE:
        return METHOD_STYLE[name]
    # Prefix match: longest key that is a prefix of name wins
    for key in sorted(METHOD_STYLE.keys(), key=len, reverse=True):
        if name.startswith(key + "_") or name == key:
            return METHOD_STYLE[key]
    return {**_DEFAULT_STYLE, "label": name}


def clean_label(name: str) -> str:
    """Return human-readable display label for a method name."""
    return get_style(name)["label"]


def get_style_info(name: str) -> Tuple[Optional[str], str, str]:
    """Return (color, linestyle, marker) tuple for matplotlib plotting."""
    s = get_style(name)
    return s["color"], s["linestyle"], s["marker"]
