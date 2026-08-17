"""Unified method style registry for the NeSyRL project.

Single source of truth for method display names, colors, line styles, and markers.
Both the plotting system (plot/base.py) and EP evaluation (eval.py) import from here.

To add a new architecture, add ONE entry to METHOD_STYLE below.
"""
import re
from typing import Tuple, Optional

METHOD_STYLE = {
    # Clean harness/architecture keys
    "cql":                          {"label": "CQL (Neural)",         "color": "tab:blue",   "marker": "o", "linestyle": "-"},
    "cql_dnn":                      {"label": "CQL (Standard MLP)",   "color": "tab:blue",   "marker": "o", "linestyle": "-"},
    "cql_dueling_resnet":           {"label": "CQL (Dueling ResNet)",  "color": "#1f77b4",    "marker": "D", "linestyle": "-"},
    "cql_transformer":              {"label": "CQL (Transformer)",    "color": "#e377c2",    "marker": "p", "linestyle": "-"},
    "cql_blendrl_human_neural":     {"label": "BlendRL Human+Neural", "color": "tab:orange", "marker": "s", "linestyle": "-"},
    "blendrl_cql_human_neural":     {"label": "BlendRL Human+Neural", "color": "tab:orange", "marker": "s", "linestyle": "-"},
    "cql_blendrl_human_dueling_resnet": {"label": "BlendRL Human+ResNet", "color": "tab:orange", "marker": "s", "linestyle": "-"},
    "blendrl_cql_human_dueling_resnet": {"label": "BlendRL Human+ResNet", "color": "tab:orange", "marker": "s", "linestyle": "-"},
    "cql_blendrl_human_dueling_resnet_rigid": {"label": "BlendRL ResNet+Rigid", "color": "tab:orange", "marker": "s", "linestyle": "-"},
    "blendrl_cql_human_dueling_resnet_rigid": {"label": "BlendRL ResNet+Rigid", "color": "tab:orange", "marker": "s", "linestyle": "-"},
    "cql_blendrl_human_dueling_resnet_continuous": {"label": "BlendRL ResNet+Soft", "color": "#17becf", "marker": "v", "linestyle": "-"},
    "blendrl_cql_human_dueling_resnet_continuous": {"label": "BlendRL ResNet+Soft", "color": "#17becf", "marker": "v", "linestyle": "-"},
    "cql_blendrl_human_rigid":      {"label": "BlendRL Rigid Logic",  "color": "tab:orange", "marker": "s", "linestyle": "-"},
    "cql_blendrl_human_continuous": {"label": "BlendRL Soft Logic",   "color": "#17becf",    "marker": "v", "linestyle": "-"},
    "cql_blendrl_human_transformer":{"label": "BlendRL Human+Transformer", "color": "#8c564b", "marker": "h", "linestyle": "-"},
    "blendrl_cql_human_transformer":{"label": "BlendRL Human+Transformer", "color": "#8c564b", "marker": "h", "linestyle": "-"},
    "cql_blendrl_cross_attention":  {"label": "BlendRL Cross-Attention", "color": "#9467bd",  "marker": "*", "linestyle": "-"},
    "blendrl_cql_cross_attention":  {"label": "BlendRL Cross-Attention", "color": "#9467bd",  "marker": "*", "linestyle": "-"},
    "cql_blendrl_human_cew":        {"label": "BlendRL Human+CEW",   "color": "tab:green",  "marker": "^", "linestyle": "--"},
    "blendrl_cql_human_cew":        {"label": "BlendRL Human+CEW",   "color": "tab:green",  "marker": "^", "linestyle": "--"},
    "cql_blendrl_cew_dueling_resnet": {"label": "BlendRL CEW+ResNet", "color": "#2ca02c",   "marker": "h", "linestyle": "-"},
    "blendrl_cql_cew_dueling_resnet": {"label": "BlendRL CEW+ResNet", "color": "#2ca02c",   "marker": "h", "linestyle": "-"},
    "cql_blendrl_human_cew_dueling_resnet": {"label": "BlendRL Human+CEW+ResNet", "color": "#8c564b", "marker": "p", "linestyle": "-"},
    "blendrl_cql_human_cew_dueling_resnet": {"label": "BlendRL Human+CEW+ResNet", "color": "#8c564b", "marker": "p", "linestyle": "-"},
    "cql_blendrl_cew_only":         {"label": "BlendRL CEW Only",     "color": "tab:red",    "marker": "D", "linestyle": "-"},
    "blendrl_cql_cew_only":         {"label": "BlendRL CEW Only",     "color": "tab:red",    "marker": "D", "linestyle": "-"},
    "iql_dnn":                      {"label": "IQL (Neural)",         "color": "#1f77b4",    "marker": "d", "linestyle": "-"},
    "iql_blendrl_human_neural":     {"label": "BlendRL Human+Neural", "color": "#d62728",    "marker": "s", "linestyle": "-"},
    "ppo_dnn":                      {"label": "PPO (Neural)",         "color": "black",      "marker": "o", "linestyle": "--"},
    "ppo_blendrl_human_neural":     {"label": "BlendRL Human+Neural", "color": "#2ca02c",    "marker": "^", "linestyle": "-"},
    "cew_base":                     {"label": "CEW",                  "color": "#ff7f0e",    "marker": "h", "linestyle": "-"},
    "cew_fyd":                      {"label": "CEW+FYD",              "color": "#bcbd22",    "marker": "p", "linestyle": "-"},
    "clinician":                    {"label": "Clinician (Dataset)",  "color": "tab:purple", "marker": "X", "linestyle": "-"},
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


def get_canonical_method_name(name: str) -> str:
    """Map method aliases to canonical registered name."""
    name = str(name).replace("/", "_")
    alias_map = {
        "blendrl_cql_human_neural": "cql_blendrl_human_neural",
        "blendrl_cql_human_transformer": "cql_blendrl_human_transformer",
        "blendrl_cql_human_cew": "cql_blendrl_human_cew",
        "blendrl_cql_cew_only": "cql_blendrl_cew_only",
        "blendrl_iql_human_neural": "iql_blendrl_human_neural",
        "blendrl_ppo_human_neural": "ppo_blendrl_human_neural",
        "cql": "cql_dnn",
        "cql_transformer": "cql_transformer",
        "iql": "iql_dnn",
        "ppo": "ppo_dnn",
    }
    return alias_map.get(name, name)


def get_method_aliases(name: str) -> set:
    """Return all known aliases for a given method name."""
    canon = get_canonical_method_name(name)
    raw = str(name).replace("/", "_")
    aliases = {name, canon, raw}
    if "cql_blendrl_" in canon:
        aliases.add(canon.replace("cql_blendrl_", "blendrl_cql_"))
    elif "blendrl_cql_" in canon:
        aliases.add(canon.replace("blendrl_cql_", "cql_blendrl_"))
    if canon == "cql_dnn":
        aliases.add("cql")
    elif canon == "cql":
        aliases.add("cql_dnn")
    if canon == "iql_dnn":
        aliases.add("iql")
    elif canon == "ppo_dnn":
        aliases.add("ppo")
    return aliases
