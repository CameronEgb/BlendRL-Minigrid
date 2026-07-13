import torch as th
from nsfr.utils.common import bool_to_probs

def low_bp(agent: th.Tensor) -> th.Tensor:
    # MAP is index 15. Z-score < -1.0 indicates hypotension
    map_val = agent[..., 15]
    return bool_to_probs(map_val < -1.0)

def high_lactate(agent: th.Tensor) -> th.Tensor:
    # Lactate is index 10. Z-score > 1.0 indicates hyperlactatemia
    lactate = agent[..., 10]
    return bool_to_probs(lactate > 1.0)

def high_creatinine(agent: th.Tensor) -> th.Tensor:
    # Creatinine is index 12. Z-score > 1.0 indicates renal dysfunction
    creatinine = agent[..., 12]
    return bool_to_probs(creatinine > 1.0)

def high_bilirubin(agent: th.Tensor) -> th.Tensor:
    # Bilirubin is index 13. Z-score > 1.0 indicates hepatic dysfunction
    bilirubin = agent[..., 13]
    return bool_to_probs(bilirubin > 1.0)

def low_platelets(agent: th.Tensor) -> th.Tensor:
    # Platelets is index 11. Z-score < -1.0 indicates thrombocytopenia/coagulation issues
    platelets = agent[..., 11]
    return bool_to_probs(platelets < -1.0)

def normal_bp(agent: th.Tensor) -> th.Tensor:
    # Stable BP corresponds to z-score >= -1.0
    map_val = agent[..., 15]
    return bool_to_probs(map_val >= -1.0)

def normal_lactate(agent: th.Tensor) -> th.Tensor:
    # Stable Lactate corresponds to z-score <= 1.0
    lactate = agent[..., 10]
    return bool_to_probs(lactate <= 1.0)

def normal_organs(agent: th.Tensor) -> th.Tensor:
    # All organ markers within normal ranges (z-score <= 1.0 and platelets >= -1.0)
    creatinine = agent[..., 12]
    bilirubin = agent[..., 13]
    platelets = agent[..., 11]
    is_normal = (creatinine <= 1.0) & (bilirubin <= 1.0) & (platelets >= -1.0)
    return bool_to_probs(is_normal)

def true(env_obj: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.ones_like(env_obj[..., 0], dtype=th.bool))
