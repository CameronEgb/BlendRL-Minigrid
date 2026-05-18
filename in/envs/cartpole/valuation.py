import torch as th
from nsfr.utils.common import bool_to_probs

def leaning_left(agent: th.Tensor) -> th.Tensor:
    # agent: [cart_pos, cart_vel, pole_angle, pole_vel]
    angle = agent[..., 2]
    return bool_to_probs(angle < -0.01)

def leaning_right(agent: th.Tensor) -> th.Tensor:
    angle = agent[..., 2]
    return bool_to_probs(angle > 0.01)

def moving_left(agent: th.Tensor) -> th.Tensor:
    vel = agent[..., 3]
    return bool_to_probs(vel < -0.01)

def moving_right(agent: th.Tensor) -> th.Tensor:
    vel = agent[..., 3]
    return bool_to_probs(vel > 0.01)

def is_stable(env_obj: th.Tensor) -> th.Tensor:
    # Stable if both angle and velocity are near zero
    # env_obj is usually the same state vector in CartPole
    angle = env_obj[..., 2]
    vel = env_obj[..., 3]
    result = (th.abs(angle) < 0.05) & (th.abs(vel) < 0.05)
    return bool_to_probs(result)

def not_stable(env_obj: th.Tensor) -> th.Tensor:
    angle = env_obj[..., 2]
    vel = env_obj[..., 3]
    result = (th.abs(angle) >= 0.05) | (th.abs(vel) >= 0.05)
    return bool_to_probs(result)

def true(env_obj: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.ones_like(env_obj[..., 0], dtype=th.bool))
