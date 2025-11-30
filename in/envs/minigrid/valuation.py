import torch as th

from nsfr.utils.common import bool_to_probs


def nothing_around(objs: th.Tensor) -> th.Tensor:
    """
    Checks if there are no "target" objects (obstacles) near the player.
    This is the inverse of being close to any of those objects.
    """
    # In Minigrid's logic state:
    # objs[:, 1] is the agent
    # objs[:, 3:] are the obstacles
    agent = objs[:, 1, :]
    obstacles = objs[:, 3:, :]
    
    # Calculate close_by probability for each obstacle
    close_probs = []
    for i in range(obstacles.size(1)):
        close_probs.append(_close_by(agent, obstacles[:, i, :]))

    # If we're close to *any* obstacle, then something is "around"
    something_is_around_prob = th.stack(close_probs).max(dim=0)[0]
    
    # "nothing_around" is the inverse of that probability
    result = 1.0 - something_is_around_prob
    return result.float()


def close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """
    Calculates the probability of the player being close to a given object.
    """
    return _close_by(player, obj)


def _close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """
    Calculates the probability of the player being close to a given object.
    A threshold of 2 is used for proximity on the grid.
    """
    proximity_threshold = 2.0
    
    player_x = player[:, 0]
    player_y = player[:, 1]
    
    obj_x = obj[:, 0]
    obj_y = obj[:, 1]
    
    # Visibility flag for the object (0 or 1)
    obj_prob = obj[:, 3] 

    # Calculate Euclidean distance
    dist_sq = (player_x - obj_x).pow(2) + (player_y - obj_y).pow(2)
    dist = dist_sq.sqrt()
    
    # Probability is 1 if within threshold, 0 otherwise, multiplied by object visibility
    is_close = bool_to_probs(dist < proximity_threshold)
    
    return is_close * obj_prob


def above(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True if player is above obj (smaller y-coordinate)."""
    player_y = player[:, 1]
    obj_y = obj[:, 1]
    obj_prob = obj[:, 3] # visibility
    return bool_to_probs(player_y < obj_y) * obj_prob


def below(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True if player is below obj (larger y-coordinate)."""
    player_y = player[:, 1]
    obj_y = obj[:, 1]
    obj_prob = obj[:, 3] # visibility
    return bool_to_probs(player_y > obj_y) * obj_prob


def left_of(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True if player is left of obj (smaller x-coordinate)."""
    player_x = player[:, 0]
    obj_x = obj[:, 0]
    obj_prob = obj[:, 3] # visibility
    return bool_to_probs(player_x < obj_x) * obj_prob


def right_of(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True if player is right of obj (larger x-coordinate)."""
    player_x = player[:, 0]
    obj_x = obj[:, 0]
    obj_prob = obj[:, 3] # visibility
    return bool_to_probs(player_x > obj_x) * obj_prob

def blocked(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """
    Checks if the player is blocked by an object in front of it.
    """
    ax = player[:, 0]
    ay = player[:, 1]
    ad = player[:, 2] # direction

    ox = obj[:, 0]
    oy = obj[:, 1]
    ob_prob = obj[:, 3]

    # Calculate position in front of the agent
    front_x = ax.clone()
    front_y = ay.clone()

    # ad: 0:right, 1:down, 2:left, 3:up
    front_x[ad == 0] += 1
    front_y[ad == 1] += 1
    front_x[ad == 2] -= 1
    front_y[ad == 3] -= 1

    # Check if obstacle is in front of the agent
    is_in_front = (front_x == ox) & (front_y == oy)
    
    return bool_to_probs(is_in_front) * ob_prob
