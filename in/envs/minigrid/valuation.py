import torch as th
from nsfr.utils.common import bool_to_probs

# ============================================================
# Minimal batch fix helpers
# ============================================================

def ensure_batch(x):
    """
    Ensures tensor has shape (batch, ...).
    GUI mode sends unbatched 1D tensors → add batch dim.
    Vectorized envs send (batch, ...) → unchanged.
    """
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x


# ============================================================
# Your original predicates (minimal modifications only)
# ============================================================

def front_clear(agent, wall):
    agent = ensure_batch(agent)
    wall = ensure_batch(wall)

    ax = agent[:, 0]
    ay = agent[:, 1]
    ad = agent[:, 2]

    wx = wall[:, 0]
    wy = wall[:, 1]

    # DIRECTIONS (your original style)
    # 0: right, 1: down, 2: left, 3: up
    dx = th.zeros_like(ax)
    dy = th.zeros_like(ay)

    dx = th.where(ad == 0, th.ones_like(dx), dx)
    dy = th.where(ad == 1, th.ones_like(dy), dy)
    dx = th.where(ad == 2, -th.ones_like(dx), dx)
    dy = th.where(ad == 3, -th.ones_like(dy), dy)

    nx = ax + dx
    ny = ay + dy

    inbounds = (nx >= 0) & (nx < 5) & (ny >= 0) & (ny < 5)
    occupied = (nx == wx) & (ny == wy)

    return bool_to_probs((~occupied & inbounds))


def left_clear(agent, wall):
    agent = ensure_batch(agent)
    wall = ensure_batch(wall)

    ax = agent[:, 0]
    ay = agent[:, 1]
    ad = agent[:, 2]

    wx = wall[:, 0]
    wy = wall[:, 1]

    dx = th.zeros_like(ax)
    dy = th.zeros_like(ay)

    dy = th.where(ad == 0, -1, dy)
    dx = th.where(ad == 1,  1, dx)
    dy = th.where(ad == 2,  1, dy)
    dx = th.where(ad == 3, -1, dx)

    nx = ax + dx
    ny = ay + dy

    inbounds = (nx >= 0) & (nx < 5) & (ny >= 0) & (ny < 5)
    occupied = (nx == wx) & (ny == wy)

    return bool_to_probs((~occupied & inbounds))


def right_clear(agent, wall):
    agent = ensure_batch(agent)
    wall = ensure_batch(wall)

    ax = agent[:, 0]
    ay = agent[:, 1]
    ad = agent[:, 2]

    wx = wall[:, 0]
    wy = wall[:, 1]

    dx = th.zeros_like(ax)
    dy = th.zeros_like(ay)

    dy = th.where(ad == 0,  1, dy)
    dx = th.where(ad == 1, -1, dx)
    dy = th.where(ad == 2, -1, dy)
    dx = th.where(ad == 3,  1, dx)

    nx = ax + dx
    ny = ay + dy

    inbounds = (nx >= 0) & (nx < 5) & (ny >= 0) & (ny < 5)
    occupied = (nx == wx) & (ny == wy)

    return bool_to_probs((~occupied & inbounds))


def blocked_ahead(agent, wall):
    agent = ensure_batch(agent)
    wall = ensure_batch(wall)

    ax = agent[:, 0]
    ay = agent[:, 1]
    ad = agent[:, 2]

    wx = wall[:, 0]
    wy = wall[:, 1]

    dx = th.zeros_like(ax)
    dy = th.zeros_like(ay)

    dx = th.where(ad == 0, 1, dx)
    dy = th.where(ad == 1, 1, dy)
    dx = th.where(ad == 2, -1, dx)
    dy = th.where(ad == 3, -1, dy)

    nx = ax + dx
    ny = ay + dy

    return bool_to_probs(((nx == wx) & (ny == wy)))


def on_goal(agent, goal):
    agent = ensure_batch(agent)
    goal = ensure_batch(goal)

    ax = agent[:, 0]
    ay = agent[:, 1]

    gx = goal[:, 0]
    gy = goal[:, 1]

    return bool_to_probs(((ax == gx) & (ay == gy)))


def adjacent_goal(agent, goal):
    agent = ensure_batch(agent)
    goal = ensure_batch(goal)

    ax = agent[:, 0]
    ay = agent[:, 1]

    gx = goal[:, 0]
    gy = goal[:, 1]

    d = (th.abs(ax - gx) + th.abs(ay - gy))
    return bool_to_probs((d == 1))

def enemy_close(agent, enemy):
    ax, ay = agent[:,0], agent[:,1]
    ex, ey = enemy[:,0], enemy[:,1]
    dist = (ax - ex).abs() + (ay - ey).abs()
    return (dist <= 1).float()  # 1 tile away


def enemy_ahead(agent, enemy):
    ax, ay, ad = agent[:,0], agent[:,1], agent[:,2]
    ex, ey = enemy[:,0], enemy[:,1]

    dx = ex - ax
    dy = ey - ay

    # ad: 0=right,1=down,2=left,3=up (MiniGrid standard)
    cond = (
            ((ad==0) & (dx>0) & (dy==0)) |
            ((ad==2) & (dx<0) & (dy==0)) |
            ((ad==1) & (dy>0) & (dx==0)) |
            ((ad==3) & (dy<0) & (dx==0))
    )
    return cond.float()


def enemy_left(agent, enemy):
    ax, ay, ad = agent[:,0], agent[:,1], agent[:,2]
    ex, ey = enemy[:,0], enemy[:,1]

    dx = ex - ax
    dy = ey - ay

    # ad: 0=right,1=down,2=left,3=up (MiniGrid standard)
    cond = (
            ((ad==0) & (dx>0) & (dy==0)) |
            ((ad==2) & (dx<0) & (dy==0)) |
            ((ad==1) & (dy>0) & (dx==0)) |
            ((ad==3) & (dy<0) & (dx==0))
    )
    return cond.float()


def enemy_right(agent, enemy):
    ax, ay, ad = agent[:,0], agent[:,1], agent[:,2]
    ex, ey = enemy[:,0], enemy[:,1]

    dx = ex - ax
    dy = ey - ay

    # ad: 0=right,1=down,2=left,3=up (MiniGrid standard)
    cond = (
            ((ad==0) & (dx>0) & (dy==0)) |
            ((ad==2) & (dx<0) & (dy==0)) |
            ((ad==1) & (dy>0) & (dx==0)) |
            ((ad==3) & (dy<0) & (dx==0))
    )
    return cond.float()
