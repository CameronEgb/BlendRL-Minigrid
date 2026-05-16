import gymnasium as gym

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


def make_env(env, clip_rewards=False):
    """
    Standard Atari wrapper stack. 
    Note: RecordEpisodeStatistics is at the TOP to record agent steps (e.g. 20M)
    rather than raw frames (e.g. 80M or 320M).
    """
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    
    # User requested NO clipping ever
    if clip_rewards:
        env = ClipRewardEnv(env)
        
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    
    # Place monitor at the very top so it sees exactly what the agent sees
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


kangaroo_modifs = [
    "disable_coconut",
    "randomize_kangaroo_position",
    "change_level_0",
]

seaquest_modifs = []
