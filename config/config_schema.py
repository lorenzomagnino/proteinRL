"""Configuration schemas using dataclasses for type safety."""

from dataclasses import dataclass


@dataclass
class Config:
    """Root configuration combining all sub-configs."""

    algo: str = "PPO"  # Options: PPO, DQN, A2C
    timesteps: int = 50000
    mode: int = 1  # 1: train, 2: test
    seed: int = 0
    env_name: str = "Protein-Design-v0"
    variable_motif: bool = False
    variable_length: bool = False
    manual: bool = False
    model_save_bool: bool = True
    dir: str = None
    test_episodes: int = 2
    take_best_model: bool = False
