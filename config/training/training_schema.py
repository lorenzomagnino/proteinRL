from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""

    timesteps: int = 50000
    mode: int = 1  # 1: train, 2: test
    seed: int = 0
