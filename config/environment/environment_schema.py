from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    """Environment configuration."""

    env_name: str = "default_environment"
    variable_motif: bool = False
    variable_length: bool = False
