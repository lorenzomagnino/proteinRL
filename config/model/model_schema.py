from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    name: str = "default_model"
    manual: bool = False
    manual_save_bool: bool = True
    dir: str = None
