import logging
import os
import sys

import hydra
from omegaconf import OmegaConf

from config.config_schema import Config
from config.config_utils import print_config_table
from learner.learner import Agent

# Add src directory to path and import to register the environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
import protein_design_env  # noqa: E402, F401 - Import to register environment


@hydra.main(version_base=None, config_path="config", config_name="defaults")
def main(cfg: Config) -> None:
    """Main function for Problem 2."""
    print_config_table(cfg, style="tree")
    # Temporarily disable struct mode to add BASE_DIR
    OmegaConf.set_struct(cfg, False)
    cfg.BASE_DIR = BASE_DIR
    OmegaConf.set_struct(cfg, True)

    if cfg.dir is None:
        cfg.dir = os.path.join(BASE_DIR, "saved-model", f"{cfg.algo}_Protein_Design_rng_length")
    agent = Agent(cfg)
    if cfg.mode == 1:
        logging.info(f"-----------Start Training with {cfg.algo}-----------")
        agent.train()


if __name__ == "__main__":
    main()
