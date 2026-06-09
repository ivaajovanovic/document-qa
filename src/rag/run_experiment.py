import sys
sys.path.append(".")

import json
import logging
from src.rag.experiment import run_experiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_all_experiments(config_path: str = "./experiments/config.json") -> None:
    """
    Runs all experiments defined in config file.

    Args:
        config_path: Path to experiments config JSON.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    arxiv_id = config["arxiv_id"]
    questions = config["questions"]
    experiments = config["experiments"]

    logger.info(f"Running {len(experiments)} experiments on paper {arxiv_id}")
    logger.info(f"Questions: {len(questions)}")

    for i, exp_config in enumerate(experiments):
        logger.info(f"\nExperiment {i+1}/{len(experiments)}: {exp_config['id']}")
        try:
            run_experiment(exp_config, questions, arxiv_id)
        except Exception as e:
            logger.error(f"Experiment {exp_config['id']} failed: {e}")
            continue

    logger.info("\nAll experiments done!")


if __name__ == "__main__":
    run_all_experiments()