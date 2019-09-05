import torch
import random


def set_seed(seed, logger=None):
    """Set python random seed, pytorch manual seed, and enable deterministic CUDNN
    """
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if logger:
        logger.info("Manual seed %s", seed)
        logger.warning("Seed training is enabled with deterministic CUDNN.")
        logger.warning("Model training will slow down considerably.")
        logger.warning("Restarting from checkpoints is undefined behaviour.")
