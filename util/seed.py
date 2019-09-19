import torch
import random


def set_seed(seed, logger=None):
    """Set python random seed, pytorch manual seed, and enable deterministic CUDNN
    seed: True, None, int
    - if None (or falsy), do not set a seed
    - if True, set a manual seed using torch.seed()
    - if type(seed) is int, set manual seed
    """
    if not seed:
        return
    if seed is True:
        seed = torch.seed()
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if logger:
        logger.info("Manual seed %s", seed)
        logger.warning("Seed training is enabled with deterministic CUDNN.")
        logger.warning("Model training will slow down considerably.")
        logger.warning("Restarting from checkpoints is undefined behaviour.")
