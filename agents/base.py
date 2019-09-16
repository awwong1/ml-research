from logging import getLogger


class BaseAgent:
    def __init__(self, config):
        """Agent constructor, called by main.py
        """
        self.config = config
        self.logger = getLogger(config.get("exp_name"))

    def run(self):
        """Experiment's main logic, called by main.py.
        """
        raise NotImplementedError

    def finalize(self):
        """Called after `run` by main.py. Clean up experiment.
        """
        # raise NotImplementedError
        return
