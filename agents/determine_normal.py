import numpy as np
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import ConcatDataset

from .base import BaseAgent
from util.reflect import init_data


class CalculateNormalizeAgent(BaseAgent):
    """Agent for determining parameters to use in torchvision.transforms.Normalize
    """

    def __init__(self, config):
        super(CalculateNormalizeAgent, self).__init__(config)
        ds_configs = config.get("datasets")
        self.datasets = []
        for ds_config in ds_configs:
            ds, dl = init_data(ds_config)
            self.datasets.append(ds)

    def run(self):
        for dataset in self.datasets:
            ds_means, ds_stds = self.derive_normalize(dataset)
            self.logger.info(dataset)
            self.logger.info("mean %s, std %s", ds_means, ds_stds)
        if len(self.datasets) > 1:
            self.logger.info("All datasets combined: ")
            all_means, all_stds = self.derive_normalize(ConcatDataset(self.datasets))
            self.logger.info("mean %s, std %s", all_means, all_stds)

    def derive_normalize(self, dataset):
        ds_values = defaultdict(list)
        for inputs, _targets in tqdm(dataset):
            channels = inputs.size()[0]
            for channel in range(channels):
                ds_values[channel].append(inputs[channel, :, :])

        ds_means = []
        ds_stds = []
        for channel in range(channels):
            data = np.dstack(ds_values[channel])
            ds_means.append(np.mean(data))
            ds_stds.append(np.std(data))
        return ds_means, ds_stds
