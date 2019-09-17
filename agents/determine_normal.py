import torch
from tqdm import tqdm
# from torch.utils.data import ConcatDataset

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
        c_iters = []
        for dataset in self.datasets:
            ds_means, ds_stds, _c_iter = self.derive_normalize(dataset)
            self.logger.info(dataset)
            self.logger.info("mean %s, std %s", ds_means, ds_stds)
            c_iters.append(_c_iter)
        if len(self.datasets) > 1:
            self.logger.info("All datasets combined: ")
            # all_means, all_stds = self.derive_normalize(ConcatDataset(self.datasets))
            # self.logger.info("mean %s, std %s", all_means, all_stds)
            all_iter = {}
            all_means = []
            all_stds = []
            for c_iter in c_iters:
                for channel in range(len(ds_means)):
                    all_sum, all_sqsum, all_count = all_iter.get(channel, (0, 0, 0))
                    c_sum, c_sqsum, c_count = c_iter.get(channel, (0, 0, 0))
                    all_sum += c_sum
                    all_sqsum += c_sqsum
                    all_count += c_count
                    all_iter[channel] = (all_sum, all_sqsum, all_count)
            for channel in range(len(ds_means)):
                all_sum, all_sqsum, all_count = all_iter.get(channel, (0, 0, 0))
                mean = all_sum / all_count
                std = ((all_sqsum / all_count) - (mean * mean)) ** 0.5
                all_means.append(mean.data.item())
                all_stds.append(std.data.item())
            self.logger.info("mean %s, std %s", all_means, all_stds)

    @torch.no_grad()
    def derive_normalize(self, dataset):
        self.logger.info(dataset)
        sample, _target = next(iter(dataset))
        channels = sample.size()[0]

        ds_means = []
        ds_stds = []
        c_iter = {}

        # iterative methods, as OOM errors are frequent with large data
        # data = torch.stack(tuple(inp[channel] for inp, _ in t))
        # ds_means.append(data.mean())
        # ds_stds.append(data.std())
        for inp, _ in tqdm(dataset):
            for channel in range(channels):
                c_sum, c_sqsum, c_count = c_iter.get(channel, (0, 0, 0))
                c_sum += inp[channel].sum()
                c_sqsum += (inp[channel] ** 2).sum()
                c_count += inp[channel].numel()
                c_iter[channel] = (c_sum, c_sqsum, c_count)

        for channel in range(channels):
            c_sum, c_sqsum, c_count = c_iter.get(channel, (0, 0, 0))
            mean = c_sum / c_count
            std = ((c_sqsum / c_count) - (mean * mean)) ** 0.5
            ds_means.append(mean.data.item())
            ds_stds.append(std.data.item())

        return ds_means, ds_stds, c_iter
