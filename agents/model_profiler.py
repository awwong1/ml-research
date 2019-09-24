import torch
from torch.utils.tensorboard import SummaryWriter
from torchprof import Profile

from .base import BaseAgent
from util.cuda import set_cuda_devices
from util.reflect import init_class, init_data
from util.seed import set_seed
from tqdm import tqdm


class ModelProfilerAgent(BaseAgent):
    def __init__(self, config):
        super(ModelProfilerAgent, self).__init__(config)

        # TensorBoard Summary Writer
        self.tb_sw = SummaryWriter(log_dir=config["tb_dir"])
        self.tb_sw.add_text("config", str(config))

        # Configure seed, if provided
        seed = config.get("seed")
        set_seed(seed, logger=self.logger)
        self.tb_sw.add_text("seed", str(seed))

        # Setup CUDA
        cpu_only = config.get("cpu_only", False)
        self.use_cuda, self.gpu_ids = set_cuda_devices(
            cpu_only, config.get("gpu_ids"), logger=self.logger
        )

        # Instantiate Datasets and Dataloaders
        self.eval_set, self.eval_loader = init_data(config.get("eval_data"))

        # Instantiate Models
        self.model = init_class(config.get("model"))
        self.model.eval()
        try:
            # Try to visualize tensorboard model graph structure
            model_input, _target = next(iter(self.eval_set))
            self.tb_sw.add_graph(self.model, model_input.unsqueeze(0))
        except Exception as e:
            self.logger.warn(e)

        # Log the classification experiment details
        self.logger.info("Eval Dataset: %s", self.eval_set)
        self.logger.info("Model: %s", self.model)
        num_params = sum([p.numel() for p in self.model.parameters()])
        num_lrn_p = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        self.logger.info(
            "Num Parameters: %(params)d (%(lrn_params)d requires gradient)",
            {"params": num_params, "lrn_params": num_lrn_p},
        )

    def run(self):
        for inputs, _ in tqdm(self.eval_loader):
            # with Profile(self.model, use_cuda=self.use_cuda) as prof:
            with torch.autograd.profiler.profile(use_cuda=self.use_cuda) as prof:
                self.model(inputs)
            print(prof.total_average)
            print()

    def finalize(self):
        self.tb_sw.close()
