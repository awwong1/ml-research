import torch
from torch.utils.tensorboard import SummaryWriter
from time import time

from .base import BaseAgent
from models.admm.c_admm import ConstrainedADMM_P, ConstrainedADMM_D
from util.cuda import set_cuda_devices
from util.reflect import init_class, init_data
from util.seed import set_seed


class ADMMProxTrainAgent(BaseAgent):
    def __init__(self, config):
        super(ADMMProxTrainAgent, self).__init__(config)

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
        self.train_set, self.train_loader = init_data(config.get("train_data"))
        self.eval_set, self.eval_loader = init_data(config.get("eval_data"))

        # Instantiate Model (teacher, to_prune)
        self.model = init_class(config.get("model"))
        self.teacher_model = init_class(config.get("model"))
        try:
            # Try to visualize tensorboard model graph structure
            model_input, _target = next(iter(self.eval_set))
            self.tb_sw.add_graph(self.model, model_input.unsqueeze(0))
        except Exception as e:
            self.logger.warn(e)

        # Load model weights
        model_checkpoint = torch.load(config["model_checkpoint"])
        self.teacher_model.load_state_dict(model_checkpoint["state_dict"])
        self.model.load_state_dict(model_checkpoint["state_dict"])

        # Support multiple GPUs using DataParallel
        if self.use_cuda:
            if len(self.gpu_ids) > 1:
                self.teacher_model = torch.nn.DataParallel(self.teacher_model).cuda()
                self.model = torch.nn.DataParallel(self.model).cuda()
            else:
                self.teacher_model = self.teacher_model.cuda()
                self.model = self.model.cuda()

        # ADMM Model Wrappers
        width_ub = config.get("width_ub", [])
        zinit = config.get("zinit", 0.0)
        yinit = config.get("yinit", 0.0)
        config.get("constraint_model")
        primal_model = ConstrainedADMM_P(self.model, width_ub)
        dual_model = ConstrainedADMM_D(len(width_ub), zinit, yinit)
        # TODO EnergyEstimateNet (approximation)
        self.logger.info(primal_model, dual_model)

    def run(self):
        self.exp_start = time()
        pass

