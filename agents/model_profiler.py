import torch
from torch.utils.tensorboard import SummaryWriter
from torchprof import Profile
from tqdm import tqdm

from .base import BaseAgent
from util.cuda import set_cuda_devices
from util.reflect import init_class, init_data
from util.seed import set_seed
from util.meters import AverageMeter


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
        self.logger.info("Batch Size: %d", self.eval_loader.batch_size)
        num_params = sum([p.numel() for p in self.model.parameters()])
        num_lrn_p = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        self.logger.info(
            "Num Parameters: %(params)d (%(lrn_params)d requires gradient)",
            {"params": num_params, "lrn_params": num_lrn_p},
        )

    def run(self):
        self_cpu_meter = AverageMeter("Self CPU Time")
        cpu_time_meter = AverageMeter("CPU Time Total")
        cuda_time_meter = AverageMeter("CUDA Time Total")

        with tqdm(self.eval_loader) as t:
            for inputs, _ in t:
                # with torch.autograd.profiler.profile(use_cuda=self.use_cuda) as prof:
                with Profile(self.model, use_cuda=self.use_cuda) as prof:
                    self.model(inputs)
                traces, measures = prof.raw()
                self_cpu_time = 0
                cpu_time_total = 0
                cuda_time_total = 0
                for trace, measure in measures.items():
                    if not measure:
                        continue
                    one_pass = measure[0]
                    self_cpu_time += sum([e.self_cpu_time_total for e in one_pass])
                    cpu_time_total += sum([e.cpu_time_total for e in one_pass])
                    cuda_time_total += sum([e.cuda_time_total for e in one_pass])

                self_cpu_meter.update(self_cpu_time)
                cpu_time_meter.update(cpu_time_total)
                cuda_time_meter.update(cuda_time_total)

                t.set_description(
                    "Self CPU Time: {}, CPU Time Total: {}, CUDA Time Total: {}".format(
                        torch.autograd.profiler.format_time(self_cpu_meter.avg),
                        torch.autograd.profiler.format_time(cpu_time_meter.avg),
                        torch.autograd.profiler.format_time(cuda_time_meter.avg),
                    )
                )
        self.logger.info(
            "Average Self CPU Time: %s, CPU Time Total: %s, CUDA Time Total: %s",
            torch.autograd.profiler.format_time(self_cpu_meter.avg),
            torch.autograd.profiler.format_time(cpu_time_meter.avg),
            torch.autograd.profiler.format_time(cuda_time_meter.avg),
        )
        self.logger.info(
            "STD Self CPU Time: %s, CPU Time Total: %s, CUDA Time Total: %s",
            torch.autograd.profiler.format_time(self_cpu_meter.std),
            torch.autograd.profiler.format_time(cpu_time_meter.std),
            torch.autograd.profiler.format_time(cuda_time_meter.std),
        )

    def finalize(self):
        self.tb_sw.close()
