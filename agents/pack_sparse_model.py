import torch

from .base import BaseAgent
from util.cuda import set_cuda_devices
from util.reflect import init_class
from models.mask import MaskSTE
from models.cifar.vgg import VGG, make_layers


class MaskableVGGPackingAgent(BaseAgent):
    """Agent for packing sparse models with masks (vgg19_bn_maskable)"""

    def __init__(self, config):
        super(MaskableVGGPackingAgent, self).__init__(config)

        # Setup CUDA
        cpu_only = config.get("cpu_only", False)
        self.use_cuda, self.gpu_ids = set_cuda_devices(
            cpu_only, config.get("gpu_ids"), logger=self.logger
        )
        map_location = None if self.use_cuda else torch.device("cpu")

        self.model = init_class(config.get("model"))

        pack_checkpoint = config["pack"]
        self.logger.info("Packing model from checkpoint: %s", pack_checkpoint)
        pack_checkpoint = torch.load(pack_checkpoint, map_location=map_location)
        self.model.load_state_dict(pack_checkpoint["state_dict"])

    def run(self):
        make_layers_config = []
        modules = list(self.model.modules())

        # Construct the model with smaller architecture
        self.logger.info("Constructing reduced model architecture...")
        apply_batch_norm = False
        num_classes = None
        for idx, module in enumerate(modules):
            if len(list(module.children())) > 0:
                continue
            if type(module) == torch.nn.Conv2d:
                continue
            elif type(module) == torch.nn.BatchNorm2d:
                apply_batch_norm = True
            elif type(module) == torch.nn.MaxPool2d:
                make_layers_config.append("M")
            elif type(module) == MaskSTE:
                bmask, _ = module.get_binary_mask()
                out_channels = int(sum(bmask.view(-1)).data.item())
                make_layers_config.append(out_channels)
            elif type(module) == torch.nn.Linear:
                assert num_classes is None
                num_classes = module.out_features
        print(make_layers_config)
        pack_model = VGG(
            make_layers(make_layers_config, batch_norm=apply_batch_norm),
            num_classes=num_classes,
            classifier_input_features=out_channels,
        )
        print(pack_model)
