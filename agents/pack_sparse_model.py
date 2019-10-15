import torch

from .base import BaseAgent
from util.cuda import set_cuda_devices
from util.reflect import init_class
from util.checkpoint import save_checkpoint
from models.mask import MaskSTE
from models.cifar.vgg import VGG, make_layers


class MaskablePackingAgent(BaseAgent):
    """Agent for packing sparse models with masks (vgg19_bn_maskable)"""

    def __init__(self, config):
        super(MaskablePackingAgent, self).__init__(config)

        # Setup CUDA
        cpu_only = config.get("cpu_only", False)
        self.use_cuda, self.gpu_ids = set_cuda_devices(
            cpu_only, config.get("gpu_ids"), logger=self.logger
        )
        map_location = None if self.use_cuda else torch.device("cpu")

        self.model = init_class(config.get("model"))
        self.apply_sigmoid = config.get("apply_sigmoid", True)
        pack_checkpoint = config["pack"]
        self.logger.info("Packing model from checkpoint: %s", pack_checkpoint)
        pack_checkpoint = torch.load(pack_checkpoint, map_location=map_location)
        self.model.load_state_dict(pack_checkpoint["state_dict"])

    def run(self):
        modules = list(self.model.modules())

        # Construct the model with smaller architecture
        if type(modules[0]) == VGG:
            binary_masks = [
                torch.Tensor([1, 1, 1])
            ]  # VGG Input RGB channels are not masked
            make_layers_config, pack_model = MaskablePackingAgent.gen_vgg_make_layers(
                modules, binary_masks
            )
            self.logger.info("Packed Model make_layers list: %s", make_layers_config)
            MaskablePackingAgent.transfer_vgg_parameters(
                self.model, pack_model, binary_masks
            )
            self.logger.info("Packed model: %s", pack_model)

            num_params = sum([p.numel() for p in pack_model.parameters()])
            num_lrn_p = sum(
                [p.numel() for p in pack_model.parameters() if p.requires_grad]
            )
            self.logger.info(
                "Num Parameters: %(params)d (%(lrn_params)d requires gradient)",
                {"params": num_params, "lrn_params": num_lrn_p},
            )
            save_checkpoint(
                {
                    "make_layers": make_layers_config,
                    "state_dict": pack_model.state_dict(),
                    "params": num_params,
                    "lrn_params": num_lrn_p,
                },
                False,
                checkpoint_dir=self.config["chkpt_dir"],
                filename="vgg-pack-{:.2e}.pth.tar".format(num_params),
            )
        else:
            raise NotImplementedError("Cannot pack sparse module: %s", modules[0])

    @staticmethod
    def transfer_vgg_parameters(base_model, pack_model, binary_masks):
        modules_packed = list(pack_model.modules())
        modules = list(base_model.modules())
        module_idx = 0
        mask_idx = 1
        for module in modules:
            module_packed = modules_packed[module_idx]
            if type(module) == MaskSTE:
                mask_idx += 1
                continue

            module_idx += 1
            if len(list(module.children())) == 0:
                assert type(module) == type(module_packed)
                if type(module) == torch.nn.MaxPool2d:
                    continue
                pre_prune = binary_masks[mask_idx - 1].view(-1).nonzero().squeeze()
                cur_prune = (
                    binary_masks[mask_idx].view(-1).nonzero().squeeze()
                    if mask_idx < len(binary_masks)
                    else None
                )
                # copy all pruned parameters over
                if type(module) == torch.nn.Conv2d:
                    module_packed.weight.data.copy_(
                        module.weight[cur_prune, ...][:, pre_prune, ...]
                    )
                    module_packed.bias.data.copy_(module.bias[cur_prune, ...])
                elif type(module) == torch.nn.BatchNorm2d:
                    module_packed.weight.data.copy_(module.weight[cur_prune, ...])
                    module_packed.bias.data.copy_(module.bias[cur_prune, ...])
                    module_packed.running_mean.data.copy_(
                        module.running_mean[cur_prune, ...]
                    )
                    module_packed.running_var.data.copy_(
                        module.running_var[cur_prune, ...]
                    )
                elif type(module) == torch.nn.Linear:
                    module_packed.weight.data.copy_(module.weight[:, pre_prune])
                    module_packed.bias.data.copy_(module.bias)

    @staticmethod
    def gen_vgg_make_layers(modules, binary_masks, use_cuda=False, vgg_class=VGG):
        make_layers_config = []
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
                binary_masks.append(bmask)
            elif type(module) == torch.nn.Linear:
                assert num_classes is None
                num_classes = module.out_features
        pack_model = vgg_class(
            make_layers(make_layers_config, batch_norm=apply_batch_norm),
            num_classes=num_classes,
            classifier_input_features=out_channels,
        )
        if use_cuda:
            pack_model = pack_model.cuda()
        return make_layers_config, pack_model

    @staticmethod
    def insert_masks_into_model(model, use_cuda=False):
        make_layers_config = []
        apply_batch_norm = False
        num_classes = None
        for idx, module in enumerate(model.modules()):
            if len(list(module.children())) > 0:
                continue
            if type(module) == torch.nn.Conv2d:
                make_layers_config.append(module.out_channels)
                continue
            elif type(module) == torch.nn.BatchNorm2d:
                apply_batch_norm = True
            elif type(module) == torch.nn.MaxPool2d:
                make_layers_config.append("M")
            elif type(module) == torch.nn.Linear:
                assert num_classes is None
                out_channels = module.in_features
                num_classes = module.out_features
        model_with_masks = VGG(
            make_layers(
                make_layers_config, batch_norm=apply_batch_norm, sparsity_mask=True
            ),
            num_classes=num_classes,
            classifier_input_features=out_channels,
        )
        if use_cuda:
            model_with_masks = model_with_masks.cuda()
        # copy over the weights
        modules_pretrained = list(model.modules())
        modules_to_prune = list(model_with_masks.modules())
        module_idx = 0
        for module_to_prune in modules_to_prune:
            module_pretrained = modules_pretrained[module_idx]
            modstr = str(type(module_to_prune))
            # Skip the masking layers
            if type(module_to_prune) == MaskSTE:
                continue
            if len(list(module_to_prune.children())) == 0:
                assert modstr == str(type(module_pretrained)), "{} !== {}".format(
                    modstr, str(type(module_pretrained))
                )
                # copy all parameters over
                param_lookup = dict(module_pretrained.named_parameters())
                for param_key, param_val in module_to_prune.named_parameters():
                    param_val.data.copy_(param_lookup[param_key].data)
                # BatchNorm layers are special and require copying of running_mean/running_var
                if "BatchNorm" in modstr:
                    module_to_prune.running_mean.copy_(module_pretrained.running_mean)
                    module_to_prune.running_var.copy_(module_pretrained.running_var)
            module_idx += 1
        return model_with_masks
