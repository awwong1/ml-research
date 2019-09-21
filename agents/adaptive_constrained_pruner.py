import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from time import time
from pprint import pformat

from .base import BaseAgent
from models.mask import MaskSTE
from util.accuracy import calculate_accuracy
from util.checkpoint import save_checkpoint
from util.cuda import set_cuda_devices
from util.losses import calculate_kd_loss
from util.meters import AverageMeter
from util.reflect import init_class, init_data
from util.seed import set_seed
from util.tablogger import TabLogger


class AdaptivePruningAgent(BaseAgent):
    """Agent for constraineed knowledge distillation and pruning experiments."""

    def __init__(self, config):
        super(AdaptivePruningAgent, self).__init__(config)

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

        # Instantiate Models
        self.pretrained_model = init_class(config.get("pretrained_model"))
        self.model = init_class(config.get("model"))

        # Load the pretrained weights
        map_location = None if self.use_cuda else torch.device("cpu")
        prune_checkpoint = config["prune"]
        self.logger.info(
            "Loading pretrained model from checkpoint: %s", prune_checkpoint
        )
        prune_checkpoint = torch.load(prune_checkpoint, map_location=map_location)
        self.pretrained_model.load_state_dict(prune_checkpoint["state_dict"])
        self.pretrained_model.eval()
        modules_pretrained = list(self.pretrained_model.modules())
        modules_to_prune = list(self.model.modules())
        module_idx = 0
        self.mask_modules = []
        for module_to_prune in modules_to_prune:
            module_pretrained = modules_pretrained[module_idx]
            modstr = str(type(module_to_prune))
            # Skip the masking layers
            if "MaskSTE" in modstr:
                self.mask_modules.append(module_to_prune)
                continue
            if len(list(module_to_prune.children())) == 0:
                assert modstr == str(type(module_pretrained))
                # copy all parameters over
                param_lookup = dict(module_pretrained.named_parameters())
                for param_key, param_val in module_to_prune.named_parameters():
                    param_val.data.copy_(param_lookup[param_key].data)
                # BatchNorm layers are special and require copying of running_mean/running_var
                if "BatchNorm" in modstr:
                    module_to_prune.running_mean.copy_(module_pretrained.running_mean)
                    module_to_prune.running_var.copy_(module_pretrained.running_var)
            module_idx += 1

        try:
            # Try to visualize tensorboard model graph structure
            model_input, _target = next(iter(self.eval_set))
            self.tb_sw.add_graph(self.model, model_input.unsqueeze(0))
        except Exception as e:
            self.logger.warn(e)

        # Instantiate task loss and optimizer
        self.task_loss_fn = init_class(config.get("task_loss"))
        self.mask_loss_fn = init_class(config.get("mask_loss"))
        self.optimizer = init_class(config.get("optimizer"), self.model.parameters())
        self.temperature = config.get("temperature", 4.0)
        self.task_loss_reg = config.get("task_loss_reg", 1.0)
        self.mask_loss_reg = config.get("mask_loss_reg", 1.0)
        self.kd_loss_reg = config.get("kd_loss_reg", 1.0)

        # Misc. Other classification hyperparameters
        self.epochs = config.get("epochs", 300)
        self.start_epoch = config.get("start_epoch", 0)
        self.gamma = config.get("gamma", 0.1)
        self.schedule = config.get("schedule", 30)
        self.lr = self.optimizer.param_groups[0]["lr"]
        self.best_acc_per_usage = {}

        # Log the classification experiment details
        self.logger.info("Train Dataset: %s", self.train_set)
        self.logger.info("Eval Dataset: %s", self.eval_set)
        self.logger.info("Task Loss (Criterion): %s", self.task_loss_fn)
        self.logger.info("Model Optimizer: %s", self.optimizer)
        self.logger.info("Model: %s", self.model)
        num_params = sum([p.numel() for p in self.model.parameters()])
        num_lrn_p = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        self.logger.info(
            "Num Parameters: %(params)d (%(lrn_params)d requires gradient)",
            {"params": num_params, "lrn_params": num_lrn_p},
        )
        self.logger.info(
            "LR: %(lr)f decreasing by a factor of %(gamma)f at every %(schedule)s epochs",
            {"lr": self.lr, "gamma": self.gamma, "schedule": self.schedule},
        )

        self.budget = config.get("budget", 4300000)
        self.criteria = config.get("criteria", "parameters")
        if self.criteria == "parameters":
            self.og_usage = sum(self.calculate_model_sparsity()).data.item()
        else:
            raise NotImplementedError("Unknown criteria: {}".format(self.criteria))
        self.logger.info(
            "Pruning from {:.2e} {} to {:.2e} {}.".format(
                self.og_usage, self.criteria, self.budget, self.criteria
            )
        )
        self.prune_difficulty_patience = config.get("prune_difficulty_patience", 4)
        self.prune_reg_multiplier = config.get("prune_reg_multiplier", 1.1)
        self.adaptive_difficulty = config.get("adaptive_difficulty_init", 1.0)
        self.logger.info(
            "Increasing adaptive difficulty of %.2f by factor of %.2f every %d epochs with no %s decrease",
            self.adaptive_difficulty,
            self.prune_reg_multiplier,
            self.prune_difficulty_patience,
            self.criteria,
        )

        # Path to in progress checkpoint.pth.tar for resuming experiment
        resume = config.get("resume")
        t_log_fpath = os.path.join(config["out_dir"], "epoch.out")
        self.t_log = TabLogger(t_log_fpath, resume=bool(resume))
        self.t_log.set_names(
            [
                "Epoch",
                "Train Task Loss",
                "Train KD Loss",
                "Train Mask Loss",
                "Train ADMM Mask Loss",
                "Train Acc",
                "Eval Task Loss",
                "Eval KD Loss",
                "Eval Mask Loss",
                "Eval ADMM Mask Loss",
                "Eval Acc",
                "LR",
            ]
        )
        if resume:
            self.logger.info("Resuming from checkpoint: %s", resume)
            res_chkpt = torch.load(resume, map_location=map_location)
            self.start_epoch = res_chkpt["epoch"]
            self.model.load_state_dict(res_chkpt["state_dict"])
            eval_acc = res_chkpt["acc"]
            self.best_acc_per_usage = res_chkpt["best_acc_per_usage"]
            self.optimizer.load_state_dict(res_chkpt["optim_state_dict"])
            self.logger.info(
                "Resumed at epoch %d, eval acc %.2f", self.start_epoch, eval_acc
            )
            self.logger.info(pformat(self.best_acc_per_usage))
            # fastforward LR to match current schedule
            for i in range(self.start_epoch):
                if i > 0 and i % self.schedule == 0:
                    new_lr = self.lr * self.gamma
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = new_lr

                    self.logger.info(
                        "LR fastforward from %(old)f to %(new)f at Epoch %(epoch)d",
                        {"old": self.lr, "new": new_lr, "epoch": i},
                    )
                    self.lr = new_lr

        self.logger.info(
            "Training from Epoch %(start)d to %(end)d",
            {"start": self.start_epoch, "end": self.epochs},
        )

        # Support multiple GPUs using DataParallel
        if self.use_cuda:
            if len(self.gpu_ids) > 1:
                self.pretrained_model = torch.nn.DataParallel(
                    self.pretrained_model
                ).cuda()
                self.model = torch.nn.DataParallel(self.model).cuda()
            else:
                self.pretrained_model = self.pretrained_model.cuda()
                self.model = self.model.cuda()

    def run(self):
        self.exp_start = time()
        self.patience_min_budget = self.og_usage
        self.patience_budget_counter = 0

        for epoch in range(self.start_epoch, self.epochs):
            if epoch > 0 and epoch % self.schedule == 0:
                new_lr = self.lr * self.gamma
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = new_lr

                self.logger.info(
                    "LR changed from %(old)f to %(new)f at Epoch %(epoch)d",
                    {"old": self.lr, "new": new_lr, "epoch": epoch},
                )
                self.lr = new_lr

            epoch_start = time()
            train_res = self.run_epoch_pass(epoch=epoch, train=True)
            with torch.no_grad():
                eval_res = self.run_epoch_pass(epoch=epoch, train=False)
            epoch_elapsed = time() - epoch_start

            self.log_epoch_info(epoch, train_res, eval_res, epoch_elapsed)

            # Check if budget has been reached
            if self.criteria == "parameters":
                current_usage = sum(self.calculate_model_parameters()).data.item()
                budget_residual = current_usage - self.budget
                # Calculate adaptive difficulty change
                if current_usage < self.patience_min_budget:
                    self.patience_min_budget = current_usage
                else:
                    self.patience_budget_counter += 1
                if self.patience_budget_counter >= self.prune_difficulty_patience:
                    self.patience_budget_counter = 0
                    self.adaptive_difficulty *= self.prune_reg_multiplier
                    self.logger.info(
                        "Increasing adaptive difficuly to %.2f due to no param reduction over %d epochs",
                        self.adaptive_difficulty,
                        self.prune_difficulty_patience,
                    )
            else:
                raise NotImplementedError("Unknown criteria: {}".format(self.criteria))

            if budget_residual <= 0:
                self.logger.info(
                    "Budget of %.2e %s reached! At %.2e %s",
                    self.budget,
                    self.criteria,
                    current_usage,
                    self.criteria,
                )
                break
        else:
            self.logger.info(
                "Could not reach budget of %.2e %s, at %.2e %s",
                self.budget,
                self.criteria,
                current_usage,
                self.criteria,
            )

    def run_epoch_pass(self, epoch=-1, train=True):
        overall_loss = AverageMeter("Overall Loss")
        mask_meter = AverageMeter("Mask Loss")
        task_meter = AverageMeter("Task Loss")
        kd_meter = AverageMeter("KD Loss")
        admm_mask_meter = AverageMeter("ADMM Mask Loss")
        acc1_meter = AverageMeter("Top 1 Acc")
        acc5_meter = AverageMeter("Top 5 Acc")

        self.model.train(train)
        dataloader = self.train_loader if train else self.eval_loader

        with tqdm(total=len(dataloader)) as t:
            for inputs, targets in dataloader:
                # First Step, update all of the weights for entire model + mask
                for parameters in self.model.parameters():
                    parameters.requires_grad = True
                batch_size = inputs.size(0)
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

                # Compute forward pass of the model
                outputs = self.model(inputs)

                # Compute accuracies and update meters
                prec1, prec5 = calculate_accuracy(
                    outputs.data, targets.data, topk=(1, 5)
                )
                acc1_meter.update(prec1.item(), batch_size)
                acc5_meter.update(prec5.item(), batch_size)

                # loss calculations only used during training
                teacher_outputs = self.pretrained_model(inputs)
                task_loss = self.task_loss_fn(outputs, targets).mul(self.task_loss_reg)
                task_meter.update(task_loss.data.item(), batch_size)
                kd_loss = calculate_kd_loss(
                    outputs, teacher_outputs, targets, temperature=self.temperature
                ).mul(self.kd_loss_reg)
                kd_meter.update(kd_loss.data.item(), batch_size)

                mask_loss = torch.zeros_like(task_loss)
                for mask_module in self.mask_modules:
                    mask, _ = mask_module.get_binary_mask()
                    mask_loss += self.mask_loss_fn(mask, target=torch.zeros_like(mask))
                mask_loss.div_(len(self.mask_modules)).mul_(self.mask_loss_reg)
                mask_meter.update(mask_loss.data.item(), batch_size)

                loss = task_loss + kd_loss + mask_loss
                overall_loss.update(loss.data.item(), batch_size)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Second Step, update the mask with respect to the budget
                for module in self.model.modules():
                    if len(list(module.children())) > 0:
                        continue
                    for parameters in module.parameters():
                        parameters.requires_grad = type(module) == MaskSTE
                if self.criteria == "parameters":
                    current_usage = sum(self.calculate_model_parameters()).data.item()
                else:
                    raise NotImplementedError(
                        "Unknown criteria: {}".format(self.criteria)
                    )

                # normalize over original usage value
                admm_mask_loss = torch.zeros_like(loss)
                for mask_module in self.mask_modules:
                    mask, _ = mask_module.get_binary_mask()
                    admm_mask_loss += self.mask_loss_fn(
                        mask, target=torch.zeros_like(mask)
                    )
                admm_mask_loss.div_(len(self.mask_modules)).mul(self.mask_loss_reg)
                admm_mask_loss = admm_mask_loss + admm_mask_loss.mul(
                    self.adaptive_difficulty
                )

                admm_mask_meter.update(admm_mask_loss.data.item(), batch_size)

                if train:
                    self.optimizer.zero_grad()
                    admm_mask_loss.backward()
                    self.optimizer.step()

                t.set_description(
                    "{mode} Epoch {epoch}/{epochs} ".format(
                        mode="Train" if train else "Eval",
                        epoch=epoch,
                        epochs=self.epochs,
                    )
                    + "Task Loss: {loss:.4f} | KD Loss: {kd:.4f} | Mask Loss: {ml:.4f} | ".format(
                        loss=task_meter.avg, kd=kd_meter.avg, ml=mask_meter.avg
                    )
                    + "ADMM Loss {al:.4f} | ".format(al=admm_mask_meter.avg)
                    + "top1: {top1:.2f}% | ".format(top1=acc1_meter.avg)
                    + "params: {:.2e}".format(current_usage)
                )
                t.update()

        return {
            "overall_loss": overall_loss.avg,
            "task_loss": task_meter.avg,
            "kd_loss": kd_meter.avg,
            "mask_loss": mask_meter.avg,
            "admm_mask_loss": admm_mask_meter.avg,
            "top1_acc": acc1_meter.avg,
            "top5_acc": acc5_meter.avg,
        }

    def log_epoch_info(self, epoch, train_res, eval_res, epoch_elapsed):
        param_usage = 0
        epoch_sparsity = {}
        for idx, mask_module in enumerate(self.mask_modules):
            mask, factor = mask_module.get_binary_mask()
            mask_sparsity = sum(mask.view(-1))
            param_usage += sum(mask.view(-1) * factor)
            epoch_sparsity["{:02d}".format(idx)] = mask_sparsity

        self.tb_sw.add_scalars("epoch_sparsity", epoch_sparsity, global_step=epoch)
        self.tb_sw.add_scalar("epoch_params", param_usage, global_step=epoch)

        self.tb_sw.add_scalars(
            "epoch",
            {
                "train_acc": train_res["top1_acc"],
                "train_task_loss": train_res["task_loss"],
                "train_kd_loss": train_res["kd_loss"],
                "train_mask_loss": train_res["mask_loss"],
                "train_admm_mask_loss": train_res["admm_mask_loss"],
                "eval_acc": eval_res["top1_acc"],
                "eval_task_loss": eval_res["task_loss"],
                "eval_kd_loss": eval_res["kd_loss"],
                "eval_mask_loss": eval_res["mask_loss"],
                "eval_admm_mask_loss": eval_res["admm_mask_loss"],
                "lr": self.lr,
                "elapsed_time": epoch_elapsed,
            },
            global_step=epoch,
        )
        self.t_log.append(
            [
                epoch,
                train_res["task_loss"],
                train_res["kd_loss"],
                train_res["mask_loss"],
                train_res["admm_mask_loss"],
                train_res["top1_acc"],
                eval_res["task_loss"],
                eval_res["kd_loss"],
                eval_res["mask_loss"],
                eval_res["admm_mask_loss"],
                eval_res["top1_acc"],
                self.lr,
            ]
        )
        self.logger.info(
            "FIN Epoch %(epoch)d/%(epochs)d LR: %(lr)f | "
            + "Train Task Loss: %(ttl).4f KDL: %(tkl).4f Mask Loss: %(tml).4f ADMM Loss: %(tadm).4f Acc: %(tacc).2f | "
            + "Eval Acc: %(eacc).2f | Params: %(params).2e | Difficulty: %(adiff).2f"
            + "Took %(dt).1fs (%(tdt).1fs)",
            {
                "epoch": epoch,
                "epochs": self.epochs,
                "lr": self.lr,
                "ttl": train_res["task_loss"],
                "tkl": train_res["kd_loss"],
                "tml": train_res["mask_loss"],
                "tadm": train_res["admm_mask_loss"],
                "tacc": train_res["top1_acc"],
                "eacc": eval_res["top1_acc"],
                "dt": epoch_elapsed,
                "params": param_usage,
                "adiff": self.adaptive_difficulty,
                "tdt": time() - self.exp_start,
            },
        )

        is_best_key = "{:.1e}".format(param_usage)
        prev_usage_best_acc = self.best_acc_per_usage.get(is_best_key, 0)
        usage_best_acc = max(eval_res["top1_acc"], prev_usage_best_acc)
        self.best_acc_per_usage[is_best_key] = usage_best_acc
        is_best = eval_res["top1_acc"] > prev_usage_best_acc

        state_dict = self.model.state_dict()
        if self.gpu_ids and len(self.gpu_ids) > 1:
            # unwrap the torch.nn.DataParallel
            state_dict = list(self.model.children())[0].state_dict()
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": state_dict,
                "acc": eval_res["top1_acc"],
                "best_acc_per_usage": self.best_acc_per_usage,
                "optim_state_dict": self.optimizer.state_dict(),
                "param_usage": param_usage,
            },
            is_best,
            checkpoint_dir=self.config["chkpt_dir"],
            filename="checkpoint-{}.pth.tar".format(is_best_key),
            best_filename="checkpoint-{}.pth.tar".format(is_best_key),
        )

    def finalize(self):
        self.logger.info("Best Acc per usage:")
        self.logger.info(pformat(self.best_acc_per_usage))
        self.tb_sw.close()
        self.t_log.close()

    @torch.no_grad()
    def calculate_model_sparsity(self):
        sparsity = []
        for module in self.model.modules():
            if type(module) == MaskSTE:
                mask, _mask_factor = module.get_binary_mask()
                layer_sparsity = sum(mask.view(-1))
                sparsity.append(layer_sparsity)
        return torch.stack(sparsity)

    @torch.no_grad()
    def calculate_model_parameters(self):
        parameters = []
        for module in self.model.modules():
            if type(module) == MaskSTE:
                mask, mask_factor = module.get_binary_mask()
                layer_parameters = sum(mask.view(-1) * mask_factor)
                parameters.append(layer_parameters)
        return torch.stack(parameters)
