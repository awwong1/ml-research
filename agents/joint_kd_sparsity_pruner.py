import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from time import time
from pprint import pformat

from .base import BaseAgent
from util.accuracy import calculate_accuracy
from util.adjust import adjust_learning_rate
from util.checkpoint import save_checkpoint
from util.cuda import set_cuda_devices
from util.losses import calculate_kd_loss
from util.meters import AverageMeter
from util.reflect import init_class, init_data
from util.seed import set_seed
from util.tablogger import TabLogger


class JointKnowledgeDistillationPruningAgent(BaseAgent):
    """Agent for unconstrained knowledge distillation & pruning experiments.
    """

    def __init__(self, config):
        super(JointKnowledgeDistillationPruningAgent, self).__init__(config)

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
        self.schedule = config.get("schedule", [150, 225])
        self.gamma = config.get("gamma", 0.1)
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
            "LR: %(lr)f decreasing by a factor of %(gamma)f at epochs %(schedule)s",
            {"lr": self.lr, "gamma": self.gamma, "schedule": self.schedule},
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
                "Train Acc",
                "Eval Task Loss",
                "Eval KD Loss",
                "Eval Mask Loss",
                "Eval Acc",
                "Num Parameters",
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
            for sched in self.schedule:
                if sched > self.start_epoch:
                    break
                new_lr = adjust_learning_rate(
                    self.optimizer,
                    sched,
                    lr=self.lr,
                    schedule=self.schedule,
                    gamma=self.gamma,
                )
                self.logger.info(
                    "LR fastforward from %(old)f to %(new)f at Epoch %(epoch)d",
                    {"old": self.lr, "new": new_lr, "epoch": sched},
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
        for epoch in range(self.start_epoch, self.epochs):
            new_lr = adjust_learning_rate(
                self.optimizer,
                epoch,
                lr=self.lr,
                schedule=self.schedule,
                gamma=self.gamma,
            )
            if new_lr != self.lr:
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

    def run_epoch_pass(self, epoch=-1, train=True):
        overall_loss = AverageMeter("Overall Loss")
        mask_meter = AverageMeter("Mask Loss")
        task_meter = AverageMeter("Task Loss")
        kd_meter = AverageMeter("KD Loss")
        acc1_meter = AverageMeter("Top 1 Acc")
        acc5_meter = AverageMeter("Top 5 Acc")

        self.model.train(train)
        dataloader = self.train_loader if train else self.eval_loader

        with tqdm(total=len(dataloader)) as t:
            for inputs, targets in dataloader:
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
                    outputs, teacher_outputs, temperature=self.temperature
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

                t.set_description(
                    "{mode} Epoch {epoch}/{epochs} ".format(
                        mode="Train" if train else "Eval",
                        epoch=epoch,
                        epochs=self.epochs,
                    )
                    + "Task Loss: {loss:.4f} | KD Loss: {kd:.4f} | Mask Loss: {ml:.4f} ".format(
                        loss=task_meter.avg, kd=kd_meter.avg, ml=mask_meter.avg
                    )
                    + "top1: {top1:.2f}% | top5: {top5:.2f}%".format(
                        top1=acc1_meter.avg, top5=acc5_meter.avg
                    )
                )
                t.update()

        return {
            "overall_loss": overall_loss.avg,
            "task_loss": task_meter.avg,
            "kd_loss": kd_meter.avg,
            "mask_loss": mask_meter.avg,
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
                "eval_acc": eval_res["top1_acc"],
                "eval_task_loss": eval_res["task_loss"],
                "eval_kd_loss": eval_res["kd_loss"],
                "eval_mask_loss": eval_res["mask_loss"],
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
                train_res["top1_acc"],
                eval_res["task_loss"],
                eval_res["kd_loss"],
                eval_res["mask_loss"],
                eval_res["top1_acc"],
                param_usage,
                self.lr
            ]
        )
        self.logger.info(
            "FIN Epoch %(epoch)d/%(epochs)d LR: %(lr)f | "
            + "Train Task Loss: %(ttl).4f KDL: %(tkl).4f Mask Loss: %(tml).4f Acc: %(tacc).2f | "
            + "Eval Acc: %(eacc).2f | Params: %(params).2e | "
            + "Took %(dt).1fs (%(tdt).1fs)",
            {
                "epoch": epoch,
                "epochs": self.epochs,
                "lr": self.lr,
                "ttl": train_res["task_loss"],
                "tkl": train_res["kd_loss"],
                "tml": train_res["mask_loss"],
                "tacc": train_res["top1_acc"],
                "eacc": eval_res["top1_acc"],
                "dt": epoch_elapsed,
                "params": param_usage,
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
