import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from time import time

from .base import BaseAgent
from util.accuracy import calculate_accuracy
from util.adjust import adjust_learning_rate
from util.checkpoint import save_checkpoint
from util.cuda import set_cuda_devices
from util.meters import AverageMeter
from util.reflect import init_class, init_data
from util.seed import set_seed
from util.tablogger import TabLogger


class ClassificationAgent(BaseAgent):
    """Agent for generic PyTorch classification experiments.
    """

    def __init__(self, config):
        super(ClassificationAgent, self).__init__(config)

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

        # Instantiate Model
        self.model = init_class(config.get("model"))
        try:
            # Try to visualize tensorboard model graph structure
            model_input, _target = next(iter(self.eval_set))
            self.tb_sw.add_graph(self.model, model_input.unsqueeze(0))
        except Exception as e:
            self.logger.warn(e)

        # Instantiate task loss and optimizer
        self.task_loss_fn = init_class(config.get("task_loss"))
        self.optimizer = init_class(config.get("optimizer"), self.model.parameters())

        # Misc. Other classification hyperparameters
        self.epochs = config.get("epochs", 300)
        self.start_epoch = config.get("start_epoch", 0)
        self.schedule = config.get("schedule", [150, 225])
        self.gamma = config.get("gamma", 0.1)
        self.lr = self.optimizer.param_groups[0]["lr"]
        self.best_acc1 = 0

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
        self.t_log = TabLogger(t_log_fpath, resume=bool(resume))  # tab logger
        self.t_log.set_names(
            [
                "Epoch",
                "Train Task Loss",
                "Train Acc",
                "Eval Task Loss",
                "Eval Acc",
                "LR",
            ]
        )
        if resume:
            self.logger.info("Resuming from checkpoint: %s", resume)
            res_chkpt = torch.load(resume)
            self.model.load_state_dict(res_chkpt["state_dict"])
            self.start_epoch = res_chkpt.get("epoch", 0)
            self.best_acc1 = res_chkpt.get("best_acc1", 0)
            optim_state_dict = res_chkpt.get("optim_state_dict")
            if optim_state_dict:
                self.optimizer.load_state_dict(optim_state_dict)
            self.logger.info(
                "Resumed at epoch %d, eval best_acc1 %.2f",
                self.start_epoch,
                self.best_acc1,
            )
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
                self.model = torch.nn.DataParallel(self.model).cuda()
            else:
                self.model = self.model.cuda()

    def run(self):
        self.exp_start = time()
        if self.epochs == 0:
            # no training, just do an evaluation pass
            with torch.no_grad():
                eval_res = self.run_epoch_pass(epoch=0, train=False)
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
        task_meter = AverageMeter("Task Loss")
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

                # Record task loss and accuracies
                task_loss = self.task_loss_fn(outputs, targets)
                task_meter.update(task_loss.data.item(), batch_size)
                prec1, prec5 = calculate_accuracy(
                    outputs.data, targets.data, topk=(1, 5)
                )
                acc1_meter.update(prec1.item(), batch_size)
                acc5_meter.update(prec5.item(), batch_size)

                if train:
                    self.optimizer.zero_grad()
                    task_loss.backward()
                    self.optimizer.step()

                t.set_description(
                    "{mode} Epoch {epoch}/{epochs} ".format(
                        mode="Train" if train else "Eval",
                        epoch=epoch,
                        epochs=self.epochs,
                    )
                    + "Task Loss: {loss:.4f} | ".format(loss=task_meter.avg)
                    + "top1: {top1:.2f}% | top5: {top5:.2f}%".format(
                        top1=acc1_meter.avg, top5=acc5_meter.avg
                    )
                )
                t.update()

        return {
            "task_loss": task_meter.avg,
            "top1_acc": acc1_meter.avg,
            "top5_acc": acc5_meter.avg,
        }

    def log_epoch_info(self, epoch, train_res, eval_res, epoch_elapsed):
        self.tb_sw.add_scalars(
            "epoch",
            {
                "train_loss": train_res["task_loss"],
                "train_acc": train_res["top1_acc"],
                "eval_loss": eval_res["task_loss"],
                "eval_acc": eval_res["top1_acc"],
                "lr": self.lr,
                "elapsed_time": epoch_elapsed,
            },
            global_step=epoch,
        )
        self.t_log.append(
            [
                epoch,
                train_res["task_loss"],
                train_res["top1_acc"],
                eval_res["task_loss"],
                eval_res["top1_acc"],
                self.lr
            ]
        )
        self.logger.info(
            "FIN Epoch %(epoch)d/%(epochs)d LR: %(lr)f | "
            + "Train Loss: %(tloss).4f Acc: %(tacc).2f | "
            + "Eval Loss: %(eloss).4f Acc: %(eacc).2f | "
            + "Took %(dt).1fs (%(tdt).1fs)",
            {
                "epoch": epoch,
                "epochs": self.epochs,
                "lr": self.lr,
                "tloss": train_res["task_loss"],
                "tacc": train_res["top1_acc"],
                "eloss": eval_res["task_loss"],
                "eacc": eval_res["top1_acc"],
                "dt": epoch_elapsed,
                "tdt": time() - self.exp_start,
            },
        )

        is_best = eval_res["top1_acc"] > self.best_acc1
        self.best_acc1 = max(eval_res["top1_acc"], self.best_acc1)
        state_dict = self.model.state_dict()
        if self.gpu_ids and len(self.gpu_ids) > 1:
            # unwrap the torch.nn.DataParallel
            state_dict = list(self.model.children())[0].state_dict()
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": state_dict,
                "acc": eval_res["top1_acc"],
                "best_acc1": self.best_acc1,
                "optim_state_dict": self.optimizer.state_dict(),
            },
            is_best,
            checkpoint_dir=self.config["chkpt_dir"],
        )

    def finalize(self):
        self.logger.info("Best Acc: %.1f", self.best_acc1)
        self.tb_sw.close()
        self.t_log.close()
