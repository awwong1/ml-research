#!/usr/bin/env python3
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from time import time
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    ToTensor,
    Normalize,
    Lambda,
    Resize,
    ToPILImage,
)
from torchvision.datasets import ImageFolder
from skimage.transform import pyramid_gaussian

from .base import BaseAgent
from util.accuracy import calculate_binary_accuracy
from util.adjust import adjust_learning_rate
from util.checkpoint import save_checkpoint
from util.cuda import set_cuda_devices
from util.meters import AverageMeter
from util.reflect import init_class
from util.seed import set_seed


def expand_multires(img_tensor):
    # get the original image dimensions
    _, height, width = img_tensor.size()
    # convert (CxHxW) to (HxWxC)
    image = img_tensor.permute(1, 2, 0)
    # construct the image pyramid
    pyramid = pyramid_gaussian(image, multichannel=True)

    resized_multires = []
    for raw_np_img in pyramid:
        raw_tensor = ToTensor()(raw_np_img)
        pil_img = ToPILImage()(raw_tensor.float())
        res_pil_img = Resize((height, width))(pil_img)
        res_tensor = ToTensor()(res_pil_img)
        resized_multires.append(res_tensor)
    multires = torch.stack(resized_multires, dim=3)
    # multires: (channel, height, width, resolution_dimension)
    return multires


class MultiResolutionModelWrapper(torch.nn.Module):
    """Wrapper model for handling multiple resolution images
    """

    def __init__(self, base_model, resolution_dimensions=9):
        super(MultiResolutionModelWrapper, self).__init__()
        self.resolution_dimensions = resolution_dimensions

        self.base_model = base_model
        # takes number of resolution_dimensions, outputs single value
        self.dim_learner = torch.nn.Linear(resolution_dimensions, 1)

    def forward(self, x):
        """x = (B, C, H, W, D)
        B: batch dimension
        C: image channels (RGB)
        H: height
        W: width
        D: number of resolution dimensions
        """
        # split each resolution dimension into its own tensor
        res_xs = x.chunk(self.resolution_dimensions, dim=-1)
        res_preds = []
        for res_x in res_xs:
            res_pred = self.base_model(res_x.squeeze(dim=-1))
            res_preds.append(res_pred)
        # combine each resolution dimension back into a single tensor
        multires_preds = torch.cat(res_preds, dim=-1)
        out = self.dim_learner(multires_preds)
        return out


class MultiResolutionFineTuneClassifier(BaseAgent):
    def __init__(self, config):
        super(MultiResolutionFineTuneClassifier, self).__init__(config)
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
        # self.train_set, self.train_loader = init_data(config.get("train_data"))
        # self.eval_set, self.eval_loader = init_data(config.get("eval_data"))
        base_transforms = [
            ToTensor(),
            Normalize(
                [0.6705237030982971, 0.6573456525802612, 0.6612830758094788],
                [0.19618913531303406, 0.22384527325630188, 0.23168295621871948],
            ),
            Lambda(expand_multires),
        ]
        self.train_set = ImageFolder(
            "data/c617a1/train",
            transform=Compose([RandomHorizontalFlip()] + base_transforms),
        )
        self.train_loader = DataLoader(
            self.train_set,
            num_workers=config.get("num_workers", 0),
            batch_size=config.get("batch_size", 128),
            shuffle=True,
        )
        self.eval_set = ImageFolder(
            "data/c617a1/eval", transform=Compose(base_transforms)
        )
        self.eval_loader = DataLoader(
            self.eval_set,
            num_workers=config.get("num_workers", 0),
            batch_size=config.get("batch_size", 128),
        )

        # Instantiate Model
        base_model = init_class(config.get("model"))
        # manually convert pretrained model into a binary classification problem
        base_model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True), torch.nn.Linear(1280, 1)
        )
        self.model = MultiResolutionModelWrapper(base_model)

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
                eval_res = self.run_epoch_pass(epoch=0, mode="Eval")
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
            train_res = self.run_epoch_pass(epoch=epoch, mode="Train")
            with torch.no_grad():
                eval_res = self.run_epoch_pass(epoch=epoch, mode="Eval")
            epoch_elapsed = time() - epoch_start

            self.log_epoch_info(epoch, train_res, eval_res, epoch_elapsed)

    def run_epoch_pass(self, epoch=-1, mode="Train"):
        task_meter = AverageMeter("Task Loss")
        acc1_meter = AverageMeter("Top 1 Acc")

        self.model.train(True if mode == "Train" else False)
        if mode == "Train":
            dataloader = self.train_loader
        elif mode == "Eval":
            dataloader = self.eval_loader

        t = tqdm(dataloader)
        for inputs, targets in t:
            batch_size = inputs.size(0)
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            # Compute forward pass of the model
            outputs = self.model(inputs)

            # Record task loss and accuracies
            task_loss = self.task_loss_fn(outputs.view(-1), targets.float())
            task_meter.update(task_loss.data.item(), batch_size)
            prec1 = calculate_binary_accuracy(outputs.data, targets.data)
            acc1_meter.update(prec1.item(), batch_size)

            if mode == "Train":
                self.optimizer.zero_grad()
                task_loss.backward()
                self.optimizer.step()

            t.set_description(
                "{mode} Epoch {epoch}/{epochs} ".format(
                    mode=mode, epoch=epoch, epochs=self.epochs
                )
                + "Task Loss: {loss:.4f} | ".format(loss=task_meter.avg)
                + "Acc: {top1:.2f}%".format(top1=acc1_meter.avg)
            )

        return {"task_loss": task_meter.avg, "top1_acc": acc1_meter.avg}

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
        self.tb_sw.close()
