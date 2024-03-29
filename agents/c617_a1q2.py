#!/usr/bin/env python3
import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from time import time
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    ToTensor,
    Normalize,
    Lambda,
    ToPILImage,
)
from torchvision.datasets import ImageFolder
from skimage.transform import pyramid_gaussian
from math import ceil
from glob import glob
from collections import Counter

from .base import BaseAgent
from util.accuracy import calculate_accuracy
from util.adjust import adjust_learning_rate
from util.checkpoint import save_checkpoint
from util.cuda import set_cuda_devices
from util.meters import AverageMeter
from util.reflect import init_class
from util.seed import set_seed


def expand_multires(img_tensor, keep=3):
    # get the original image dimensions
    _, height, width = img_tensor.size()
    # convert (CxHxW) to (HxWxC)
    image = img_tensor.permute(1, 2, 0)
    # construct the image pyramid
    pyramid = pyramid_gaussian(image, multichannel=True)

    multi_res = []
    for idx, raw_np_img in enumerate(pyramid):
        if idx >= keep:
            break
        raw_tensor = ToTensor()(raw_np_img)
        pil_img = ToPILImage()(raw_tensor.float())
        # pil_img = Resize((height, width))(pil_img)
        res_tensor = ToTensor()(pil_img)
        multi_res.append(res_tensor)
    # multires: (channel, height, width, resolution_dimension)
    # return torch.stack(multi_res, dim=3)
    return multi_res


class MultiResolutionModelWrapper(torch.nn.Module):
    """Wrapper model for handling multiple resolution images
    """

    def __init__(self, base_model, num_resolutions=3, base_out_size=300):
        super(MultiResolutionModelWrapper, self).__init__()
        self.num_resolutions = num_resolutions
        self.base_model = base_model
        self.dim_learner = torch.nn.Sequential(
            torch.nn.Tanh(), torch.nn.Linear(base_out_size * num_resolutions, 2)
        )

    def forward(self, multi_x):
        """x = (B, C, H, W)
        B: batch dimension
        C: image channels (RGB)
        H: height
        W: width
        """
        res_preds = []
        assert self.num_resolutions == len(
            multi_x
        ), "mismatched number of image resolutions"
        for x in multi_x:
            res_pred = self.base_model(x)
            res_preds.append(res_pred)
        # combine each resolution dimension back into a single tensor
        # multires_preds = torch.cat(res_preds, dim=-1)
        pre_out = torch.stack(res_preds, dim=-1)
        out = self.dim_learner(pre_out.flatten(start_dim=1))
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
        ds = ImageFolder(
            "data/c617a1/train_eval",
            transform=Compose(
                [
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Normalize(
                        [0.663295328617096, 0.6501832604408264, 0.6542291045188904],
                        [0.19360290467739105, 0.22194330394268036, 0.23059576749801636],
                    ),
                    Lambda(expand_multires),
                ]
            ),
        )
        train_set_ratio, eval_set_ratio = config.get(
            "train_eval_split_ratio", [0.85, 0.15]
        )
        train_len = ceil(len(ds) * train_set_ratio)
        eval_len = len(ds) - train_len
        self.train_set, self.eval_set = random_split(ds, [train_len, eval_len])
        self.train_loader = DataLoader(
            self.train_set,
            num_workers=config.get("num_workers", 0),
            batch_size=config.get("batch_size", 128),
            shuffle=True
        )
        self.eval_loader = DataLoader(
            self.eval_set, 
            num_workers=config.get("num_workers", 0),
            batch_size=config.get("batch_size", 128),
            shuffle=True
        )

        # Instantiate Model
        base_model = init_class(config.get("model"))
        # Freeze all of the parameters (except for final classification layer which we add afterwards)
        for param in base_model.parameters():
            param.requires_grad = False

        # manually convert pretrained model into a binary classification problem
        base_out_size = 300
        base_model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True), torch.nn.Linear(1280, base_out_size)
        )
        self.model = MultiResolutionModelWrapper(base_model, base_out_size=base_out_size)

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
            batch_size = inputs[0].size(0)
            if self.use_cuda:
                inputs = [inp.cuda() for inp in inputs]
                targets = targets.cuda(non_blocking=True)

            # Compute forward pass of the model
            outputs = self.model(inputs)

            # update targets to match resolution dimensionality
            # resolution_dimensions = outputs.size()[-1]
            # target_vals = torch.zeros(
            #     (batch_size, resolution_dimensions),
            #     dtype=targets.dtype,
            #     layout=targets.layout,
            #     device=targets.device,
            # )
            # target_vals.index_fill_(0, targets, 1)

            # Record task loss and accuracies
            task_loss = self.task_loss_fn(outputs, targets)
            task_meter.update(task_loss.data.item(), batch_size)
            # prec1, = calculate_multires_accuracy(outputs.data, targets.data)
            prec1, = calculate_accuracy(outputs.data, targets.data)
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


class TestSetEvaluator(BaseAgent):
    def __init__(self, config):
        super(TestSetEvaluator, self).__init__(config)

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
        # self.test_set, self.test_loader = init_data(config.get("test_data"))
        base_transforms = [
            ToTensor(),
            Normalize(
                [0.663295328617096, 0.6501832604408264, 0.6542291045188904],
                [0.19360290467739105, 0.22194330394268036, 0.23059576749801636],
            ),
            Lambda(expand_multires),
        ]
        self.test_set = ImageFolder(
            "data/c617a1/test", transform=Compose(base_transforms)
        )
        self.test_loader = DataLoader(
            self.test_set,
            num_workers=config.get("num_workers", 0),
            batch_size=config.get("batch_size", 1),
            shuffle=True,
        )

        # Instantiate Model
        base_model = init_class(config.get("model"))
        # manually convert pretrained model into a binary classification problem
        base_out_size = 300
        base_model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True), torch.nn.Linear(1280, base_out_size)
        )
        self.model = MultiResolutionModelWrapper(base_model, base_out_size=base_out_size)

        self.logger.info("Test Dataset: %s", self.test_set)
        self.checkpoints = config.get("checkpoints")
        self.map_location = None if self.use_cuda else torch.device("cpu")
        # Support multiple GPUs using DataParallel
        if self.use_cuda:
            if len(self.gpu_ids) > 1:
                self.model = torch.nn.DataParallel(self.model).cuda()
            else:
                self.model = self.model.cuda()

    def run(self):
        for checkpoint_path in self.checkpoints:
            self.logger.info("Loading '%s'", checkpoint_path)
            chkpt = torch.load(checkpoint_path, map_location=self.map_location)
            self.logger.info(
                "Epoch %d, eval_acc: %.2f, best_eval_acc: %.2f",
                chkpt["epoch"],
                chkpt["acc"],
                chkpt["best_acc1"],
            )
            self.model.load_state_dict(chkpt["state_dict"])
            acc1_meter = AverageMeter("Top 1 Acc")

            self.model.eval()

            t = tqdm(self.test_loader)
            test_accuracies = []

            for inputs, targets in t:
                batch_size = inputs[0].size(0)
                if self.use_cuda:
                    inputs = [inp.cuda() for inp in inputs]
                    targets = targets.cuda(non_blocking=True)

                # Compute forward pass of the model
                outputs = self.model(inputs)

                # Record task loss and accuracies
                prec1, = calculate_accuracy(outputs.data, targets.data)
                acc1_meter.update(prec1.item(), batch_size)
                test_accuracies.append(prec1.item())

                t.set_description("Test Acc: {top1:.2f}%".format(top1=acc1_meter.avg))
            self.logger.info("Test Acc: %.2f", acc1_meter.avg)

            self.logger.info("Avg Test Tile Acc: %.2f", acc1_meter.avg)
            file_names = list(glob(os.path.join("data/c617a1/test", "**", "*.png")))
            assert len(test_accuracies) == len(
                file_names
            ), "accuracy length does not match files"
            base_file_names = [fname.rsplit("-", 1)[-1] for fname in file_names]
            img_accuracies = {}
            for base_name, accuracy in zip(base_file_names, test_accuracies):
                img_acc = img_accuracies.get(base_name, [])
                img_acc.append(accuracy)
                img_accuracies[base_name] = img_acc

            for key, value in img_accuracies.items():
                self.logger.info(
                    "%s: %.2f [%s]",
                    key,
                    float(sum(value)) / len(value),
                    str(Counter(value)),
                )

    def finalize(self):
        self.tb_sw.close()
