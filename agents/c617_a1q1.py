#!/usr/bin/env python3
import os
import random
import torch
from glob import glob
from PIL import Image
from tqdm import tqdm
from math import ceil
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose
from time import time
from collections import Counter

from .base import BaseAgent
from util.accuracy import calculate_accuracy
from util.adjust import adjust_learning_rate
from util.checkpoint import save_checkpoint
from util.cuda import set_cuda_devices
from util.meters import AverageMeter
from util.reflect import init_class, fetch_class, init_data
from util.seed import set_seed
from util.tablogger import TabLogger


class PreprocessTileImage(BaseAgent):
    def __init__(self, config):
        super(PreprocessTileImage, self).__init__(config)
        self.tile_width = config.get("tile_width", 256)
        self.tile_height = config.get("tile_height", 256)
        self.input_glob = config.get("input_glob", "data/c617a1raw/Pictures/*.tif")
        self.output_dir = config.get("output_dir", "data/c617a1")
        self.train_ratio = config.get("train_ratio", 0.8)
        self.eval_ratio = config.get("eval_ratio", 0.1)
        self.test_ratio = config.get("test_ratio", 0.1)

        os.makedirs(os.path.join(self.output_dir, "pos"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "neg"), exist_ok=True)

    def run(self):
        out_glob = os.path.join(self.output_dir, "**", "*.png")
        if len(set(glob(out_glob))):
            self.logger.info("out glob exists, skipping raw image tiling")
        else:
            self.tile_all_images()
        self.partition_datasets()
        # log the number of train/eval and test image tiles
        train_eval_pos = list(
            glob(os.path.join(self.output_dir, "train_eval", "pos", "*.png"))
        )
        train_eval_neg = list(
            glob(os.path.join(self.output_dir, "train_eval", "neg", "*.png"))
        )
        test_pos = list(glob(os.path.join(self.output_dir, "test", "pos", "*.png")))
        test_neg = list(glob(os.path.join(self.output_dir, "test", "neg", "*.png")))
        self.logger.info("test: [pos %d, neg %d]", len(test_pos), len(test_neg))
        self.logger.info(
            "train_eval: [pos %d, neg %d]", len(train_eval_pos), len(train_eval_neg)
        )
        train_multiplier = self.train_ratio + (self.test_ratio / 2)
        train_pos = ceil(len(train_eval_pos) * train_multiplier)
        train_neg = ceil(len(train_eval_neg) * train_multiplier)
        self.logger.info("train: [pos %d, neg %d]", train_pos, train_neg)
        self.logger.info(
            "eval: [pos %d, neg %d]",
            len(train_eval_pos) - train_pos,
            len(train_eval_neg) - train_neg,
        )

    def tile_all_images(self):
        # Iterate through all the input images
        for infile in tqdm(glob(self.input_glob)):
            filename, ext = os.path.splitext(infile)
            basename = os.path.basename(filename)
            datatype = "pos" if "pos" in basename else "neg"
            im = Image.open(infile)

            im_x, im_y = im.size[0], im.size[1]
            cur_x, cur_y = 0, 0

            # calculate patches
            num_x_steps = ceil(im_x / self.tile_width) - 1
            num_y_steps = ceil(im_y / self.tile_height) - 1
            with tqdm(total=num_x_steps * num_y_steps) as pbar:
                for cur_x_idx in range(0, num_x_steps):
                    cur_x = cur_x_idx * self.tile_width
                    if cur_x >= im_x:
                        cur_x = im_x - self.tile_width
                    for cur_y_idx in range(0, num_y_steps):
                        cur_y = cur_y_idx * self.tile_height
                        if cur_y >= im_y:
                            cur_y = im_y - self.tile_height

                        crop = im.crop(
                            (
                                cur_x,
                                cur_y,
                                cur_x + self.tile_width,
                                cur_y + self.tile_height,
                            )
                        )
                        tilename = "x{}-y{}-{}.png".format(cur_x, cur_y, basename)
                        tile_outfile = os.path.join(self.output_dir, datatype, tilename)
                        crop.save(tile_outfile, "PNG")

                        pbar.set_description(tile_outfile)
                        pbar.update()

    def partition_datasets(self):
        # test set needs to contain whole images
        # split the dataset of pictures into training/evaluation/test sets
        all_images = set(glob(self.input_glob))
        num_train_images = ceil(len(all_images) * self.train_ratio)
        num_eval_images = ceil(len(all_images) * self.eval_ratio)
        num_test_images = len(all_images) - num_train_images - num_eval_images

        # split out the test images
        while True:
            test_images = random.sample(all_images, num_test_images)
            test_pos_images = [img for img in test_images if "pos" in img]
            test_neg_images = [img for img in test_images if "neg" in img]
            if len(test_pos_images) == 5 and len(test_neg_images) == 5:
                break
                # all_images.difference_update(test_images)
        self.logger.info("test images (pos): %s", ", ".join(test_pos_images))
        self.logger.info("test images (neg): %s", ", ".join(test_neg_images))

        # symlink the training, validation, and test sets according to ratios
        pos_glob = os.path.join(self.output_dir, "pos") + "/*.png"
        neg_glob = os.path.join(self.output_dir, "neg") + "/*.png"

        # make directories
        os.makedirs(os.path.join(self.output_dir, "train_eval", "pos"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "train_eval", "neg"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "test", "pos"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "test", "neg"), exist_ok=True)

        pos_test_basenames = [
            os.path.basename(os.path.splitext(test_img)[0])
            for test_img in test_pos_images
        ]
        for infile in tqdm(glob(pos_glob)):
            filename, ext = os.path.splitext(infile)
            basename = os.path.basename(infile)
            is_test_img = (
                len(
                    [
                        test_img
                        for test_img in pos_test_basenames
                        if basename.endswith("-{}.png".format(test_img))
                    ]
                )
                > 0
            )
            if is_test_img:
                sym_out = os.path.join(self.output_dir, "test", "pos", basename)
            else:
                sym_out = os.path.join(self.output_dir, "train_eval", "pos", basename)
            os.symlink(
                os.path.abspath(os.path.join(self.output_dir, "pos", basename)), sym_out
            )

        neg_test_basenames = [
            os.path.basename(os.path.splitext(test_img)[0])
            for test_img in test_neg_images
        ]
        for infile in tqdm(glob(neg_glob)):
            filename, ext = os.path.splitext(infile)
            basename = os.path.basename(infile)
            is_test_img = (
                len(
                    [
                        test_img
                        for test_img in neg_test_basenames
                        if basename.endswith("-{}.png".format(test_img))
                    ]
                )
                > 0
            )
            if is_test_img:
                sym_out = os.path.join(self.output_dir, "test", "neg", basename)
            else:
                sym_out = os.path.join(self.output_dir, "train_eval", "neg", basename)
            os.symlink(
                os.path.abspath(os.path.join(self.output_dir, "neg", basename)), sym_out
            )


class FineTuneClassifier(BaseAgent):
    """Agent for fine tuning CMPUT617 Assignment 1 binary classifier
    """

    def __init__(self, config):
        super(FineTuneClassifier, self).__init__(config)

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
        # self.train_set, self.train_loader = init_data(config.get("train_eval_data"))
        train_eval_config = config.get("train_eval_data")
        ds_class = fetch_class(train_eval_config["name"])
        d_transform = list(map(init_class, train_eval_config.get("transform", [])))
        d_ttransform = list(
            map(init_class, train_eval_config.get("target_transform", []))
        )
        ds = ds_class(
            *train_eval_config.get("args", []),
            **train_eval_config.get("kwargs", {}),
            transform=Compose(d_transform) if d_transform else None,
            target_transform=Compose(d_ttransform) if d_ttransform else None
        )
        train_set_ratio, eval_set_ratio = train_eval_config.get(
            "train_eval_split_ratio", [0.85, 0.15]
        )
        train_len = ceil(len(ds) * train_set_ratio)
        eval_len = len(ds) - train_len
        self.train_set, self.eval_set = random_split(ds, [train_len, eval_len])
        self.train_loader = DataLoader(
            self.train_set, **train_eval_config.get("dataloader_kwargs", {})
        )
        self.eval_loader = DataLoader(
            self.eval_set, **train_eval_config.get("dataloader_kwargs", {})
        )

        # Instantiate Model
        self.model = init_class(config.get("model"))
        # Freeze all of the parameters (except for final classification layer which we add afterwards)
        for param in self.model.parameters():
            param.requires_grad = False
        # manually convert pretrained model into a binary classification problem
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True), torch.nn.Linear(1280, 2)
        )
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
        self.map_location = None if self.use_cuda else torch.device("cpu")
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
            res_chkpt = torch.load(resume, map_location=self.map_location)
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
            task_loss = self.task_loss_fn(outputs, targets)
            task_meter.update(task_loss.data.item(), batch_size)
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
        self.t_log.append(
            [
                epoch,
                train_res["task_loss"],
                train_res["top1_acc"],
                eval_res["task_loss"],
                eval_res["top1_acc"],
                self.lr,
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
        self.tb_sw.close()
        self.t_log.close()


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
        self.test_set, self.test_loader = init_data(config.get("test_data"))

        # Instantiate Model
        self.model = init_class(config.get("model"))
        # manually convert pretrained model into a binary classification problem
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True), torch.nn.Linear(1280, 2)
        )

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
                batch_size = inputs.size(0)
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

                # Compute forward pass of the model
                outputs = self.model(inputs)

                # Record task loss and accuracies
                prec1, = calculate_accuracy(outputs.data, targets.data)
                acc1_meter.update(prec1.item(), batch_size)
                test_accuracies.append(prec1.item())

                t.set_description("Test Acc: {top1:.2f}%".format(top1=acc1_meter.avg))
            self.logger.info("Avg Test Tile Acc: %.2f", acc1_meter.avg)
            img_path = self.config.get("test_data", {}).get("args", [])[0]
            file_names = list(glob(os.path.join(img_path, "**", "*.png")))
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
