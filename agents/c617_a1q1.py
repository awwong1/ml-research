#!/usr/bin/env python3
import os
import random
import torch
from glob import glob
from PIL import Image
from tqdm import tqdm
from math import ceil
from torch.utils.tensorboard import SummaryWriter
from time import time

from .base import BaseAgent
from util.accuracy import calculate_binary_accuracy
from util.adjust import adjust_learning_rate
from util.checkpoint import save_checkpoint
from util.cuda import set_cuda_devices
from util.meters import AverageMeter
from util.reflect import init_class, init_data
from util.seed import set_seed


class PreprocessTileImage(BaseAgent):
    def __init__(self, config):
        super(PreprocessTileImage, self).__init__(config)
        self.tile_width = config.get("tile_width", 256)
        self.tile_height = config.get("tile_height", 256)
        self.input_glob = config.get("input_glob", "data/c617a1raw/Pictures/*.tif")
        self.output_dir = config.get("output_dir", "data/c617a1")
        self.train_ratio = config.get("train_ratio", 0.7)
        self.eval_ratio = config.get("eval_ratio", 0.2)
        self.test_ratio = config.get("test_ratio", 0.1)

        os.makedirs(os.path.join(self.output_dir, "pos"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "neg"), exist_ok=True)

    def run(self):
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

        # symlink the training, validation, and test sets according to ratios
        pos_glob = os.path.join(self.output_dir, "pos") + "/*.png"
        neg_glob = os.path.join(self.output_dir, "neg") + "/*.png"

        self.logger.info(pos_glob)
        self.logger.info(neg_glob)
        pos_images = set(glob(pos_glob))
        neg_images = set(glob(neg_glob))

        train_pos_len = ceil(len(pos_images) * self.train_ratio)
        eval_pos_len = ceil(len(pos_images) * self.eval_ratio)
        test_pos_len = len(pos_images) - train_pos_len - eval_pos_len
        self.logger.info(
            "pos train %d | eval %d | test %d | total %d ",
            train_pos_len,
            eval_pos_len,
            test_pos_len,
            len(pos_images),
        )

        train_neg_len = ceil(len(neg_images) * self.train_ratio)
        eval_neg_len = ceil(len(neg_images) * self.eval_ratio)
        test_neg_len = len(neg_images) - train_neg_len - eval_neg_len
        self.logger.info(
            "neg train %d | eval %d | test %d | total %d ",
            train_neg_len,
            eval_neg_len,
            test_neg_len,
            len(neg_images),
        )

        # make directories
        os.makedirs(os.path.join(self.output_dir, "train", "pos"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "train", "neg"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "eval", "pos"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "eval", "neg"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "test", "pos"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "test", "neg"), exist_ok=True)

        train_pos_images = random.sample(pos_images, train_pos_len)
        pos_images.difference_update(train_pos_images)
        eval_pos_images = random.sample(pos_images, eval_pos_len)
        pos_images.difference_update(eval_pos_images)
        test_pos_images = random.sample(pos_images, test_pos_len)
        pos_images.difference_update(test_pos_images)
        assert len(pos_images) == 0

        train_neg_images = random.sample(neg_images, train_neg_len)
        neg_images.difference_update(train_neg_images)
        eval_neg_images = random.sample(neg_images, eval_neg_len)
        neg_images.difference_update(eval_neg_images)
        test_neg_images = random.sample(neg_images, test_neg_len)
        neg_images.difference_update(test_neg_images)
        assert len(neg_images) == 0

        for img in train_pos_images:
            basename = os.path.basename(img)
            os.symlink(
                os.path.abspath(os.path.join(self.output_dir, "pos", basename)),
                os.path.join(self.output_dir, "train", "pos", basename),
            )
        for img in eval_pos_images:
            basename = os.path.basename(img)
            os.symlink(
                os.path.abspath(os.path.join(self.output_dir, "pos", basename)),
                os.path.join(self.output_dir, "eval", "pos", basename),
            )
        for img in test_pos_images:
            basename = os.path.basename(img)
            os.symlink(
                os.path.abspath(os.path.join(self.output_dir, "pos", basename)),
                os.path.join(self.output_dir, "test", "pos", basename),
            )

        for img in train_neg_images:
            basename = os.path.basename(img)
            os.symlink(
                os.path.abspath(os.path.join(self.output_dir, "neg", basename)),
                os.path.join(self.output_dir, "train", "neg", basename),
            )
        for img in eval_neg_images:
            basename = os.path.basename(img)
            os.symlink(
                os.path.abspath(os.path.join(self.output_dir, "neg", basename)),
                os.path.join(self.output_dir, "eval", "neg", basename),
            )
        for img in test_neg_images:
            basename = os.path.basename(img)
            os.symlink(
                os.path.abspath(os.path.join(self.output_dir, "neg", basename)),
                os.path.join(self.output_dir, "test", "neg", basename),
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
        self.train_set, self.train_loader = init_data(config.get("train_data"))
        self.eval_set, self.eval_loader = init_data(config.get("eval_data"))

        # Instantiate Model
        self.model = init_class(config.get("model"))
        # manually convert pretrained model into a binary classification problem
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True), torch.nn.Linear(1280, 1)
        )
        # self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)
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
            torch.nn.Dropout(p=0.2, inplace=True), torch.nn.Linear(1280, 1)
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
            for inputs, targets in t:
                batch_size = inputs.size(0)
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

                # Compute forward pass of the model
                outputs = self.model(inputs)

                # Record task loss and accuracies
                prec1 = calculate_binary_accuracy(outputs.data, targets.data)
                acc1_meter.update(prec1.item(), batch_size)

                t.set_description("Test Acc: {top1:.2f}%".format(top1=acc1_meter.avg))
            self.logger.info("Test Acc: %.2f", acc1_meter.avg)

    def finalize(self):
        self.tb_sw.close()
