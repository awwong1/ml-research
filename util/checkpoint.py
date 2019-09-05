import shutil
import torch
from os import path


def save_checkpoint(
    state,
    is_best,
    checkpoint_dir="",
    filename="checkpoint.pth.tar",
    best_filename="model_best.pth.tar",
):
    checkpoint_fp = path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_fp)
    if is_best and filename != best_filename:
        best_checkpoint_fp = path.join(checkpoint_dir, best_filename)
        shutil.copyfile(checkpoint_fp, best_checkpoint_fp)
