import torch


def calculate_kd_loss(outputs, teacher_outputs, targets, temperature=4):
    """Calculate Knowledge Distillation Loss"""
    p = torch.nn.functional.log_softmax(
        outputs.div(temperature), dim=1)
    q = torch.nn.functional.softmax(
        teacher_outputs.div(temperature), dim=1)
    return torch.nn.functional.kl_div(p, q, reduction="sum").mul(
        temperature**2).div(outputs.shape[0])
