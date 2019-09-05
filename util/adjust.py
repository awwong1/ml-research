def adjust_learning_rate(optimizer, step, lr=0.1, schedule=[], gamma=1):
    if step in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    return lr
