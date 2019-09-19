import torch


@torch.no_grad()
def calculate_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@torch.no_grad()
def calculate_multires_accuracy(output, targets, topk=(1,)):
    """Computes the accuracy for the k top predictions for the specified values of k.
    Support multi resolution model output"""
    assert topk == (1,), "only top 1 correct currently supported"
    maxk = max(topk)
    # batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred.squeeze_()

    correct = []
    for batch_idx, target in enumerate(targets):
        res_pred, counts = torch.unique(pred[batch_idx], return_counts=True)
        correct.append((res_pred[counts.argmax()] == target).float())
    return (sum(correct).float().div(len(correct)).mul(100), )
    # res = []
    # for k in topk:
    #     correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    #     res.append(correct_k.mul_(100.0 / batch_size))
    # res.append(correct.sum().float().div(correct.numel()).mul(100))
    # return res

@torch.no_grad()
def calculate_binary_accuracy(output, target, threshold=0.5, apply_sigmoid=True):
    """Compute the accuracy for binary classification where
        target is either 0, 1 and output is a tensor of same size with arbitrary values"""
    if apply_sigmoid:
        check = torch.nn.Sigmoid()(output)
    else:
        check = output
    pred_y = (check >= threshold).view(-1) == target.view(-1).bool()
    return (pred_y.sum().float() / pred_y.numel()) * 100
