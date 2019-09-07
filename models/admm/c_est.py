import torch


class ConstraintEstimatorWidthRescale(torch.nn.Module):
    def __init__(self, scales):
        super(ConstraintEstimatorWidthRescale, self).__init__()
        self.scales = torch.nn.Parameter(
            torch.Tensor(scales, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x):
        assert x.dim() != 1
        x = x / self.scales
        return torch.cat(
            [
                (x[:, 0].detach() * x[:, 1]).unsqueeze(1),
                x[:, 1:-2] * x[:, 2:-1],
                (x[:, -2] * x[:, -1].detach()).unsqueeze(1),
            ],
            dim=1,
        )


class ConstraintEstimatorNet(torch.nn.Module):
    """Bilinear model for estimating constraint
    """

    def __init__(self, n_nodes=None, preprocessor=lambda x: x):
        super(ConstraintEstimatorNet, self).__init__()
        if n_nodes is None:
            n_nodes = [8, 1]
        self.islinear = len(n_nodes) == 2

        layers = []
        for i, _ in enumerate(n_nodes):
            if i < len(n_nodes) - 1:
                layer = torch.nn.Linear(n_nodes[i], n_nodes[i + 1], bias=True)
                if len(n_nodes) == 2:
                    layer.weight.data.zero_()
                    layer.bias.data.zero_()
                layers.append(layer)
                if i < len(n_nodes) - 2:
                    layers.append(torch.nn.SELU())
        self.regressor = torch.nn.Sequential(*layers)

    def forward(self, x):
        single_data = x.dim() == 1
        if single_data:
            x = x.unsqueeze(0)
        res = self.regressor(self.preprocessor(x))
        if single_data:
            res = res.squeeze(0)
        return res
